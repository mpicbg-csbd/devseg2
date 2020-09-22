import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys
import ipdb
import itertools
import warnings
# import pickle
import os,shutil
from time import time

# from  pprint   import pprint
from  types    import SimpleNamespace
from  math     import floor,ceil
from  pathlib  import Path

# import tifffile
import numpy         as np
# import skimage.io    as io

# from scipy.ndimage        import zoom, label
# from scipy.ndimage.morphology import binary_dilation
# from skimage.segmentation import find_boundaries
# from scipy.ndimage        import convolve
from skimage.feature      import peak_local_max
from skimage.measure      import regionprops

from segtools.numpy_utils import collapse2, normalize3, plotgrid

# from segtools import color
from segtools import torch_models
from segtools.point_matcher import match_points_single, match_unambiguous_nearestNeib
from segtools.ns2dir import load, save, flatten

# import predict
import collections

from denoise_utils import structN2V_masker


def setup_dirs(savedir):
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)
  (savedir/'m').mkdir(exist_ok=True)
  shutil.copy("/projects/project-broaddus/devseg_2/src/denoiser.py",savedir)

def eg_img_meta():
  img_meta = SimpleNamespace()
  img_meta.voxel_size = np.array([0.09,0.09])
  img_meta.time_step  = 1 ## 1.5 for second dataset?
  return img_meta

def config_example():
  config = SimpleNamespace()
  ## prediction / evaluation / training
  config.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=torch_models.nn.Sequential)
  
  ## general img2img model ltraining
  config.sampler      = flat_sampler
  config.masker       = structN2V_masker
  config.batch_shape  = np.array([1,1,512,512])
  config.batch_space  = np.array([512,512])

  ## accumulate gradients, print loss, save x,y,yt,model, save vali_pred, total time in Fwd/Bwd passes
  config.times = [10,100,500,4000,10**5]
  config.lr = 1e-5

  return config

# def get_net(config):
#   net = config.getnet().cuda()
#   try:
#     net.load_state_dict(torch.load(config.best_model))
#   except:
#     print("no best_model, randomizing model weights...")
#     torch_models.init_weights(net)
#   return net

def normalize_td_vd(ta,td,vd):
  """
  this way we
  """
  mu  = np.mean(td.input)
  sig = np.std(td.input)
  ta.mu  = mu
  ta.sig = sig  
  td.input  = (td.input-mu)/sig
  td.target = (td.target-mu)/sig
  vd.input  = (vd.input-mu)/sig
  vd.target = (vd.target-mu)/sig

def normalize_raw(raw,ta):
  """
  this way we
  """
  try:
    mu =ta.mu
    sig=ta.sig
  except:
    mu =np.mean(raw)
    sig=np.std(raw)

  return (raw-mu)/sig, (mu,sig)

def train_init(config):
  config.savedir = Path(config.savedir).resolve()
  setup_dirs(config.savedir)

  ## training params we don't want to control directly, and artifacts that change over training time.
  ta = SimpleNamespace(i=1,losses=[],lr=config.lr,save_count=0,vali_scores=[],timings=[],heights=[])
  # ta.__dict__.update(**{**defaults,**ta.__dict__})

  ## load train and vali data
  td,vd = config.load_train_and_vali_data(config)
  normalize_td_vd(ta,td,vd)

  ## model
  m = SimpleNamespace()
  m.net = config.getnet().cuda()
  m.opt = torch.optim.Adam(m.net.parameters(), lr = config.lr)

  T  = SimpleNamespace(m=m,vd=vd,td=td,ta=ta,config=config)

  return T

def train(T):
  """
  requires m.net, td.input, td.target
  provides ta.losses and ta.i
  """

  m  = T.m
  vd = T.vd
  td = T.td
  ta = T.ta
  config = T.config
  
  for ta.i in range(ta.i,config.times[-1]):
    x,yt,w  = config.sampler(T,td)
    x  = torch.from_numpy(x).float().cuda()
    yt = torch.from_numpy(yt).float().cuda()
    w  = torch.from_numpy(w).float().cuda()

    y    = m.net(x)
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    loss.backward()

    # continue

    if ta.i%config.times[0]==0:
      m.opt.step()
      m.opt.zero_grad()
      ta.losses.append((loss/w.mean()).detach().cpu())
      ta.heights.append(y.max().detach().cpu())

    if ta.i%config.times[1]==0:
      ta.timings.append(time())
      dt = 0 if len(ta.timings)==1 else ta.timings[-1]-ta.timings[-2]
      l  = np.mean(ta.losses[-100:])
      ymax,ystd = float(y.max()), float(y.std())
      ytmax,ytstd = float(yt.max()), float(yt.std())
      print(f"i={ta.i:04d}, shape={x.shape}, loss={l:4f}, dt={dt:4f}, y={ymax:4f},{ystd:4f} yt={ytmax:4f},{ytstd:4f}", flush=True)

    if ta.i%config.times[2]==0:
      with warnings.catch_warnings():
        n = ta.i//config.times[2]
        save(ta , config.savedir/"ta/")
        save(x[0,0].detach().cpu().numpy().astype(np.float16)  , config.savedir/f"epoch/x/a{n:03d}.npy")
        save(y[0,0].detach().cpu().numpy().astype(np.float16)  , config.savedir/f"epoch/y/a{n:03d}.npy")
        save(yt[0,0].detach().cpu().numpy().astype(np.float16) , config.savedir/f"epoch/yt/a{n:03d}.npy")
        torch.save(m.net.state_dict(), config.savedir / f'm/net{n:03d}.pt')

    if ta.i%config.times[3]==0:
      n = ta.i//config.times[3]
      # validate(vd,T)

## Data Loading, Sampling, Weights, Etc

def add_meta_to_td(td):
  # td.axes = "TCYX"
  # td.dims = {k:v for k,v in zip(td.axes, td.input.shape)}
  # td.in_samples  = td.input.shape[0]
  # td.in_chan     = td.input.shape[1]
  td.in_space    = np.array(td.input.shape[2:])
  td.ndim        = len(td.in_space)

def validate(vd, T):
  vs = []
  vali_imgs = []
  n = T.ta.i//T.config.times[3]
  with torch.no_grad():
    for i in range(vd.input.shape[0]):
      res = T.m.net(torch.from_numpy(vd.input[[i]]).float().cuda()).cpu().numpy()[0]
      if vd.input.shape[0] > 10: vali_imgs.append(res)
      else: save(res[0].astype(np.float16),T.config.savedir / f"ta/vali_pred/e{n:03d}_i{i}.tif")
  if vd.input.shape[0] > 10:
    vali_imgs = np.array(vali_imgs)
    save(vali_imgs.astype(np.float16),T.config.savedir / f"ta/vali_pred/e{n:03d}_all.tif")

def flat_sampler(T,td):
  "sample from everywhere independent of annotations"

  ## sample from anywhere
  def patch():
    _pt = np.floor(np.random.rand(td.ndim)*(td.in_space - T.config.batch_space)).astype(int)
    ss = [slice(_pt[i],_pt[i] + T.config.batch_space[i]) for i in range(td.ndim)]
    st = np.random.randint(td.input.shape[0])
    ss = ([st],slice(None),*ss)
    x  = td.input[ss]
    yt = td.target[ss]
    x,yt,w = T.config.masker(x,yt,T)
    # ipdb.set_trace()
    return x,yt,w

  x,yt,w = zip(*[patch() for _ in range(T.config.batch_shape[0])])
  x  = np.concatenate(x,0)
  yt = np.concatenate(yt,0)
  w  = np.concatenate(w,0)

  return x,yt,w

def index_only_sampler(T,td):
  "sample from everywhere independent of annotations"

  st = np.random.randint(0,td.input.shape[0],T.config.batch_shape[0])
  x  = td.input[st]
  yt = td.target[st]
  x,yt,w = T.config.masker(x,yt,T)

  return x,yt,w

def predict_raw(net,img,dims,ta=None,**kwargs3d):

  assert dims in ["NCYX","NBCYX","CYX","ZYX","CZYX","NCZYX","NZYX",]

  # if ta:
  img,(mu,sig) = normalize_raw(img,ta)

  # ipdb.set_trace()
  with torch.no_grad():
    if dims=="NCYX":
      def f(i): return net(torch.from_numpy(img[[i]]).cuda().float()).cpu().numpy()
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="NBCYX":
      def f(i): return net(torch.from_numpy(img[i]).cuda().float()).cpu().numpy()
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="CYX":
      res = net(torch.from_numpy(img[None]).cuda().float()).cpu().numpy()[0]
    if dims=="ZYX":
      ## assume 1 channel. remove after prediction.
      res = torch_models.apply_net_tiled_3d(net,img[None],**kwargs3d)[0]
    if dims=="CZYX":
      res = torch_models.apply_net_tiled_3d(net,img)
    if dims=="NCZYX":
      def f(i): return torch_models.apply_net_tiled_3d(net,img[i],**kwargs3d)[0]
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="NZYX":
      def f(i): return torch_models.apply_net_tiled_3d(net,img[i,None],**kwargs3d)[0]
      res = np.array([f(i) for i in range(img.shape[0])])

  img = img*sig + mu ## un-normalize

  return res



notes = """

Tue Feb 18 10:10:11 2020

ground truth points are used in three places:
1. training data generation. here we require points at full res, although we may downsample the target after creation.
2. content_sampler. Here we need points with the same scale as the training data, which may have been downsampled.
3. validation. again here we need points with same scale as training data.

What's the right way to handle these needs. Should we load points from traj. Pass them from disbi. Or build them from man_tracks?
When should we do the rescaling?
- points and target should be downscaled at same place at same time. (either in training data generation, or in disbi method.)
- we shouldn't have to _hack_ downscaled training into our workflow. it should be explicitly supported. This means we need a separate attribute for downscaled data dir.
- how much of the workflow should take place in the downscaled space?
  1. net training. net prediction. validation evaluation. should all be in downscaled space.
  2. after peak_local_max we should immediately upscale points back to full res AFTER prediction.
  3. making movies of the result should be done on full res data?

Sun Mar 15 19:02:08 2020

What are the advantages / disadvantages of turning this module into a StructN2V object?
Most of the inner functions wouldn't need the trainer,ta,arguments...
i.e. if you have unique _global_ names you don't need to pass them as params.
And this would give all our functions access to these global names.

# Tue Apr 14 15:35:21 2020

I'm not getting good results with every_30_min_timelapse.tiff.
But that is the clearest image we have.

# Sat May 23 17:35:10 2020

How do we want to handle normalization?
1. It should be done by the method (not externally by user), during training and prediction.
2. You should be able to normalize the predictions in the same way as the training data (using exact same mean and var).
But let's try without that first, then see if it makes a difference.

# 2020-09-20

Not sure the U-net worked as was prev written. Unet2/Unet3 work with 2D or 3D images, and the 2/3 describes the number of downsamplings.
So we have to make sure the standard patch size (512,512) goes with a (2,2) pool and (5,5) kern.



"""








