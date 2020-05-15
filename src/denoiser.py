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

from denoise_utils import nearest_neib_sampler, footprint_sampler


def setup_dirs(savedir):
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)
  (savedir/'m').mkdir(exist_ok=True)
  shutil.copy("/projects/project-broaddus/devseg_2/src/denoiser.py",savedir)

def eg_img_meta():
  img_meta = SimpleNamespace()
  img_meta.voxel_size = np.array([1.0,0.09,0.09])
  img_meta.time_step  = 1 ## 1.5 for second dataset?
  return img_meta

def config(img_meta):
  config = SimpleNamespace()
  # config.savedir = Path("detector_test/")

  print(img_meta)

  ## prediction / evaluation / training
  config.f_net_args    = ((16,[[1],[1]]), dict(finallayer=torch_models.nn.Sequential)) ## *args and **kwargs
  # config.norm          = lambda img: normalize3(img,2,99.4,clip=True)
  config.norm          = lambda img: img ## no norm for alex's images
  ## TODO: add image config.unnorm : (img,norm_state) -> img # to be applied after prediction (only really important for N2V/restoration tasks)
  ## Also, config.norm should return and image and a norm_state we can use later (But this data will get lost! train/eval/pred are separate tasks! we can save to disk in training_artifacts...)
  config.pt_norm       = lambda pts: pts
  config.pt_unnorm     = lambda pts: pts

  ## evaluation / training
  config.rescale_for_matching = (2,1,1)
  config.dub=10

  ## training [detector only]
  # config.sigmas       = np.array([1,7,7])
  # config.kernel_shape = np.array([43,43,43]) ## 7sigma in each direction
  ## general img2img model ltraining
  config.sampler      = flat_sampler
  config.masker       = nearest_neib_sampler
  config.patch_space  = np.array([16,128,128])
  config.patch_full   = np.array([1,1,16,128,128])
  config.i_final      = 10**5
  config.bp_per_epoch = 4*10**3
  config.lr = 1e-5
  ## fg/bg weights stuff for fluorescence images & class-imbalanced data
  # config.fg_bg_thresh = np.exp(-16/2)
  # config.bg_weight_multiplier = 0.0
  # config.weight_decay = True

  return config

def _load_net(config):
  args,kwargs = config.f_net_args
  net = torch_models.Unet3(*args,**kwargs).cuda()
  try:
    net.load_state_dict(torch.load(config.best_model))
  except:
    print("no best_model, randomizing model weights...")
    torch_models.init_weights(net)
  return net

def train_init(config):
  config.savedir = Path(config.savedir).resolve()
  setup_dirs(config.savedir)

  ## load train and vali data
  vd,td = config.load_train_and_vali_data(config)

  ## model
  m = SimpleNamespace()
  m.net = _load_net(config)

  ## training params we don't want to control directly, and artifacts that change over training time.
  ta = SimpleNamespace(i=0,losses=[],lr=config.lr,i_final=config.i_final,save_count=0,vali_scores=[],timings=[],heights=[])
  # defaults = dict(i=0,losses=[],lr=2e-4,i_final=config.bp_per_epoch*22,save_count=0,vali_scores=[],timings=[],heights=[])
  # ta.__dict__.update(**{**defaults,**ta.__dict__})

  T  = SimpleNamespace(m=m,vd=vd,td=td,ta=ta,config=config)

  return T



@DeprecationWarning
def init(config):
  savedir = Path(config.train_dir).resolve()
  setup_dirs(savedir)
  config.savedir = savedir

  ## training artifacts
  ta = SimpleNamespace()
  ta.savedir = savedir
  ta.use_denoised = False

  ## Transfer from trainer
  # ta.i_final = trainer.i_final
  # ta.rescale_for_matching = trainer.rescale_for_matching
  # ta.bp_per_epoch = trainer.bp_per_epoch

  # ## test data
  # ta.valitimes  = trainer.valitimes
  # ta.traintimes = trainer.traintimes

  print("Training Times : ", config.traintimes)
  print("Vali Times : ", config.valitimes)

  vd = config.load_data(config.valitimes, config)
  td = config.load_data(config.traintimes, config)
  # save(td.target.max(1).astype(np.float16),ta.savedir/'ta/mx_vali/target.tif')

  ## model
  m = SimpleNamespace()
  args,kwargs = config.f_net_args
  m.net = torch_models.Unet3(*args,**kwargs).cuda()
  torch_models.init_weights(m.net)

  ## describe input shape
  ta.axes = "TCZYX"
  ta.dims = {k:v for k,v in zip(ta.axes, td.input.shape)}
  ta.in_samples  = td.input.shape[0]
  ta.in_chan     = td.input.shape[1]
  ta.in_space    = np.array(td.input.shape[2:])
  ta.patch_space = config.patch_space
  ta.patch_full  = config.patch_full

  T = SimpleNamespace(m=m,vd=vd,td=td,ta=ta,c=config)

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

  # defaults = dict(i=0,losses=[],lr=1e-4,i_final=config.bp_per_epoch*22,save_count=0,vali_scores=[],timings=[],heights=[])
  # ta.__dict__.update(**{**defaults,**ta.__dict__})

  try: m.opt
  except: m.opt  = torch.optim.Adam(m.net.parameters(), lr = ta.lr)
  
  # _x,_yt,_w  = config.sampler(T)

  # x  = torch.zeros(_x.shape).float().cuda()
  # yt = torch.zeros(_yt.shape).float().cuda()
  # w  = torch.zeros(_w.shape).float().cuda()

  for ta.i in range(ta.i,ta.i_final):
    ta.timings.append(time()) ## 1

    # _x  = _x + np.random.rand(*_x.shape)*0.05
    # _yt = _yt + np.random.rand(*_yt.shape)*0.05
    # _w  = np.ones(_w.shape).float().cuda()
    x,yt,w  = config.sampler(T)

    # ipdb.set_trace()

    ta.timings.append(time()) ## 2

    x  = torch.from_numpy(x).float().cuda()
    yt = torch.from_numpy(yt).float().cuda()
    w  = torch.from_numpy(w).float().cuda()
    # x[...]  = torch.from_numpy(_x)[...] # torch.from_numpy(x).float().cuda()
    # yt[...] = torch.from_numpy(_yt)[...] # torch.from_numpy(yt).float().cuda()
    # w[...]  = torch.from_numpy(_w)[...] # torch.from_numpy(w).float().cuda()

    ta.timings.append(time()) ## 3

    ## put patches through the net, then backprop
    y    = m.net(x)
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    loss.backward()

    ta.timings.append(time()) ## 4

    if ta.i%10==0:
      m.opt.step()
      m.opt.zero_grad()

    if ta.i%50==0:
      ## monitoring training and validation ## takes 0.2 sec!!! so expensive! (but not when inside this mod? i don't understand...)
      ta.losses.append((loss/w.mean()).detach().cpu())
      ta.heights.append(y.max().detach().cpu())

    if ta.i%100==0:
      dt = ta.timings[-1]-ta.timings[-400] if ta.i>0 else 0
      l  = np.mean(ta.losses[-100:])
      ymax,ystd = float(y.max()), float(y.std())
      ytmax,ytstd = float(yt.max()), float(yt.std())
      print(f"i={ta.i:04d}, loss={l:7f}, dt={dt:7f}, y={ymax:4f},{ystd:4f} yt={ytmax:4f},{ytstd:4f}", flush=True)

    if ta.i%500==0:
      with warnings.catch_warnings():
        save(ta , config.savedir/"ta/")
        save(x[0,0].detach().cpu().numpy().max(0)  , config.savedir/f"epoch/x/a{ta.i//500:03d}.npy")
        save(y[0,0].detach().cpu().numpy().max(0)  , config.savedir/f"epoch/y/a{ta.i//500:03d}.npy")
        save(yt[0,0].detach().cpu().numpy().max(0) , config.savedir/f"epoch/yt/a{ta.i//500:03d}.npy")

    if ta.i%config.bp_per_epoch==config.bp_per_epoch-1:
      ta.save_count += 1
      validate(vd,T)
      torch.save(m.net.state_dict(), config.savedir / f'm/net{ta.save_count:02d}.pt')

## Data Loading, Sampling, Weights, Etc

def add_meta_to_td(td):
  td.axes = "TCZYX"
  td.dims = {k:v for k,v in zip(td.axes, td.input.shape)}
  td.in_samples  = td.input.shape[0]
  td.in_chan     = td.input.shape[1]
  # td.in_space    = td.input.shape[2:]
  td.in_space    = np.array(td.input.shape[2:])


def validate(vd, T):
  # m,d,ta = T.m
  vs = []
  with torch.no_grad():
    for i in range(vd.input.shape[0]):
      # res = torch_models.apply_net_tiled_3d(m.net,vd.input[i])
      res = torch_models.apply_net_tiled_3d(T.m.net,vd.input[i])
      # pts = peak_local_max(res[0],threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
      # score3  = match_unambiguous_nearestNeib(d.gt[i],pts,dub=3,scale=ta.rescale_for_matching)
      # score10 = match_unambiguous_nearestNeib(d.gt[i],pts,dub=10,scale=ta.rescale_for_matching)
      # s3  = [score3.n_matched,  score3.n_proposed,  score3.n_gt]
      # s10 = [score10.n_matched, score10.n_proposed, score10.n_gt]
      # print(ta.i,i,s3,s10)
      # vs.append([s3,s10])
      # save(res[0],config.savedir / f"ta/pred/e{e}_i{i}.tif")
      # save(res[0,ta.patch_space[0]//2].astype(np.float16),config.savedir / f"ta/ms_z/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0).astype(np.float16),T.config.savedir / f"ta/mx_z/e{T.ta.save_count:02d}_i{i}.tif")
      # save(res[0].max(1).astype(np.float16),ta.savedir / f"ta/mx_y/e{ta.save_count:02d}_i{i}.tif")
      # save(res[0].max(2).astype(np.float16),ta.savedir / f"ta/mx_x/e{ta.save_count:02d}_i{i}.tif")
      # save(res[0].astype(np.float16),ta.savedir / f"ta/vali_full/e{ta.save_count:02d}_i{i}.tif")
  # ta.vali_scores.append(vs)

def load_isbi_training_data(times,config):
  d = SimpleNamespace()
  d.input  = np.array([load(config.input_dir / f"t{n:03d}.tif") for n in times])
  d.input  = config.norm(d.input)
  d.target = d.input
  d.target = d.target[:,None]
  d.input  = d.input[:,None]

  return d

def flat_sampler(T):
  "sample from everywhere independent of annotations"

  ## sample from anywhere
  _pt = np.floor(np.random.rand(3)*(T.td.in_space-T.config.patch_space)).astype(int)
  sz,sy,sx = [slice(_pt[i],_pt[i] + T.config.patch_space[i]) for i in range(3)]
  st = np.random.randint(T.td.dims['T'])

  x  = T.td.input[[st],:,sz,sy,sx]
  yt = T.td.target[[st],:,sz,sy,sx]
  # w  = np.ones(x.shape)
  # w  = weights(yt,T.ta,trainer)
  x,yt,w = T.config.masker(x,yt,T)

  return x,yt,w

def predict_raw(T,img):
  img = T.config.norm(img)
  res = torch_models.apply_net_tiled_3d(T.m.net,img[None])[0]
  return res

## mask builder






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



"""









