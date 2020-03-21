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
from segtools.math_utils import conv_at_pts4
# from segtools import color
from segtools import torch_models
from segtools.point_matcher import match_points_single, match_unambiguous_nearestNeib
from segtools.ns2dir import load, save, flatten

import predict
import collections



def setup_dirs(savedir):
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)
  (savedir/'m').mkdir(exist_ok=True)
  shutil.copy("/projects/project-broaddus/devseg_2/src/train_StructN2V.py",savedir)

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
  config = T.c

  defaults = dict(i=0,losses=[],lr=2e-4,i_final=config.bp_per_epoch*22,save_count=0,vali_scores=[],timings=[],heights=[])
  ta.__dict__.update(**{**defaults,**ta.__dict__})

  try: m.opt
  except: m.opt  = torch.optim.Adam(m.net.parameters(), lr = ta.lr)
  
  for ta.i in range(ta.i,ta.i_final):
    ta.timings.append(time())

    x,yt,w = config.sampler(T)
    x  = torch.from_numpy(x).float().cuda()
    yt = torch.from_numpy(yt).float().cuda()
    w  = torch.from_numpy(w).float().cuda()

    ## put patches through the net, then backprop
    y    = m.net(x)
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    loss.backward()

    if ta.i%10==0:
      m.opt.step()
      m.opt.zero_grad()

    ## monitoring training and validation

    ta.losses.append(float(loss/w.mean()))
    ta.heights.append(float(y.max()))

    if ta.i%100==0:
      print(f"i={ta.i:04d}, loss={np.mean(ta.losses[-100:])}", flush=True)
      # print("x", float(x.mean()),float(x.std()))
      print("y  {:5f} {:5f}".format(float(y.max()),float(y.std())))
      print("yt {:5f} {:5f}".format(float(yt.max()),float(yt.std())))
      # print("w", float(w.mean()),float(w.std()))
      with warnings.catch_warnings():
        save(ta , config.savedir/"ta/")
        save(x[0,0].detach().cpu().numpy().max(0)  , config.savedir/f"epoch/x/a{ta.i//100:03d}.npy")
        save(y[0,0].detach().cpu().numpy().max(0)  , config.savedir/f"epoch/y/a{ta.i//100:03d}.npy")
        save(yt[0,0].detach().cpu().numpy().max(0) , config.savedir/f"epoch/yt/a{ta.i//100:03d}.npy")

    if ta.i%config.bp_per_epoch==0:
      ta.save_count += 1
      with torch.no_grad():
        validate(vd,T)
        torch.save(m.net.state_dict(), config.savedir / f'm/net{ta.save_count:02d}.pt')

## Data Loading, Sampling, Weights, Etc

def validate(vd, T):
  # m,d,ta = T.m
  vs = []
  with torch.no_grad():
    for i in range(vd.input.shape[0]):
      res = predict.apply_net_tiled_3d(T.m.net,vd.input[i])
      # pts = peak_local_max(res[0],threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
      # score3  = match_unambiguous_nearestNeib(d.gt[i],pts,dub=3,scale=ta.rescale_for_matching)
      # score10 = match_unambiguous_nearestNeib(d.gt[i],pts,dub=10,scale=ta.rescale_for_matching)
      # s3  = [score3.n_matched,  score3.n_proposed,  score3.n_gt]
      # s10 = [score10.n_matched, score10.n_proposed, score10.n_gt]
      # print(ta.i,i,s3,s10)
      # vs.append([s3,s10])
      # save(res[0],config.savedir / f"ta/pred/e{e}_i{i}.tif")
      # save(res[0,ta.patch_space[0]//2].astype(np.float16),config.savedir / f"ta/ms_z/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0).astype(np.float16),T.c.savedir / f"ta/mx_z/e{T.ta.save_count:02d}_i{i}.tif")
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
  _pt = np.floor(np.random.rand(3)*(T.ta.in_space-T.ta.patch_space)).astype(int)
  sz,sy,sx = [slice(_pt[i],_pt[i] + T.ta.patch_space[i]) for i in range(3)]
  st = np.random.randint(T.ta.dims['T'])

  x  = T.td.input[[st],:,sz,sy,sx]
  yt = T.td.target[[st],:,sz,sy,sx]
  # w  = np.ones(x.shape)
  # w  = weights(yt,T.ta,trainer)
  x,yt,w = apply_mask(x,yt)

  return x,yt,w

def apply_mask(x,yt,):
  kern = np.zeros((1,17,1)) ## must be odd
  kern[0,:,0] = 1
  kern[0,8,0] = 2
  # kern[9] = 1
  # kern[9,1,1] = 2
  # patch_space = x.shape
  ma = mask_from_footprint(x.shape[2:],kern,)
  # print(ma.shape)
  ma = ma[None,None] ## samples , channels
  yt = x.copy()
  x[ma>0] = np.random.rand(*x.shape)[ma>0]
  w = (ma==2).astype(np.float)
  return x,yt,w

def mask_from_footprint(sh,kern,frac=0.01):
  # takes a shape sh. returns random mask with that shape
  pts = (np.random.rand(int(np.prod(sh)*frac),3)*sh).astype(int)
  target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
  target = target.astype(np.uint8)
  return target

def mask_2(patch_size,frac):
  "build random mask for small number of central pixels"
  n = int(np.prod(patch_size) * frac)
  kern = np.zeros((19,3,3)) ## must be odd
  kern[:,1,1] = 1
  kern[9] = 1
  kern[9,1,1] = 1
  mask = np.random.rand(*patch_size)<frac
  indices = np.indices(patch_size)[:,mask]
  deltas  = np.indices(kern.shape)[:,kern==1]
  newmask = np.zeros(patch_size)
  for dx in deltas.T:
    inds = (indices+dx[:,None]).T.clip(min=[0,0,0],max=np.array(patch_size)-1).T
    newmask[tuple(inds)] = 1

def sparse_3set_mask(d):
  "build random mask for small number of central pixels"
  n = int(np.prod(d.patch_size) * d.frac)
  z_inds = np.random.randint(0,d.patch_size[0],n)
  y_inds = np.random.randint(0,d.patch_size[1],n)
  x_inds = np.random.randint(0,d.patch_size[2],n)
  ma = np.zeros(d.patch_size,dtype=np.int)
  
  for i in d.xmask:
    m = x_inds+i == (x_inds+i).clip(0,d.patch_size[2]-1)
    ma[z_inds[m], y_inds[m],x_inds[m]+i] = 1

  for i in d.ymask:
    m = y_inds+i == (y_inds+i).clip(0,d.patch_size[1]-1)
    ma[z_inds[m], y_inds[m]+i,x_inds[m]] = 1

  for i in d.zmask:
    m = z_inds+i == (z_inds+i).clip(0,d.patch_size[0]-1)
    ma[z_inds[m]+i, y_inds[m],x_inds[m]] = 1

  ma = ma.astype(np.uint8)
  ma[z_inds,y_inds,x_inds] = 2
  return ma





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





"""









