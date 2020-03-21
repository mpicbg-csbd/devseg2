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

# from  pprint   import pprint
from  types    import SimpleNamespace
from  math     import floor,ceil
from  pathlib  import Path

import tifffile
import numpy         as np
import skimage.io    as io
# import matplotlib.pyplot as plt
# plt.switch_backend("agg")

from scipy.ndimage        import zoom, label
# from scipy.ndimage.morphology import binary_dilation
from skimage.feature      import peak_local_max
from skimage.segmentation import find_boundaries
from skimage.measure      import regionprops
from skimage.morphology   import binary_dilation
from scipy.ndimage import convolve

from segtools.numpy_utils import collapse2, normalize3, plotgrid
from segtools import color
from segtools.defaults.ipython import moviesave
from segtools.math_utils import conv_at_pts,conv_at_pts2

from ns2dir import load, save

import torch_models
import predict

from time import time


notes = """
## Usage

import denoise as p
m,d,td,ta = p.init()
p.train(m,td,ta)

## Names

m  :: models (update in place)
d  :: validation data
td :: training data
ta :: training artifacts (update in place)

if you want to stop training just use Ctrl-C, it will up at the same iteration where you left off because of state in ta and m.

"""

def init():
  savedir = Path("/projects/project-broaddus/devseg_2/e05_flydenoise/test/")
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)

  ta = SimpleNamespace()
  ta.savedir = savedir

  ta.train_times = [0,5,10,15,45]
  ta.vali_times = [0,10,30,40,50]

  td = SimpleNamespace()
  td.input = np.array([load(f"/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01/t{i:03d}") for i in ta.trains])
  td.intut = td.input / 1300
  td.input = td.input[None]

  vd = SimpleNamespace()
  vd.raw = vd.raw[ta.valis]

  m = SimpleNamespace() ## model
  m.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  (savedir/'m').mkdir(exist_ok=True)

  return m,vd,td,ta

def train(m,d,td,ta):
  """
  requires m.net, td.input, td.target
  provides ta.losses and ta.i
  """

  n_chan, n_samples = td.input.shape[:2]
  shape_zyx = np.array(td.input.shape)[2:]

  # ta.dmask = SimpleNamespace(patch_size=(10,128,128),xmask=[0],ymask=[0],zmask=range(-10,10),frac=0.01)
  defaults = dict(i=0,i_final=10**5,lr=1e-6,losses=[],save_count=0,timing=[])
  ta.__dict__.update(**{**defaults,**ta.__dict__})
  opt = torch.optim.Adam(m.net.parameters(),lr=ta.lr)

  for ta.i in range(ta.i,ta.i_final):
    ta.timing.append(time())

    _pt = np.floor(np.random.rand(3)*(shape_zyx-(10,128,128))).astype(int)
    sz,sy,sx = slice(_pt[0],_pt[0]+10), slice(_pt[1],_pt[1]+128), slice(_pt[2],_pt[2]+128)
    sc = np.random.randint(n_samples)
    # sc = 2 ## only train on middle sample (timepoint)

    ## apply masking, feed to net, and accumulate gradients
    xorig = td.input[0,sc,sz,sy,sx]
    xmasked = xorig.clone()
    # ma = torch.from_numpy(sparse_3set_mask(ta.dmask))
    ma = torch.from_numpy(mask_from_footprint((10,128,128)))
    xmasked[ma>0] = torch.rand(xmasked.shape).cuda()[ma>0]
    y  = m.net(xmasked[None,None])[0,0]
    loss = torch.abs((y-xorig)[ma==2]**1).mean() # + weight*torch.abs((y-global_avg)**1).mean()
    loss.backward()

    if ta.i%1==0:
      opt.step()
      opt.zero_grad()

    ## extra stuff for monitoring training

    ta.losses.append(float(loss))

    if ta.i%100==0:
      print(f"i={ta.i:04d}, loss={np.mean(ta.losses[-100:])}")
      with warnings.catch_warnings():
        save(ta,ta.savedir/"ta/")
        # ipdb.set_trace()
        # np.save(ta.savedir / f"img{ta.i//20:03d}.npy",y[32].detach().cpu().numpy())

    if ta.i%1000==0:
      ta.save_count += 1
      with torch.no_grad():
        torch.save(m.net.state_dict(), ta.savedir / f'm/net{ta.save_count:02d}.pt')
        validate(m,d,td,ta)
        # res = predict.apply_net_tiled_3d(m.net,td.input[:,2].detach().cpu().numpy())
        # tifffile.imsave(ta.savedir / f"ta/res{k:02d}.tif",res,compress=0)

def validate(m,d,td,ta):
  with torch.no_grad():
    for i in range(d.raw.shape[0]):
      print(f"predict on {i}")
      res = predict.apply_net_tiled_3d(m.net,d.raw[None,i])
      save(res[0,5],ta.savedir / f"ta/ms/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0),ta.savedir / f"ta/mx/e{ta.save_count:02d}_i{i}.tif")

# def predict_on_divisiontimes():
#   raw = load("/projects/project-broaddus/devseg_2/raw/every_30_min_timelapse.tiff")
#   raw = raw[[31, 21, 17, 9,]]
#   for i,img in enumerate(raw):

## helper functions

# def sparse_conv(data,kern):
#   assert data.dtype is np.dtype('bool')
#   for p in np.indices(data.shape)[:,data].T:
#     pass
    
def mask_from_footprint(sh):
  # takes a shape sh. returns random mask with that shape
  kern = np.zeros((19,3,3)) ## must be odd
  kern[:,1,1] = 1
  kern[9] = 1
  kern[9,1,1] = 2
  pts = (np.random.rand(int(np.prod(sh)*0.01),3)*sh).astype(int)
  target = conv_at_pts2(pts,kern,sh,lambda a,b:np.maximum(a,b))
  target = target.astype(np.uint8)
  return target

  # target[tuple(pts.T)]=2
  # target2 = np.zeros(target.shape)
  # target2[tuple(pts.T)] = 1

  # d = SimpleNamespace()
  # d.target = target
  # d.target2 = target2
  # d.pts = pts
  # return d

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
