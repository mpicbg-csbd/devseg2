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

from ns2dir import load, save

import torch_models
import predict

notes = """
## Usage

import denoise as p
m,d,td,ta = p.init()
p.train(m,td,ta)

## Names

m  :: models (update in place)
d  :: raw data
td :: training data
ta :: training artifacts (update in place)

if you want to stop training just use Ctrl-C, it will up at the same iteration where you left off because of state in ta and m.

"""

def init():
  savedir = Path("/projects/project-broaddus/devseg_2/e01/test2/")
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)

  ta = SimpleNamespace()
  ta.savedir = savedir

  d = SimpleNamespace()
  # d.raw = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t{n:03d}.tif") for n in [0,10,100,189]])
  # d.raw = normalize3(d.raw,2,99.6)
  # d.raw = load("/projects/project-broaddus/devseg_2/raw/every_30_min_timelapse.tiff")

  fs = [
    "detection_t_00_x_730_y_512_z_30.tiff",
    "detection_t_14_x_457_y_821_z_24.tiff",
    "detection_t_21_x_453_y_878_z_35.tiff",
    "detection_t_27_x_401_y_899_z_39.tiff",
    "detection_t_30_x_355_y_783_z_7.tiff",
    "detection_t_24_x_562_y_275_z_12.tiff",
    "detection_t_38_x_700_y_573_z_17.tiff",
    ]

  ## detection,time,z,y,x
  d.raw = np.array([load(f"/projects/project-broaddus/devseg_2/raw/det/{f}") for f in fs])
  d.raw = d.raw.reshape((-1, 10, 1024, 1024)) ## combine detection and time
  ta.valisamples = [0,15,30,45,62,]

  # return d
  # d.raw = d.raw[[0,10,20,30,39]]
  # d.raw = d.raw[[20]]
  # d.raw = normalize3(d.raw,2,99.6) ## Already normalized!

  td = SimpleNamespace()  ## training/target data
  td.input  = torch.from_numpy(d.raw[None]).float()
  td.input  = td.input.cuda()

  m = SimpleNamespace() ## model
  m.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  (savedir/'m').mkdir(exist_ok=True)

  return m,d,td,ta

def train(m,d,td,ta):
  """
  requires m.net, td.input, td.target
  provides ta.losses and ta.i
  """

  n_chan, n_samples = td.input.shape[:2]
  shape_zyx = np.array(td.input.shape)[2:]
  dmask = SimpleNamespace(patch_size=(10,128,128),xmask=[0],ymask=[0],frac=0.01)

  defaults = dict(i=0,lr=3e-5,losses=[],save_count=0)
  ta.__dict__.update(**{**defaults,**ta.__dict__})
  opt = torch.optim.Adam(m.net.parameters(),lr=ta.lr)

  for ta.i in range(ta.i,10**5):
    _pt = np.floor(np.random.rand(3)*(shape_zyx-(10,128,128))).astype(int)
    sz,sy,sx = slice(_pt[0],_pt[0]+10), slice(_pt[1],_pt[1]+128), slice(_pt[2],_pt[2]+128)
    sc = np.random.randint(n_samples)
    # sc = 2 ## only train on middle sample (timepoint)

    ## apply masking, feed to net, and accumulate gradients
    xorig = td.input[0,sc,sz,sy,sx]
    xmasked = xorig.clone()
    ma = torch.from_numpy(sparse_3set_mask(dmask))
    xmasked[ma>0] = torch.rand(xmasked.shape).cuda()[ma>0]
    y  = m.net(xmasked[None,None])[0,0]
    loss = torch.abs((y-xorig)[ma==2]**1).mean() # + weight*torch.abs((y-global_avg)**1).mean()
    # print(float(loss),flush=True)
    loss.backward()

    if ta.i%10==0:
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

    if ta.i%2000==0:
      ta.save_count += 1
      with torch.no_grad():
        torch.save(m.net.state_dict(), ta.savedir / f'm/net{ta.save_count:02d}.pt')
        predict4(m,d,td,ta)
        # res = predict.apply_net_tiled_3d(m.net,td.input[:,2].detach().cpu().numpy())
        # tifffile.imsave(ta.savedir / f"ta/res{k:02d}.tif",res,compress=0)


def predict4(m,d,td,ta):
  with torch.no_grad():
    for i in range(d.raw[ta.valisamples].shape[0]):
      print(f"predict on {i}")
      res = predict.apply_net_tiled_3d(m.net,d.raw[None,i])
      save(res[0,5],ta.savedir / f"ta/ms/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0),ta.savedir / f"ta/mx/e{ta.save_count:02d}_i{i}.tif")

# def predict_on_divisiontimes():
#   raw = load("/projects/project-broaddus/devseg_2/raw/every_30_min_timelapse.tiff")
#   raw = raw[[31, 21, 17, 9,]]
#   for i,img in enumerate(raw):


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

  ma = ma.astype(np.uint8)
  ma[z_inds,y_inds,x_inds] = 2
  return ma