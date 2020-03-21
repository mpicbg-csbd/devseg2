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
  savedir = Path("/projects/project-broaddus/devseg_2/e01/test4/")
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)

  ta = SimpleNamespace()
  ta.savedir = savedir

  # d.raw = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t{n:03d}.tif") for n in [0,10,100,189]])
  # d.raw = normalize3(d.raw,2,99.6)
  # d.raw = load("/projects/project-broaddus/devseg_2/raw/every_30_min_timelapse.tiff")

  # ta.trains = [
  #   "detection_t_00_x_730_y_512_z_30.tiff",
  #   "detection_t_14_x_457_y_821_z_24.tiff",
  #   "detection_t_21_x_453_y_878_z_35.tiff",
  #   "detection_t_24_x_562_y_275_z_12.tiff",
  #   "detection_t_27_x_401_y_899_z_39.tiff",
  #   "detection_t_30_x_355_y_783_z_7.tiff",
  #   "detection_t_38_x_700_y_573_z_17.tiff",
  #   ]
  ta.trains = list(ddinv.keys())[:-5]
  # ta.valis = list(ddinv.keys())[-5:]
  ta.valis = [0,10,30,40,50]

  # ta.valis = [
  #   "detection_t_00_x_730_y_512_z_30.tiff",
  #   "detection_t_09_x_303_y_814_z_25.tiff",
  #   # "detection_t_15_x_885_y_526_z_21.tiff",
  #   # "detection_t_20_x_641_y_831_z_13.tiff",
  #   "detection_t_26_x_252_y_802_z_0.tiff",
  #   # "detection_t_28_x_452_y_401_z_31.tiff",
  #   "detection_t_30_x_353_y_932_z_2.tiff",
  #   # "detection_t_33_x_614_y_330_z_33.tiff",
  #   "detection_t_39_x_193_y_348_z_17.tiff",
  #   ]

  def load2(fi):
    res = load(fi)
    res = res[0] ## only the round ones at time zero
    res = res[:,512-100:512+100,512-100:512+100] ## focus on them
    return res

  # ## detection,time,z,y,x
  # d = SimpleNamespace()
  # d.raw = np.array([load2(f"/projects/project-broaddus/devseg_2/raw/det/{f}") for f in ta.valis])
  # # d.raw = d.raw[:,5] ## take 5th timepoint from each
  # # d.raw = d.raw.reshape((-1, 10, 1024, 1024)) ## combine detection and times

  ## detection,time,z,y,x
  # d2 = SimpleNamespace()
  # d2.raw = np.array([load2(f"/projects/project-broaddus/devseg_2/raw/det/{f}") for f in ta.trains])
  # d2.raw = d2.raw.reshape((-1, 10, 1024, 1024)) ## combine detection and time

  d2 = SimpleNamespace()
  d2.raw = np.array([load2(f"/projects/project-broaddus/devseg_2/raw/det/{f}") for f in ta.trains])

  d = SimpleNamespace()
  d.raw = d2.raw[ta.valis]

  td = SimpleNamespace()  ## training/target data
  td.input  = torch.from_numpy(d2.raw[None]).float()
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


ddinv = {
 'detection_t_00_x_730_y_512_z_30.tiff': 'fish_1',
 'detection_t_02_x_736_y_440_z_13.tiff': 'fish_3',
 'detection_t_09_x_303_y_814_z_25.tiff': 'fish_1',
 'detection_t_09_x_600_y_343_z_30.tiff': 'fish_0',
 'detection_t_10_x_745_y_668_z_9.tiff': 'fish_3',
 'detection_t_12_x_655_y_294_z_10.tiff': 'fish_3',
 'detection_t_13_x_503_y_793_z_8.tiff': 'fish_3',
 'detection_t_13_x_696_y_416_z_35.tiff': 'fish_4',
 'detection_t_14_x_457_y_821_z_24.tiff': 'fish_3',
 'detection_t_14_x_759_y_413_z_39.tiff': 'fish_4',
 'detection_t_15_x_688_y_313_z_29.tiff': 'fish_3',
 'detection_t_15_x_749_y_571_z_19.tiff': 'fish_1',
 'detection_t_15_x_885_y_526_z_21.tiff': 'fish_4',
 'detection_t_16_x_324_y_875_z_43.tiff': 'fish_1',
 'detection_t_16_x_533_y_857_z_35.tiff': 'fish_3',
 'detection_t_16_x_611_y_629_z_36.tiff': 'fish_4',
 'detection_t_17_x_105_y_492_z_39.tiff': 'fish_0',
 'detection_t_17_x_737_y_413_z_27.tiff': 'fish_3',
 'detection_t_18_x_499_y_296_z_0.tiff': 'fish_3',
 'detection_t_19_x_205_y_826_z_42.tiff': 'fish_1',
 'detection_t_20_x_622_y_435_z_29.tiff': 'fish_1',
 'detection_t_20_x_641_y_831_z_13.tiff': 'fish_3',
 'detection_t_21_x_453_y_878_z_35.tiff': 'fish_3',
 'detection_t_21_x_69_y_672_z_39.tiff': 'fish_0',
 'detection_t_22_x_416_y_328_z_0.tiff': 'fish_3',
 'detection_t_23_x_668_y_506_z_17.tiff': 'fish_1',
 'detection_t_23_x_803_y_515_z_30.tiff': 'fish_4',
 'detection_t_24_x_562_y_275_z_12.tiff': 'fish_3',
 'detection_t_25_x_295_y_400_z_2.tiff': 'fish_4',
 'detection_t_25_x_674_y_507_z_10.tiff': 'fish_1',
 'detection_t_26_x_246_y_900_z_24.tiff': 'fish_1',
 'detection_t_26_x_252_y_802_z_0.tiff': 'fish_2',
 'detection_t_26_x_271_y_409_z_1.tiff': 'fish_4',
 'detection_t_26_x_544_y_858_z_6.tiff': 'fish_3',
 'detection_t_27_x_272_y_796_z_0.tiff': 'fish_2',
 'detection_t_27_x_286_y_402_z_1.tiff': 'fish_4',
 'detection_t_27_x_401_y_899_z_39.tiff': 'fish_3',
 'detection_t_27_x_466_y_360_z_25.tiff': 'fish_1',
 'detection_t_28_x_452_y_401_z_31.tiff': 'fish_2',
 'detection_t_28_x_501_y_892_z_21.tiff': 'fish_3',
 'detection_t_28_x_728_y_570_z_0.tiff': 'fish_1',
 'detection_t_29_x_672_y_498_z_3.tiff': 'fish_1',
 'detection_t_29_x_696_y_435_z_6.tiff': 'fish_3',
 'detection_t_30_x_254_y_811_z_0.tiff': 'fish_2',
 'detection_t_30_x_353_y_932_z_2.tiff': 'fish_1',
 'detection_t_30_x_355_y_783_z_7.tiff': 'fish_3',
 'detection_t_31_x_193_y_437_z_4.tiff': 'fish_0',
 'detection_t_31_x_540_y_877_z_26.tiff': 'fish_3',
 'detection_t_32_x_606_y_337_z_32.tiff': 'fish_3',
 'detection_t_33_x_158_y_362_z_23.tiff': 'fish_1',
 'detection_t_33_x_254_y_809_z_0.tiff': 'fish_2',
 'detection_t_33_x_614_y_330_z_33.tiff': 'fish_3',
 'detection_t_34_x_601_y_334_z_34.tiff': 'fish_3',
 'detection_t_35_x_625_y_407_z_15.tiff': 'fish_3',
 'detection_t_36_x_678_y_424_z_24.tiff': 'fish_3',
 'detection_t_37_x_375_y_865_z_28.tiff': 'fish_3',
 'detection_t_38_x_700_y_573_z_17.tiff': 'fish_3',
 'detection_t_39_x_193_y_348_z_17.tiff': 'fish_1',
 'detection_t_39_x_634_y_747_z_13.tiff': 'fish_3', 
 }