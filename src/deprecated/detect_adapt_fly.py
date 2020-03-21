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
from scipy.ndimage        import convolve

from segtools.numpy_utils import collapse2, normalize3, plotgrid
from segtools.math_utils import conv_at_pts2
from segtools import color
from segtools.defaults.ipython import moviesave

from ns2dir import load, save

import files
import torch_models
import predict
import point_matcher

import collections

def flatten(l):
  for el in l:
    if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
      yield from flatten(el)
    else:
      yield el


notes = """
## Usage

import detect_adapt_fly
m,d,td,ta = detect_adapt_fly.init("my_local_dir/experiment01/")
detect_adapt_fly.train(m,d,td,ta)

## Names

m  :: models (update in place)
vd :: validation data
td :: training data
ta :: training artifacts (update in place)

if you want to stop training just use Ctrl-C, it will restart at the same iteration where you left off because of state in ta and m.
"""

def setup_dirs(savedir):
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)
  (savedir/'m').mkdir(exist_ok=True)
  shutil.copy("/projects/project-broaddus/devseg_2/src/detect_adapt.py",savedir)

def init(savedir):
  savedir = Path(savedir).resolve()
  setup_dirs(savedir)

  ## training artifacts
  ta = SimpleNamespace()
  ta.savedir = savedir
  ta.use_denoised = False

  ## test data
  ta.valitimes = [0,25,49]
  vd = build_training_data(ta.valitimes, ta.use_denoised)

  ## training data
  ta.traintimes = np.r_[1:49]
  td = build_training_data(ta.traintimes, ta.use_denoised)
  # td.input  = td.raw[:,None]
  # td.target = td.target[:,None]
  save(td.target.max(1).astype(np.float16),ta.savedir/'ta/mx_vali/target.tif')

  ## model
  m = SimpleNamespace()
  m.net = torch_models.Unet3(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  torch_models.init_weights(m.net)

  return m,vd,td,ta

def train(m,vd,td,ta):
  """
  requires m.net, td.input, td.target
  provides ta.losses and ta.i
  """

  n_samples,n_chan = td.input.shape[:2]
  ta.n_samples = n_samples
  _sh = np.array(td.input.shape)[2:]
  # weight,global_avg = 1e1, 0.3 #td.target.mean()

  defaults = dict(i=0,losses=[],lr=2e-4,i_final=10**5,save_count=0,vali_scores=[],timings=[])
  ta.__dict__.update(**{**defaults,**ta.__dict__})

  thresh = 0.005
  def mkweights():
    x = td.target
    ## weight inversely prop to count
    ws = 1/np.array([(x<thresh).sum(), (x>thresh).sum()]).astype(np.float)
    # ws[0] /= 5 ## artificially suppress importance of bg points
    ws[0] = 0 ## 
    ws = ws / ws.mean() ## weight.mean = 1
    ta.ws = ws
    return ws
  ws = mkweights()
  print(ws)

  try: m.opt
  except: m.opt = torch.optim.Adam(m.net.parameters(), lr = ta.lr)
  
  w = torch.zeros(1,1,16,128,128).cuda() ## just create once
  ## linearly decay scalar input to value 1 after 3 epochs, then flat
  decayto1 = lambda x: x*(1-ta.i/(1000*10)) + ta.i/(1000*10) if ta.i<=(1000*10) else 1

  for ta.i in range(ta.i,ta.i_final):
    ta.timings.append(time())

    sc = np.random.randint(td.input.shape[0])
    _ipt = np.random.randint(0,len(td.gt[sc]))
    _pt = td.gt[sc][_ipt]
    # print(_pt.shape)
    # _pt = np.floor(np.random.rand(3)*(_sh-(16,128,128))).astype(int)
    _pt = _pt + (2*np.random.rand(3)-1)*(2,40,40) - (8,64,64) ## center patch and jitter
    _pt = _pt.clip(min=[0,0,0],max=[_sh-(16,128,128)])[0]
    _pt = _pt.astype(int)
    sz,sy,sx = slice(_pt[0],_pt[0]+16), slice(_pt[1],_pt[1]+128), slice(_pt[2],_pt[2]+128)

    x  = torch.from_numpy(td.input[[sc],:,sz,sy,sx]).float().cuda()
    yt = torch.from_numpy(td.target[[sc],:,sz,sy,sx]).float().cuda()
    y  = m.net(x)
    w[yt<thresh] = ws[0] # decayto1(ws[0])
    w[yt>thresh] = ws[1] # decayto1(ws[1])
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    loss.backward()

    if ta.i%10==0:
      m.opt.step()
      m.opt.zero_grad()

    ## monitoring training and validation

    ta.losses.append(float(loss/w.mean()))

    if ta.i%100==0:
      print(f"i={ta.i:04d}, loss={np.mean(ta.losses[-100:])}")
      with warnings.catch_warnings():
        save(ta , ta.savedir/"ta/")
        # save(y[0,0].detach().cpu().numpy().max(0)  , ta.savedir/f"epoch/y/a{ta.i//100:03d}.npy")
        # save(x[0,0].detach().cpu().numpy().max(0)  , ta.savedir/f"epoch/x/a{ta.i//100:03d}.npy")
        # save(yt[0,0].detach().cpu().numpy().max(0) , ta.savedir/f"epoch/yt/a{ta.i//100:03d}.npy")

    if ta.i%600==0:
      ta.save_count += 1
      with torch.no_grad():
        validate(m,vd,ta)
        torch.save(m.net.state_dict(), ta.savedir / f'm/net{ta.save_count:02d}.pt')

def validate(m,d,ta):
  vs = []
  with torch.no_grad():
    for i in range(d.input.shape[0]):
      res = predict.apply_net_tiled_3d(m.net,d.input[i])
      pts = peak_local_max(res[0],threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
      score3  = point_matcher.match_points_single(d.gt[i],pts,dub=3)
      score10 = point_matcher.match_points_single(d.gt[i],pts,dub=10)
      scr = (ta.i,i,score3,score10)
      print("e i match pred true", scr)
      vs.append(list(flatten(scr)))
      # save(res[0],ta.savedir / f"ta/pred/e{e}_i{i}.tif")
      save(res[0,16].astype(np.float16),ta.savedir / f"ta/ms/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0).astype(np.float16),ta.savedir / f"ta/mx/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(1).astype(np.float16),ta.savedir / f"ta/mx_y/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(2).astype(np.float16),ta.savedir / f"ta/mx_x/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].astype(np.float16),ta.savedir / f"ta/vali_full/e{ta.save_count:02d}_i{i}.tif")
  ta.vali_scores.append(vs)

## helper functions

def build_training_data(times,use_denoised=False):

  d = SimpleNamespace()
  if use_denoised:
    d.input = np.array([load(f"/projects/project-broaddus/devseg_2/e01/test/pred/Fluo-N3DH-CE/01/t{n:03d}.tif") for n in times])
  else:
    d.input = np.array([load(f"/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01/t{n:03d}.tif") for n in times])
  d.input = d.input / 1300 # normalize3(d.raw,2,99.6) # no clipping, min already set to zero

  def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)
  d.gt  = [mantrack2pts(load(f"/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01_GT/TRA/man_track{n:03d}.tif")) for n in times]

  s  = np.array([1,3,3])   ## sigma for gaussian
  ks = np.array([7,21,21]) ## kernel size. must be all odd
  sh = d.input[0].shape

  def place_kern_at_pts(pts):
    def f(x):
      x = x - (ks-1)/2
      return np.exp(-(x*x/s/s).sum()/2)
    kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
    kern = kern / kern.max()
    target = conv_at_pts2(pts,kern,sh,lambda a,b:np.maximum(a,b))
    return target

  d.target = np.array([place_kern_at_pts(pts) for pts in d.gt])
  d.target = d.target[:,None]
  d.input = d.input[:,None]

  return d

