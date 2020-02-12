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
from segtools.math_utils import conv_at_pts4
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
  ta.valitimes = [0,5,33,34,100,189]
  d = build_training_data(ta.valitimes, '02', ta.use_denoised)

  ## training data
  ta.traintimes = [0,5,33,34,100,189]
  d2 = build_training_data(ta.traintimes, '02', ta.use_denoised)
  td = SimpleNamespace()
  td.input  = torch.from_numpy(d2.raw).float()[:,None]
  td.target = torch.from_numpy(d2.target).float()[:,None]
  td.input  = td.input.cuda()
  td.target = td.target.cuda()
  save(d2.target.max(1).astype(np.float16),ta.savedir/'ta/mx_vali/target.tif')

  ## model
  m = SimpleNamespace()
  m.net = torch_models.Unet3(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  torch_models.init_weights(m.net)

  return m,d,td,ta

def train(m,d,td,ta):
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

  thresh = 0.01
  def mkweights():
    x = td.target
    ## weight inversely prop to count
    ws = 1/np.array([(x<thresh).sum(), (x>thresh).sum()]).astype(np.float)
    ws[0] /= 5 ## artificially suppress importance of bg points
    ws = ws / ws.mean() ## weight.mean = 1
    ta.ws = ws
    return ws
  ws = mkweights()
  print(ws)

  opt = torch.optim.Adam(m.net.parameters(), lr = ta.lr)
  w = torch.zeros(1,1,16,128,128).cuda() ## just create once
  ## linearly decay scalar input to value 1 after 3 epochs, then flat
  decayto1 = lambda x: x*(1-ta.i/(1000*3)) + ta.i/(1000*3) if ta.i<=(1000*3) else 1

  for ta.i in range(ta.i,ta.i_final):
    ta.timings.append(time())

    _pt = np.floor(np.random.rand(3)*(_sh-(16,128,128))).astype(int)
    sz,sy,sx = slice(_pt[0],_pt[0]+16), slice(_pt[1],_pt[1]+128), slice(_pt[2],_pt[2]+128)
    sc = np.random.randint(td.input.shape[0])

    x  = td.input[[sc],:,sz,sy,sx]
    yt = td.target[[sc],:,sz,sy,sx]

    y  = m.net(x)
    w[yt<thresh] = decayto1(ws[0])
    w[yt>thresh] = decayto1(ws[1])
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    loss.backward()

    if ta.i%10==0:
      opt.step()
      opt.zero_grad()

    ## monitoring training and validation

    ta.losses.append(float(loss/w.mean()))

    if ta.i%100==0:
      print(f"i={ta.i:04d}, loss={np.mean(ta.losses[-100:])}")
      with warnings.catch_warnings():
        save(ta , ta.savedir/"ta/")
        # save(y[0,0,8].detach().cpu().numpy()  , ta.savedir/f"epoch/y/a{ta.i//20:03d}.npy")
        # save(x[0,0,8].detach().cpu().numpy()  , ta.savedir/f"epoch/x/a{ta.i//20:03d}.npy")
        # save(yt[0,0,8].detach().cpu().numpy() , ta.savedir/f"epoch/yt/a{ta.i//20:03d}.npy")

    if ta.i%1000==0:
      ta.save_count += 1
      with torch.no_grad():
        validate(m,d,ta)
        torch.save(m.net.state_dict(), ta.savedir / f'm/net{ta.save_count:02d}.pt')

def validate(m,d,ta):
  vs = []
  with torch.no_grad():
    for i in range(d.raw.shape[0]):
      res = predict.apply_net_tiled_3d(m.net,d.raw[i,None])
      pts = peak_local_max(res[0],threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
      score3  = point_matcher.match_points_single(d.gt[i],pts,dub=3)
      score10 = point_matcher.match_points_single(d.gt[i],pts,dub=10)
      scr = (ta.i,i,score3,score10)
      print("e i match pred true", scr)
      vs.append(list(flatten(scr)))
      # save(res[0],ta.savedir / f"ta/pred/e{e}_i{i}.tif")
      save(res[0,16],ta.savedir / f"ta/ms/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0),ta.savedir / f"ta/mx/e{ta.save_count:02d}_i{i}.tif")
  ta.vali_scores.append(vs)

## helper functions

def build_training_data(times,dset='01',use_denoised=False):

  d = SimpleNamespace()
  if use_denoised:
    d.raw = np.array([load(f"/projects/project-broaddus/devseg_2/e01/test/pred/Fluo-N3DH-CE/{dset}/t{n:03d}.tif") for n in times])
  else:
    d.raw = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{dset}/t{n:03d}.tif") for n in times])
  d.raw = normalize3(d.raw,2,99.6)
  def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)
  d.gt  = [mantrack2pts(load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{dset}_GT/TRA/man_track{n:03d}.tif")) for n in times]

  s  = np.array([1,3,3])   ## sigma for gaussian
  ks = np.array([7,21,21]) ## kernel size. must be all odd
  sh = d.raw[0].shape

  def place_kern_at_pts(pts):
    def f(x):
      x = x - (ks-1)/2
      return np.exp(-(x*x/s/s).sum()/2)
    kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
    kern = kern / kern.max()
    target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
    return target

  d.target = np.array([place_kern_at_pts(pts) for pts in d.gt])

  return d

