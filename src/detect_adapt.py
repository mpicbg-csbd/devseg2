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
from scipy.ndimage        import convolve

from segtools.numpy_utils import collapse2, normalize3, plotgrid
from segtools.math_utils import conv_at_pts
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

import detect
m,d,td,ta = detect.init()
detect.train(m,d,td,ta)

## Names

m  :: models (update in place)
d  :: raw data
td :: training data
ta :: training artifacts (update in place)

if you want to stop training just use Ctrl-C, it will up at the same iteration where you left off because of state in ta and m.

"""

def setup_dirs(savedir):
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)
  (savedir/'m').mkdir(exist_ok=True)
  # (savedir/'epoch').mkdir(exist_ok=True)

def init(savedir, n):
  savedir = Path(savedir).resolve()
  setup_dirs(savedir)

  ## training artifacts
  ta = SimpleNamespace()
  ta.savedir = savedir

  ## test data
  d = SimpleNamespace()
  d.raw = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t{n:03d}.tif") for n in [0,10,100,189]])
  d.raw = normalize3(d.raw,2,99.6)
  def pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)
  d.gt  = [pts(load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01_GT/TRA/man_track{n:03d}.tif")) for n in [0,10,100,189]]
  
  ## training data

  s  = np.array([1,3,3])   ## sigma for gaussian
  ks = np.array([7,21,21]) ## kernel size. must be all odd

  def place_kern_at_pts(pts):
    def f(x):
      x = x - (ks-1)/2
      return np.exp(-(x*x/s/s).sum()/2)
    kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
    kern = kern / kern.max()
    # target = np.zeros(d.raw[0].shape)
    # target[tuple(pts.T)] = 1
    # target = convolve(target,kern,mode='constant',cval=0)
    target = conv_at_pts(pts,kern)
    target = target[3:,10:,10:]
    sh = np.array(d.raw[0].shape) - target.shape
    target = np.pad(target,[(0,sh[0]), (0,sh[1]), (0,sh[2])], mode='constant')
    return target

  target = np.array([place_kern_at_pts(d.gt[i]) for i in range(4)])

  td = SimpleNamespace()
  td.input  = torch.from_numpy(d.raw).float()[:,None]
  td.target = torch.from_numpy(target).float()[:,None]
  td.input  = td.input.cuda()
  td.target = td.target.cuda()

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

  defaults = dict(i=0,losses=[],lr=2e-4,i_final=10**5,save_count=0,scores=[])
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

  # indices = np.arange(n_samples)
  # np.random.shuffle(indices)

  for ta.i in range(ta.i,ta.i_final):

    _pt = np.floor(np.random.rand(3)*(_sh-(16,128,128))).astype(int)
    sz,sy,sx = slice(_pt[0],_pt[0]+16), slice(_pt[1],_pt[1]+128), slice(_pt[2],_pt[2]+128)
    sc = np.random.randint(4)

    # ix = indices[ta.i%n_samples]
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

    ## extra stuff for monitoring training and shuffling

    # if (ix%n_samples)==n_samples-1:
    #   np.random.shuffle(indices)

    ta.losses.append(float(loss/w.mean()))
    # ta.losses.append(float(loss))

    if ta.i%100==0:
      print(f"i={ta.i:04d}, loss={np.mean(ta.losses[-100:])}")
      with warnings.catch_warnings():
        save(ta , ta.savedir/"ta/")
        # save(y[0,0,8].detach().cpu().numpy()  , ta.savedir/f"epoch/y/a{ta.i//20:03d}.npy")
        # save(x[0,0,8].detach().cpu().numpy()  , ta.savedir/f"epoch/x/a{ta.i//20:03d}.npy")
        # save(yt[0,0,8].detach().cpu().numpy() , ta.savedir/f"epoch/yt/a{ta.i//20:03d}.npy")

    if ta.i%(ta.i_final//41)==0:
      ta.save_count += 1
      with torch.no_grad():
        predict4(m,d,ta)
        torch.save(m.net.state_dict(), ta.savedir / f'm/net{ta.save_count:02d}.pt')

def predict4(m,d,ta):
  with torch.no_grad():
    for i in range(4):
      res = predict.apply_net_tiled_3d(m.net,d.raw[i,None])
      pts = peak_local_max(res[0],threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
      score3  = point_matcher.match_points_single(d.gt[i],pts,dub=3)
      score10 = point_matcher.match_points_single(d.gt[i],pts,dub=10)
      scr = (ta.i,i,score3,score10)
      print("e i match pred true", scr)
      ta.scores.append(list(flatten(scr)))
      # save(res[0],ta.savedir / f"ta/pred/e{e}_i{i}.tif")
      save(res[0,16],ta.savedir / f"ta/ms/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0),ta.savedir / f"ta/mx/e{ta.save_count:02d}_i{i}.tif")


