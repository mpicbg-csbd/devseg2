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
  td = SimpleNamespace()
  print(n)
  if n<=7: ## [1,7]
    data  = np.load(f'/lustre/projects/project-broaddus/isbi_segway/dataset1/center_sigma{n}/data_label.npz')
    xs,ys = data['X'],data['Y']
  elif n<=11: ## [8,11]
    data  = np.load('/lustre/projects/project-broaddus/devseg_experiments/cl_datagen/data/data.npy')
    data  = collapse2(data[None],'ctsqzyx','q,ts,c,z,y,x') ## channel,time,sample,target type,z,y,x
    xs,ys = data[0],data[n-7] ## n param used here!
    xs,ys = xs[::2],ys[::2]
  elif n<=15:
    data  = np.load('/lustre/projects/project-broaddus/devseg_experiments/cl_datagen/data/data.npy')
    data  = data[:50]
    data  = collapse2(data[None],'ctsqzyx','q,ts,c,z,y,x') ## channel,time,sample,target type,z,y,x
    xs,ys = data[0],data[n-11] ## n param used here!
  elif n<=19:
    data  = np.load('/lustre/projects/project-broaddus/devseg_experiments/cl_datagen/data/data.npy')
    data  = data[np.r_[:50,50:100:10,100,125,150,175,189]]
    data  = collapse2(data[None],'ctsqzyx','q,ts,c,z,y,x') ## channel,time,sample,target type,z,y,x
    xs,ys = data[0],data[n-15] ## n param used here!
  elif n<=26:
    data  = np.load('/lustre/projects/project-broaddus/devseg_experiments/cl_datagen/data/data.npy')
    freq = 1/np.array(files.ls2)
    freq = freq/freq.sum()*100 ## this gives an average of around 60 patches
    m = np.random.rand(len(freq)) < freq
    data  = data[m]
    data  = collapse2(data[None],'ctsqzyx','q,ts,c,z,y,x') ## channel,time,sample,target type,z,y,x
    xs,ys = data[0],data[1 if n%2==0 else 3] ## n param used here!
  elif n<=30:
    data  = np.load('/lustre/projects/project-broaddus/devseg_experiments/cl_datagen/data/data.npy')
    freq = 1/np.array(files.ls2)
    freq = freq/freq[n] ## gives freq.clip(0,1).sum() ≈ 60
    vals,inds = np.unique(freq.clip(0,1).cumsum().astype(int),return_index=True)
    data  = data[inds]
    data  = collapse2(data[None],'ctsqzyx','q,ts,c,z,y,x') ## channel,time,sample,target type,z,y,x
    xs,ys = data[0],data[n-26] ## n param used here!
  elif n<=34:
    pass

  td.input  = torch.from_numpy(xs).float()
  td.target = torch.from_numpy(ys).float()
  td.input  = td.input.cuda()
  td.target = td.target.cuda()

  ## model
  m = SimpleNamespace()
  m.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  m.net.init_weights()

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

  defaults = dict(i=0,losses=[],lr=2e-4,i_final=n_samples*400+1,save_count=0,scores=[])
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

  opt  = torch.optim.Adam(m.net.parameters(), lr = ta.lr)
  indices = np.arange(n_samples)
  np.random.shuffle(indices)
  w = td.target[[0]].clone() ## just create once
  ## linearly decay scalar input to value 1 after 3 epochs, then flat
  decayto1 = lambda x: x*(1-ta.i/(n_samples*3)) + ta.i/(n_samples*3) if ta.i<=(n_samples*3) else 1

  for ta.i in range(ta.i,ta.i_final):

    ix = indices[ta.i%n_samples]
    x  = td.input[[ix]]
    yt = td.target[[ix]]
    y  = m.net(x)
    w[yt<thresh] = decayto1(ws[0])
    w[yt>thresh] = decayto1(ws[1])
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    loss.backward()

    if ta.i%10==0:
      opt.step()
      opt.zero_grad()

    ## extra stuff for monitoring training and shuffling

    if (ix%n_samples)==n_samples-1:
      np.random.shuffle(indices)

    ta.losses.append(float(loss/w.mean()))

    if ta.i%(n_samples//4)==0:
      print(f"i={ta.i:04d}, loss={np.mean(ta.losses[-(n_samples//4):])}")
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





