import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import sys
# import ipdb
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

from ns2dir import ns2dir, dir2ns

import torch_models
import predict

notes = """
## Usage

import projection as p
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
  savedir = "/projects/project-broaddus/rachana_fish/e01/train1/"
  shutil.rmtree(savedir)
  os.mkdir(savedir)

  d = dir2ns('/projects/project-broaddus/rachana_fish/anno/train1/')
  d.img = normalize3(d.img,2,99.6)

  td = SimpleNamespace()  ## training/target data
  td.pts = np.floor(d.pts).astype(int)
  td.target = np.zeros(d.max.shape)
  td.target[tuple(td.pts.T)] = 1
  sh=np.array([30,30])
  def f(x):
    x = (x-sh/2)/8
    return np.exp(-(x*x).sum()/2)
  kern = np.array([f(x) for x in np.indices(sh).reshape((len(sh),-1)).T]).reshape(sh)
  td.target = convolve(td.target,kern)
  td.target = td.target/td.target.max()
  # td.target = convolve(td.target,np.ones((3,3)))
  # td.target = convolve(td.target,np.ones((3,3)))
  td.target = torch.from_numpy(td.target).float()
  td.target = td.target/td.target.max()
  td.input  = torch.from_numpy(d.img[None]).float()
  # ns2dir(td,'/projects/project-broaddus/rachana_fish/e01/train1/td2/')

  m = SimpleNamespace() ## model
  m.net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.Sequential).cuda()

  td.input = td.input.cuda()
  td.target = td.target.cuda()

  ta = SimpleNamespace()
  ta.savedir = Path(savedir)

  return m,d,td,ta

def train(m,td,ta):
  """
  requires m.net, td.input, td.target
  provides ta.losses and ta.i
  """

  _sh = np.array(td.input.shape)[1:]
  weight,global_avg = 1.0, td.target.mean()
  opt = torch.optim.Adam(m.net.parameters(), lr = 3e-5)

  try: ta.i
  except: 
    ta.i=0
    ta.losses = []

  for ta.i in range(ta.i,10000):
    c0 = np.floor(np.random.rand(3)*(_sh-(64,64,64))).astype(int)
    sz,sy,sx = slice(c0[0],c0[0]+64), slice(c0[1],c0[1]+64), slice(c0[2],c0[2]+64)
    ## NOTE: we ignore sz and use entire depth of image
    y  = m.net(td.input[None,:,:,sy,sx])
    # yp = (y*y.softmax(2)).sum(2)
    yp = y.max(2)[0]
    loss = torch.abs((yp-td.target[sy,sx])**1).mean() + weight*torch.abs((yp-global_avg)**1).mean()
    ta.losses.append(float(loss))
    print(float(loss))
    loss.backward()

    if ta.i%10==0:
      opt.step()
      opt.zero_grad()

    if ta.i%20==0:
      with warnings.catch_warnings():
        ns2dir(ta,ta.savedir / "ta/")
        tifffile.imsave(ta.savedir / f"img{ta.i//20:03d}.tif",yp[0,0].detach().cpu().numpy(),compress=9)

    if ta.i%500==0:
      weight *= 0.5

    if ta.i%1000==0:
      with torch.no_grad():
        k = ta.i//1000
        res = predict.apply_net_tiled_3d(m.net,td.input.detach().cpu().numpy())
        tifffile.imsave(ta.savedir / f"res{k:02d}.tif",res,compress=0)
        (ta.savedir/'m').mkdir(exist_ok=True)
        torch.save(m.net.state_dict(), ta.savedir / f'm/net{ta.i:03d}.pt')
