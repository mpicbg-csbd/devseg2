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


def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

def setup_dirs(savedir):
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)
  (savedir/'m').mkdir(exist_ok=True)
  shutil.copy("/projects/project-broaddus/devseg_2/src/detect_isbi.py",savedir)

def init(trainer):
  savedir = Path(trainer.train_dir).resolve()
  setup_dirs(savedir)

  ## training artifacts
  ta = SimpleNamespace()
  ta.savedir = savedir
  ta.use_denoised = False

  ## Transfer from trainer
  ta.i_final = trainer.i_final
  ta.rescale_for_matching = trainer.rescale_for_matching
  ta.bp_per_epoch = trainer.bp_per_epoch

  ## test data
  ta.valitimes  = trainer.valitimes
  ta.traintimes = trainer.traintimes

  print("Training Times : ", trainer.traintimes)
  print("Vali Times : ", trainer.valitimes)

  vd = build_training_data(ta.valitimes, trainer)
  td = build_training_data(ta.traintimes, trainer)
  # save(td.target.max(1).astype(np.float16),ta.savedir/'ta/mx_vali/target.tif')

  ## model
  m = SimpleNamespace()
  args,kwargs = trainer.f_net_args
  m.net = torch_models.Unet3(*args,**kwargs).cuda()
  # m.net = trainer.f_net()
  torch_models.init_weights(m.net)

  if trainer.sampler == 'content':
    m.sampler = content_sampler
  else:
    assert trainer.sampler == 'flat'
    m.sampler = flat_sampler

  ## describe input shape
  ta.axes = "TCZYX"
  ta.dims = {k:v for k,v in zip(ta.axes, td.input.shape)}
  ta.in_samples     = td.input.shape[0]
  ta.in_chan        = td.input.shape[1]
  ta.in_space       = np.array(td.input.shape[2:])
  ta.patch_space = trainer.patch_space
  ta.patch_full  = trainer.patch_full


  T = SimpleNamespace(m=m,vd=vd,td=td,ta=ta)

  return T


@DeprecationWarning
def add_global_weights_to_ta(ta,td,trainer):
  thresh = trainer.fg_bg_thresh #0.005
  x = td.target
  ## weight inversely prop to count
  ws = 1/np.array([(x<thresh).sum(), (x>thresh).sum()]).astype(np.float)
  ws[0] *= trainer.bg_weight_multiplier

def build_training_data(times,trainer):
  d = SimpleNamespace()
  d.input = np.array([load(trainer.input_dir / f"t{n:03d}.tif") for n in times])
  d.input = trainer.norm(d.input)
  # d.gt    = [mantrack2pts(load(trainer.TRAdir / f"man_track{n:03d}.tif")) for n in times]
  d.gt    = np.array([trainer.traj_gt_train[n] for n in times])

  s  = trainer.sigmas # np.array([1,3,3])   ## sigma for gaussian
  ks = trainer.kernel_shape # np.array([7,21,21]) ## kernel size. must be all odd
  sh = d.input[0].shape

  def place_kern_at_pts(pts):
    def f(x):
      x = x - (ks-1)/2
      return np.exp(-(x*x/s/s).sum()/2)
    kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
    kern = kern / kern.max()
    target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
    return target

  d.target = np.array([place_kern_at_pts(pts) for pts in d.gt])

  d.target = d.target[:,None]
  d.input = d.input[:,None]

  # if trainer.wildcards.name == "trib_3d_downsample":
  #   ## downsample target to match images
  #   d.target = d.target[:,:,::3,::3,::3]
  #   d.gt  = d.gt / 3 ## this still works even though the array is weird! (a shape (n,) array with each element being another array of different shape)
  return d

def content_sampler(ta,td,trainer):
  "sample near ground truth annotations (but flat over time)"

  # w = np.array([x.shape[0] for x in td.gt])
  # w = 1/w
  # w = w/w.sum()
  # st = np.random.choice(np.arange(ta.dims['T']), p=w)
  st = np.random.randint(ta.dims['T'])

  ## sample a region near annotations
  _ipt = np.random.randint(0,len(td.gt[st]))
  _pt = td.gt[st][_ipt] ## sample one centerpoint from the chosen time
  _pt = _pt + (2*np.random.rand(3)-1)*ta.patch_space*0.1 ## jitter
  _pt = _pt - ta.patch_space//2 ## center
  _pt = _pt.clip(min=[0,0,0],max=[ta.in_space - ta.patch_space])[0]
  _pt = _pt.astype(int)
  sz,sy,sx = [slice(_pt[i],_pt[i] + ta.patch_space[i]) for i in range(3)]

  x = td.input[[st],:,sz,sy,sx]
  yt = td.target[[st],:,sz,sy,sx]
  w = weights(yt,ta,trainer)

  return x,yt,w

def flat_sampler(ta,td,trainer):
  "sample from everywhere independent of annotations"

  ## sample from anywhere
  _pt = np.floor(np.random.rand(3)*(ta.in_space-ta.patch_space)).astype(int)
  sz,sy,sx = [slice(_pt[i],_pt[i] + ta.patch_space[i]) for i in range(3)]
  st = np.random.randint(ta.dims['T'])

  x  = td.input[[st],:,sz,sy,sx]
  yt = td.target[[st],:,sz,sy,sx]
  w  = np.ones(x.shape)
  w = weights(yt,ta,trainer)

  return x,yt,w

def weights(yt,ta,trainer):
  "weight pixels in the slice based on pred patch content"
  thresh = trainer.fg_bg_thresh
  w = np.ones(yt.shape)
  m0 = yt<thresh
  m1 = yt>thresh
  if 0 < m0.sum() < m0.size:
    ws = 1/np.array([m0.mean(), m1.mean()]).astype(np.float)
    ws[0] *= trainer.bg_weight_multiplier
    ws /= ws.mean()
    if np.isnan(ws).any(): ipdb.set_trace()

    ## linearly decay scalar input to value 1 after 3 epochs, then flat
    if trainer.weight_decay:
      decayto1 = lambda x: x*(1-ta.i/(trainer.bp_per_epoch*3)) + ta.i/(trainer.bp_per_epoch*3) if ta.i<=(trainer.bp_per_epoch*3) else 1
    else:
      decayto1 = lambda x: x
    w[yt<thresh]  = decayto1(ws[0])
    w[yt>=thresh] = decayto1(ws[1])

  return w

def train(T,trainer):
  """
  requires m.net, td.input, td.target
  provides ta.losses and ta.i
  """

  m  = T.m
  vd = T.vd
  td = T.td
  ta = T.ta

  defaults = dict(i=0,losses=[],lr=2e-4,i_final=trainer.bp_per_epoch*22,save_count=0,vali_scores=[],timings=[],heights=[])
  ta.__dict__.update(**{**defaults,**ta.__dict__})

  try: m.opt
  except: m.opt  = torch.optim.Adam(m.net.parameters(), lr = ta.lr)
  
  for ta.i in range(ta.i,ta.i_final):
    ta.timings.append(time())

    x,yt,w = m.sampler(ta,td,trainer)
    x  = torch.from_numpy(x).float().cuda()
    yt = torch.from_numpy(yt).float().cuda()
    w = torch.from_numpy(w).float().cuda()

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
        save(ta , ta.savedir/"ta/")
        save(x[0,0].detach().cpu().numpy().max(0)  , ta.savedir/f"epoch/x/a{ta.i//100:03d}.npy")
        save(y[0,0].detach().cpu().numpy().max(0)  , ta.savedir/f"epoch/y/a{ta.i//100:03d}.npy")
        save(yt[0,0].detach().cpu().numpy().max(0) , ta.savedir/f"epoch/yt/a{ta.i//100:03d}.npy")

    if ta.i%trainer.bp_per_epoch==0:
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
      score3  = match_unambiguous_nearestNeib(d.gt[i],pts,dub=3,scale=ta.rescale_for_matching)
      score10 = match_unambiguous_nearestNeib(d.gt[i],pts,dub=10,scale=ta.rescale_for_matching)
      s3  = [score3.n_matched,  score3.n_proposed,  score3.n_gt]
      s10 = [score10.n_matched, score10.n_proposed, score10.n_gt]
      print(ta.i,i,s3,s10)
      vs.append([s3,s10])
      # save(res[0],ta.savedir / f"ta/pred/e{e}_i{i}.tif")
      # save(res[0,ta.patch_space[0]//2].astype(np.float16),ta.savedir / f"ta/ms_z/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0).astype(np.float16),ta.savedir / f"ta/mx_z/e{ta.save_count:02d}_i{i}.tif")
      # save(res[0].max(1).astype(np.float16),ta.savedir / f"ta/mx_y/e{ta.save_count:02d}_i{i}.tif")
      # save(res[0].max(2).astype(np.float16),ta.savedir / f"ta/mx_x/e{ta.save_count:02d}_i{i}.tif")
      # save(res[0].astype(np.float16),ta.savedir / f"ta/vali_full/e{ta.save_count:02d}_i{i}.tif")
  ta.vali_scores.append(vs)


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


"""









