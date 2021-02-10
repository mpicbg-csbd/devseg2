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
from segtools.math_utils import conv_at_pts4, conv_at_pts_multikern
# from segtools import color
from segtools import torch_models
from segtools.point_matcher import match_points_single, match_unambiguous_nearestNeib
from segtools.ns2dir import load, save, flatten

# import predict
import collections
import isbi_tools


def setup_dirs(savedir):
  if savedir.exists(): shutil.rmtree(savedir)
  (savedir/'m').mkdir(exist_ok=True,parents=True)
  shutil.copy("/projects/project-broaddus/devseg_2/src/detector2.py",savedir)

def _config_example():
  config = SimpleNamespace()
  
  config.savedir = Path("An/Example/Directory")
  ## Build the network
  config.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
  
  ## fg/bg weights stuff for fluorescence images & class-imbalanced data
  class StandardGen(object):
    def __init__(self):
      self.x  = np.random.rand(512,512)
      self.yt = np.random.rand(512,512)
      self.w  = np.ones([512,512])
    
    def sample(self,time,train_mode=True):
      if train_mode:
        x  = np.random.rand(512,512)
        yt = np.random.rand(512,512)
        w  = np.ones([512,512])
      else:
        x  = np.random.rand(512,512)
        yt = np.random.rand(512,512)
        w  = np.ones([512,512])
      x = x[None,None]
      yt = yt[None,None]
      w = w[None,None]
      return x,yt,w
  
  def loss(net,s):
    y = net(s.x)
    loss = np.mean(y-s.yt)
    return y,loss

  config.loss = loss
  config.datagen = StandardGen()
  config.n_vali_samples = 10
  config.time_validate = 400
  config.time_total = 1_000
  config.save_every_n = 1
  config.lr = 2e-4
  config.vali_metrics = [lambda y,s : 1.0,]
  config.vali_minmax = [np.max,] # how to pick best params from vali_metric
  
  return config

def check_config(config):
  "trivial check that keys match"
  d = _config_example().__dict__
  e = config.__dict__

  missing = d.keys() - e.keys()
  print("missing: ", missing)
  assert len(missing) is 0, str(missing)

  extra = e.keys() - d.keys()
  print("extra: ", extra)
  assert len(extra) is 0, str(extra)

  for k,v in d.items():
    # assert type(d[k]) is type(e[k]), str(type(d[k]))
    if type(d[k]) is not type(e[k]): print(str(type(d[k])), "is not ", str(type(e[k])))
  print("Keys and Value Types Agree: Config Check Passed.")

def train_continue(config,weights_file):
  check_config(config)
  config.savedir = Path(config.savedir).resolve()
  m = SimpleNamespace()
  m.net = config.getnet().cuda()
  m.net.load_state_dict(torch.load(weights_file))
  m.opt = torch.optim.Adam(m.net.parameters(), lr = config.lr)
  ta = load(config.savedir / "ta/")

  td,vd = SimpleNamespace(),SimpleNamespace()
  td.s = SimpleNamespace() #config.datagen.sample(0)
  vd.s = SimpleNamespace() #config.datagen.sample(0,train_mode=0)
  td.s.x  = load(config.savedir / f"patches_train/x/a_i0.tif")
  td.s.yt = load(config.savedir / f"patches_train/yt/a_i0.tif")
  vd.s.x  = load(config.savedir / f"patches_vali/x/a_i0.tif")
  vd.s.yt = load(config.savedir / f"patches_vali/yt/a_i0.tif")

  T  = SimpleNamespace(m=m,vd=vd,td=td,ta=ta,c=config)
  return T

def train_init(config):
  check_config(config)
  config.savedir = Path(config.savedir).resolve()
  setup_dirs(config.savedir)
  ## load model, save receptive field, randomize weights...
  m = SimpleNamespace()
  m.net = config.getnet().cuda()
  m.opt = torch.optim.Adam(m.net.parameters(), lr = config.lr)
  # save(torch_models.receptivefield(m.net,kern=(3,5,5)),config.savedir / 'receptive_field.tif')
  torch_models.init_weights(m.net)
  print("Weights randomized...")
  
  td,vd = SimpleNamespace(),SimpleNamespace()
  td.s = config.datagen.sample(0)
  vd.s = config.datagen.sample(0,train_mode=0)
  save(_prepsave(td.s.x, proj=False),   config.savedir / f"patches_train/x/a_i0.tif")
  save(_prepsave(td.s.yt, proj=False),  config.savedir / f"patches_train/yt/a_i0.tif")
  save(_prepsave(vd.s.x, proj=False),   config.savedir / f"patches_vali/x/a_i0.tif")
  save(_prepsave(vd.s.yt, proj=False),  config.savedir / f"patches_vali/yt/a_i0.tif")

  ta = SimpleNamespace(i=1,losses=[],lr=config.lr,save_count=0,vali_scores=[],timings=[])
  ta.vali_names = [f.__name__ for f in config.vali_metrics]
  ta.best_weights_time = np.zeros(len(config.vali_metrics)+1)
  T  = SimpleNamespace(m=m,ta=ta,c=config,td=td,vd=vd)
  return T

def train(T):
  m  = T.m
  ta = T.ta
  config = T.c
  _losses = []

  for ta.i in range(ta.i,config.time_total+1):

    s = config.datagen.sample(ta.i)
    y,loss = config.loss(m.net,s) #config.datagen.sample(ta.i,train_mode=1))
    loss.backward()
    y = y.detach().cpu().numpy()

    if ta.i%5==0:
      m.opt.step()
      m.opt.zero_grad()
      _losses.append(float((loss/s.w.mean()).detach().cpu()))

    if ta.i%20==0:
      ta.timings.append(time())
      dt = 0 if len(ta.timings)==1 else ta.timings[-1]-ta.timings[-2]
      l = float(loss)
      ymax,ystd = float(y.max()), float(y.std())
      ytmax,ytstd = float(s.yt.max()), float(s.yt.std())
      print(f"i={ta.i:04d}, shape={y.shape}, loss={l:4f}, dt={dt:4f}, y={ymax:4f},{ystd:4f} yt={ytmax:4f},{ytstd:4f}",end='\r',flush=True)

    if ta.i % config.time_validate == 0:
      n = ta.i // config.time_validate
      ta.losses.append(np.mean(_losses[-config.time_validate:]))
      save(ta , config.savedir/"ta/")
      validate(T)
      check_weights_and_save(T,n)
      if ta.i % (config.time_validate*config.save_every_n) == 0:
        save_patches(T,n)

def _proj(x):
  assert x.ndim in [2,3]
  if x.ndim==2:
    return x
  else:
    return x.max(0)

def _prepsave(x,proj=True):
  "x is 2D/3D array with no channels or batches"
  if type(x) is torch.Tensor: x = x.detach().cpu().numpy()
  if type(x) is np.ndarray:
    if proj:
      return _proj(x.astype(np.float16))
    else:
      return x.astype(np.float16)
  assert False, "should be ndarray"

def save_patches(T,n):
  i=0
  with torch.no_grad():
    y  = T.m.net(torch.from_numpy(T.td.s.x).float().cuda()[None,None]).detach().cpu().numpy()[0][0]
    save(_prepsave(y),   T.c.savedir / f"patches_train/y/a{n:03d}_i{i}.tif")
    y  = T.m.net(torch.from_numpy(T.vd.s.x).float().cuda()[None,None]).detach().cpu().numpy()[0][0]
    save(_prepsave(y),   T.c.savedir / f"patches_vali/y/a{n:03d}_i{i}.tif")

def _vali_loss_single(T):
  s = T.c.datagen.sample(T.ta.i,train_mode=0)
  with torch.no_grad(): y,loss = T.c.loss(T.m.net,s)
  y = y.detach().cpu().numpy()
  loss = loss.cpu().numpy()
  res = [loss] + [f(y,s) for f in T.c.vali_metrics]
  return res

def validate(T):
  allscores = np.stack([_vali_loss_single(T) for _ in range(T.c.n_vali_samples)])
  T.ta.vali_scores.append(allscores.mean(0))

def check_weights_and_save(T,n):
  m = T.m
  ta = T.ta
  config = T.c
  torch.save(m.net.state_dict(), config.savedir / f'm/best_weights_latest.pt')
  ta.best_weights_latest = n
  _vs = np.array(ta.vali_scores)

  if _vs[:,0].min()==_vs[-1,0]:
    ta.best_weights_time[0] = n
    torch.save(m.net.state_dict(), config.savedir / f'm/best_weights_loss.pt')

  for i in range(1, _vs.shape[1]):
    valiname = T.ta.vali_names[i-1]
    f_max = config.vali_minmax[i-1]
    if f_max is None: continue
    if f_max(_vs[:,i])==_vs[-1,i]:
      ta.best_weights_time[i] = n
      torch.save(m.net.state_dict(), config.savedir / f'm/best_weights_{valiname}.pt')



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

Sun Mar 15 17:58:59 2020

Renamed to train_detector from detect_isbi
Renamed to detector from train_detector. This is because we're incorporating prediction and evaluation here as well.
The philosophy is that this module is equivalent to CPnet _the method_. And a method should know how to predict on new data, and how to evaluate and optimize itself.
There are many paths and values present in the various configs (trainer, evaluator, predictor) that are only used in a small number of places internally.
Some values, like paths probably should be controlled independently from the outside.
We don't need to choose where to load and save predictions in the initial config! This can/should be done later after loading the method.
But normalization really needs to be consistent everywhere and should be chosen before loading.

Should we split up initialization into generic and training-specific ?
NOTE: prediction and evaluation probably don't need to have an init, do they?
An init() is only useful when you want to return some piece of state that will be updated in place on subsequent calls.

If they have no init() do they need a config?
Good to have a config when you would otherwise have a really complex set of args in a function def, especially if repeated everywhere.
But prediction, etc are small and the args are almost totally unique.

Another un-nice-ness is how we sometimes use a folder of man_track.tif images and sometimes we use a pre-computed trajectory pickle.
The method shouldn't require a pre-computed traj pickle, but should accept one if you have it. 
It could also compute one for you if you don't have it? Then it would save it in the GT folder... (WARNING, these folders should maybe be read only?)

How should we specify the training data to load? 
- two directories and an (optional?) list of times?
- two lists of file names?
- note: need to keep control over exactly which images are in train vs vali sets, because for c. elegans it matters.
- an image dir with list of times and a pickle of points?
? what if the directory of data has nothing to do with time????!!!! This is a _detection_ method, time needn't be involved. This is isbi specific.
- a directory of raw images and matching (separate) points pickles?
- multiple directories with multiple lists of names?

really we should be able to specify _any_ of these things. So let's make a function which is flexible.
The only one which works under any circumstance is a list of file names.

The annoying thing about closures is that if i want to change code in a closure I have to re-initialize my training artifacts...
Is it the fact that the func is a closure? Or that it's an attribute?


Important thing: It may be quite helpful for our method to know about the simple image metadata like voxel size.
We should be able to take this as given. We can use it as a proxy for object shape and content auto-correlation, and thus for:
- patch size, net shape, receptive field, plm_footprint, rescale_for_matching, etc...
This is extra important when we're trying to apply the method to a wide variety of datasets without optimizing hyperparams jointly across all of them.

---

Sat Mar 21 13:36:00 2020

Something that leaves a bad taste in my mouth:
When i build a config and pass a config to detector, i am often uncertain about what paths it contains, and if those paths are up to date, and where the training data will be saved....
To my surprise when i look at our config function, _there are no paths inside_. That means they are all added from experiments!
This feels wrong and leads to my uncertainty...
_When do I need to know about paths_?
Just when beginning to train! That's when the artifacts are produced, and at no other time!
We could make a rule that configs are not allowed to contain paths, and that paths must be passed explicitly to the train() function.
What about other places where paths are used?
- in experiments we manage paths with snakemake and the deps object
- also for data loading we manage paths with the loader object...

2020-09-19

Let's see if this method still makes sense...
Trying to make `experiments2.job14_celegans()` work as a detector.

TODO: 
- [x] Take predict_raw from denoiser.

Tue Sep 29 13:13:23 2020

peak_local_max will overdetect if we have multiple neighbouring pixels with the exact same (max) value!
This shouldn't happen often, but it does because we have a weird problem with our precicted images where all pixels have the same value...
Is this a problem with the target? NO. Target is good!
Is this a problem with my input normalization ? clipping ? 

FRESH START DETECTOR 2 WITH LESS TRAINING DATA STUFF

Fri Nov 27 13:43:39 2020

Moved sampling / train/vali data creation to a separate experiment e20_trainset().
This reduces complexity, improves speed, makes the workflow more reproducible and makes training data easier to inspect and share.
We also reduced the number of time_* params available to the user, replacing them with sensible defaults.
We make the training process more standard, saving loss and vali_loss each epoch for easy plotting.
Iterating deterministically over the dataset ensure an even distribution, as compared to random sampling training patches from a flat distribution which results in a highly uneven sample dist! see [[Loic's Sandwich problem]].
More thoughts and considerations can be found in [[Notes on Sampling Training Data]].




"""









