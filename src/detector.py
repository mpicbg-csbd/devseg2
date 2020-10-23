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
  shutil.copy("/projects/project-broaddus/devseg_2/src/detector.py",savedir)

def _config_example():
  config = SimpleNamespace()
  
  config.savedir = Path("An/Example/Directory")
  ## Build the network
  config.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
  ## detector target kernel shape
  # config.sigmas       = np.array([1,7,7])
  # config.kernel_shape = np.array([43,43,43]) ## 7 sigma in each direction?
  config.rescale_for_matching = [2,1,1]
  ## fg/bg weights stuff for fluorescence images & class-imbalanced data
  config.fg_bg_thresh = np.exp(-16/2)
  config.bg_weight_multiplier = 0.0
  config.weight_decay = True
  config.time_weightdecay = 400 # for pixelwise weights
  ## image sampling
  config.sampler      = content_sampler ## sampler :: ta,td,config -> x,yt,w
  config.patch_space  = np.array([16,128,128])
  config.batch_shape  = np.array([1,1,16,128,128])
  config.batch_axes   = "BCZYX"
  # generic
  # config.times = [10,100,500,4000,100_000]
  # config.times = [10,100,200,400,1_000]
  config.time_agg = 10 # aggregate gradients before backprop
  config.time_print = 100
  config.time_savecrop = 200
  config.time_validate = 400
  config.time_total = 1_000
  config.lr = 2e-4

  config.load_train_and_vali_data = lambda config: ("td","vd") # config -> traindata, validata
  return config

def check_config(config):
  "trivial check that keys match"
  d = config_example().__dict__
  e = config.__dict__

  missing = d.keys() - e.keys()
  print("missing: ", missing)
  assert len(missing) is 0, str(missing)

  extra = e.keys() - d.keys()
  print("extra: ", extra)
  assert len(extra) is 0, str(extra)

  for k,v in d.items(): 
    assert type(d[k]) is type(e[k]), str(type(d[k]))
  print("Keys and Value Types Agree: Config Check Passed.")


def train_continue(config,weights_file):
  check_config(config)
  config.savedir = Path(config.savedir).resolve()
  m = SimpleNamespace()
  m.net = config.getnet().cuda()
  m.net.load_state_dict(torch.load(weights_file))
  m.opt = torch.optim.Adam(m.net.parameters(), lr = config.lr)
  ta = load(config.savedir / "ta/")
  td,vd = config.load_train_and_vali_data(config)
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
  ta = SimpleNamespace(i=1,losses=[],lr=config.lr,save_count=0,vali_scores=[],timings=[],heights=[])
  td,vd = config.load_train_and_vali_data(config)
  T  = SimpleNamespace(m=m,vd=vd,td=td,ta=ta,c=config)
  return T

def train(T):
  """
  requires m.net, td.input, td.target
  provides ta.losses and ta.i
  """

  m  = T.m
  vd = T.vd
  td = T.td
  ta = T.ta
  config = T.c
  
  for ta.i in range(ta.i,config.time_total+1):
    x,yt,w = config.sampler(ta,td,config)
    x  = torch.from_numpy(x).float().cuda()
    yt = torch.from_numpy(yt).float().cuda()
    w  = torch.from_numpy(w).float().cuda()

    ## put patches through the net, then backprop
    y    = m.net(x)
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    loss.backward()

    if ta.i%config.time_agg==0:
      m.opt.step()
      m.opt.zero_grad()

    if ta.i%config.time_loss==0:
      ta.losses.append(float((loss/w.mean()).detach().cpu()))
      ta.heights.append(float(y.max().detach().cpu()))

    if ta.i%config.time_print==0:
      ta.timings.append(time())
      dt = 0 if len(ta.timings)==1 else ta.timings[-1]-ta.timings[-2]
      l  = np.mean(ta.losses[-10:])
      ymax,ystd = float(y.max()), float(y.std())
      ytmax,ytstd = float(yt.max()), float(yt.std())
      print(f"i={ta.i:04d}, shape={x.shape}, loss={l:4f}, dt={dt:4f}, y={ymax:4f},{ystd:4f} yt={ytmax:4f},{ytstd:4f}", flush=True)

    def _proj(x):
      assert x.ndim in [2,3]
      if x.ndim==2:
        return x
      else:
        return x.max(0)

    if ta.i%config.time_savecrop==0:
      with warnings.catch_warnings():
        n = ta.i//config.time_savecrop
        save(ta , config.savedir/"ta/")
        _stack = _proj(x[0,0].detach().cpu().numpy().astype(np.float16)) ; save(_stack, config.savedir/f"epoch/x/a{n:03d}.npy")
        _stack = _proj(y[0,0].detach().cpu().numpy().astype(np.float16)) ; save(_stack, config.savedir/f"epoch/y/a{n:03d}.npy")
        _stack = _proj(yt[0,0].detach().cpu().numpy().astype(np.float16)); save(_stack, config.savedir/f"epoch/yt/a{n:03d}.npy")
        _stack = _proj(w[0,0].detach().cpu().numpy().astype(np.float16)) ; save(_stack, config.savedir/f"epoch/w/a{n:03d}.npy")

    if ta.i%config.time_validate==0:
      n = ta.i//config.time_validate
      validate(vd,T)


def validate(vd,T):
  ## Task-Specific Stuff: Data Loading, Sampling, Weights, Validation, Etc
  m,ta,config = T.m, T.ta, T.c
  vs = []
  n = ta.i//config.time_validate
  for i in range(vd.input.shape[0]):
    # res = torch_models.apply_net_tiled_3d(m.net,vd.input[i])
    if vd.input[0].ndim==4:
      dims = "CZYX"
      footy = np.ones((3,8,8))
    else:
      dims = "CYX"
      footy = np.ones((8,8))
    
    res = predict_raw(m.net,vd.input[i],dims)
    valloss = (np.abs(res-vd.target[i])**2).mean()

    ## detection scores
    pts = peak_local_max(res[0],threshold_abs=.2,exclude_border=False,footprint=footy)
    score3  = match_unambiguous_nearestNeib(vd.gt[i],pts,dub=3,scale=config.rescale_for_matching)
    score10 = match_unambiguous_nearestNeib(vd.gt[i],pts,dub=10,scale=config.rescale_for_matching)
    s3  = [score3.f1, score3.n_matched,  score3.n_proposed,  score3.n_gt]
    s10 = [score10.f1, score10.n_matched, score10.n_proposed, score10.n_gt]
    st  = f"{ta.i:5d} {i} {score10.f1:6.3f} {score10.n_matched:4d} {score10.n_proposed:4d} {score10.n_gt:4d}"
    print(st)

    ## save detections and loss, but only the basic info to save space
    vs.append({'3':score3.f1, '10':score10.f1, 'loss':valloss})
    # save(res[0].max(0).astype(np.float16),config.savedir / f"mx_z/e{n:03d}_i{i}.tif")

  ta.vali_scores.append(vs)

  torch.save(m.net.state_dict(), config.savedir / f'm/best_weights_latest.pt')
  ta.best_weights_latest = n

  valilosses = [sum([x['loss'] for x in xx]) for xx in ta.vali_scores]
  if np.min(valilosses)==valilosses[-1]:
    print(f"New best mse weights at {n}")
    ta.best_weights_mse = n
    torch.save(m.net.state_dict(), config.savedir / f'm/best_weights_mse.pt')

  valilosses = [sum([x['10'] for x in xx]) for xx in ta.vali_scores]
  if np.max(valilosses)==valilosses[-1]:
    print(f"New best f1 weights at {n}")
    ta.best_weights_f1 = n
    torch.save(m.net.state_dict(), config.savedir / f'm/best_weights_f1.pt')


def pts2target(list_of_pts,sh,sigmas):
  s  = np.array(sigmas)
  ks = (s*7).astype(np.int)
  # s  = config.sigmas # np.array([1,3,3])   ## sigma for gaussian
  # ks = config.kernel_shape # np.array([7,21,21]) ## kernel size. must be all odd
  
  def place_kern_at_pts(pts):
    def f(x):
      x = x - (ks-1)/2
      return np.exp(-(x*x/s/s).sum()/2)
    kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
    kern = kern / kern.max()
    target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
    return target

  target = np.array([place_kern_at_pts(pts) for pts in list_of_pts])
  return target

def pts2target_many(list_of_pts,sh,list_of_sigmas):
  return np.array([pts2target([x],sh,sig)[0] for x,sig in zip(list_of_pts,list_of_sigmas)])


def content_sampler(ta,td,config):
  """
  sample near ground truth annotations (but flat over time)
  requires td.gt points
  """

  st = np.random.randint(td.input.shape[0])

  size_space = np.array(td.input.shape[2:])
  ndim = len(size_space)
  _ipt = np.random.randint(0,len(td.gt[st]))
  _pt = td.gt[st][_ipt] ## sample one centerpoint from the chosen time
  _pt = _pt + (2*np.random.rand(ndim)-1)*config.patch_space*0.1 ## jitter
  _pt = _pt - config.patch_space//2 ## center
  _pt = _pt.clip(min=[0]*ndim,max=[size_space - config.patch_space])[0]
  _pt = _pt.astype(int)
  ss = tuple(slice(_pt[i],_pt[i] + config.patch_space[i]) for i in range(ndim))
  ss = np.s_[[st],:] + ss
  
  x  = td.input[ss]
  yt = td.target[ss]
  x,yt = augment(x,yt)
  w  = weights(yt,ta,config)

  return x,yt,w

def flat_sampler(ta,td,config):
  "sample from everywhere independent of annotations"

  ## sample from anywhere
  size_space = np.array(td.input.shape[2:])
  ndim = len(size_space)
  _pt = np.floor(np.random.rand(ndim)*(size_space - config.patch_space)).astype(int)
  st = np.random.randint(td.input.shape[0]) 
  ss = tuple(slice(_pt[i],_pt[i] + config.patch_space[i]) for i in range(ndim))
  ss = np.s_[[st],:] + ss

  x  = td.input[ss]
  yt = td.target[ss]
  # w  = np.ones(x.shape)
  x,yt = augment(x,yt)
  w  = weights(yt,ta,config)

  return x,yt,w

def weights(yt,ta,trainer):
  "weight pixels in the slice based on pred patch content"
  thresh = trainer.fg_bg_thresh
  w = np.ones(yt.shape)
  m0 = yt<thresh # background
  m1 = yt>thresh # foreground
  if 0 < m0.sum() < m0.size:
    ws = 1/np.array([m0.mean(), m1.mean()]).astype(np.float)
    ws[0] *= trainer.bg_weight_multiplier
    ws /= ws.mean()
    if np.isnan(ws).any(): ipdb.set_trace()

    if trainer.weight_decay:
      t0 = trainer.time_weightdecay
      ## decayto1 :: linearly decay scalar x to value 1 after 3 epochs, then const
      decayto1 = lambda x: x*(1-ta.i/(t0*3)) + ta.i/(t0*3) if ta.i<=(t0*3) else 1
    else:
      decayto1 = lambda x: x
    w[yt<thresh]  = decayto1(ws[0])
    w[yt>=thresh] = decayto1(ws[1])

  return w

def augment(x,y):
  noiselevel = 0.2
  ndim = x.ndim-2
  ## TODO: this only works when number of channels==1. probably want indep noise for each channel.
  x += np.random.uniform(0,noiselevel,(1,)*ndim)*np.random.uniform(-1,1,x.shape)

  ## evenly sample all random flips and 90deg XY rotations (not XZ or YZ rotations)
  ## TODO: double check the groups here.
  ## TODO: maybe this could all be shorter with modular arithmetic. dim -2 is always Y, dim -1 is always X.
  if ndim==3:
    space_dims = {'Z':2,'Y':3,'X':4}
  elif ndim==2:
    space_dims = {'Y':2,'X':3}
  
  for d in space_dims.values():
    if np.random.rand() < 0.5:
      x  = np.flip(x,d)
      y  = np.flip(y,d)
  if np.random.rand() < 0.5:
    x = x.swapaxes(space_dims['Y'],space_dims['X'])
    y = y.swapaxes(space_dims['Y'],space_dims['X'])

  x = x.copy()
  y = y.copy()

  return x,y

def predict_raw(net,img,dims,**kwargs3d):
  """
  each elem of N dimension sent to gpu separately.
  When possible, try to make the output dimensions match the input dimensions by e.g. removing singleton dims.
  """
  assert dims in ["NCYX","NBCYX","CYX","ZYX","CZYX","NCZYX","NZYX","YX"]

  with torch.no_grad():
    if dims=="NCYX":
      def f(i): return net(torch.from_numpy(img[[i]]).cuda().float()).cpu().numpy()
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="NBCYX":
      def f(i): return net(torch.from_numpy(img[i]).cuda().float()).cpu().numpy()
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="CYX":
      res = torch_models.apply_net_2d(net,img,)
      # res = net(torch.from_numpy(img[None]).cuda().float()).cpu().numpy()[0]
    if dims=="YX":
      res = torch_models.apply_net_2d(net,img[None],)[0]
      # res = net(torch.from_numpy(img[None,None]).cuda().float()).cpu().numpy()[0,0]
    if dims=="ZYX":
      ## assume 1 channel. remove after prediction.
      res = torch_models.apply_net_tiled_3d(net,img[None],**kwargs3d)[0]
    if dims=="CZYX":
      res = torch_models.apply_net_tiled_3d(net,img)
    if dims=="NCZYX":
      def f(i): return torch_models.apply_net_tiled_3d(net,img[i],**kwargs3d)[0]
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="NZYX":
      def f(i): return torch_models.apply_net_tiled_3d(net,img[i,None],**kwargs3d)[0]
      res = np.array([f(i) for i in range(img.shape[0])])

  return res







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


"""









