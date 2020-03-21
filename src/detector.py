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

# import predict
import collections
import isbi_tools

def setup_dirs(savedir):
  if savedir.exists(): shutil.rmtree(savedir)
  savedir.mkdir(exist_ok=True,parents=True)
  (savedir/'m').mkdir(exist_ok=True)
  shutil.copy("/projects/project-broaddus/devseg_2/src/detector.py",savedir)

# def load_object():
#   loader = SimpleNamespace()
#   loader.input_dir = "base/indir1_example"
#   loader.gt_dir = "base/gtdir_example"
#   # loader.

def eg_img_meta():
  img_meta = SimpleNamespace()
  img_meta.voxel_size = np.array([1.0,0.09,0.09])
  img_meta.time_step  = 1 ## 1.5 for second dataset?
  return img_meta

def config(img_meta):
  config = SimpleNamespace()
  # config.savedir = Path("detector_test/")

  print(img_meta)

  ## prediction / evaluation / training
  config.f_net_args    = ((16,[[1],[1]]), dict(finallayer=torch_models.nn.Sequential)) ## *args and **kwargs
  config.norm          = lambda img: normalize3(img,2,99.4,clip=True)
  config.plm_footprint = np.ones((3,10,10))
  config.threshold_abs = 0.1

  ## evaluation / training
  config.rescale_for_matching = (2,1,1)
  config.dub=10

  ## training only
  # kernz,kernxy = 1,7

  config.sigmas       = np.array([1,7,7])
  config.kernel_shape = np.array([43,43,43]) ## 7sigma in each direction
  config.sampler      = content_sampler
  config.patch_space  = np.array([16,128,128])
  config.patch_full   = np.array([1,1,16,128,128])
  config.fg_bg_thresh = np.exp(-16/2)
  config.bg_weight_multiplier = 0.0
  config.weight_decay = True
  config.i_final      = 31*600
  config.bp_per_epoch = 600

  return config

def _load_net(config):
  args,kwargs = config.f_net_args
  net = torch_models.Unet3(*args,**kwargs).cuda()
  try:
    net.load_state_dict(torch.load(config.best_model))
  except:
    torch_models.init_weights(net)
  return net

def train_init(config):
  config.savedir = Path(config.savedir).resolve()
  setup_dirs(config.savedir)

  ## load train and vali data
  vd,td = config.load_train_and_vali_data(config)

  ## model
  m = SimpleNamespace()
  m.net = _load_net(config)

  ## training params we don't want to control directly, and artifacts that change over training time.
  ta = SimpleNamespace(i=0,losses=[],lr=2e-4,i_final=config.bp_per_epoch*22,save_count=0,vali_scores=[],timings=[],heights=[])
  # defaults = dict(i=0,losses=[],lr=2e-4,i_final=config.bp_per_epoch*22,save_count=0,vali_scores=[],timings=[],heights=[])
  # ta.__dict__.update(**{**defaults,**ta.__dict__})

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

  try: m.opt
  except: m.opt  = torch.optim.Adam(m.net.parameters(), lr = ta.lr)
  
  for ta.i in range(ta.i,ta.i_final):
    ta.timings.append(time())

    x,yt,w = config.sampler(ta,td,config)
    x  = torch.from_numpy(x).float().cuda()
    yt = torch.from_numpy(yt).float().cuda()
    w  = torch.from_numpy(w).float().cuda()

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
        save(ta , config.savedir/"ta/")
        save(x[0,0].detach().cpu().numpy().max(0)  , config.savedir/f"epoch/x/a{ta.i//100:03d}.npy")
        save(y[0,0].detach().cpu().numpy().max(0)  , config.savedir/f"epoch/y/a{ta.i//100:03d}.npy")
        save(yt[0,0].detach().cpu().numpy().max(0) , config.savedir/f"epoch/yt/a{ta.i//100:03d}.npy")

    if ta.i%config.bp_per_epoch==0:
      ta.save_count += 1
      with torch.no_grad():
        validate(vd,T)
        torch.save(m.net.state_dict(), config.savedir / f'm/net{ta.save_count:02d}.pt')


## Task-Specific Stuff: Data Loading, Sampling, Weights, Validation, Etc

def validate(vd,T):
  m,ta,config = T.m, T.ta, T.c
  vs = []
  with torch.no_grad():
    for i in range(vd.input.shape[0]):
      res = torch_models.apply_net_tiled_3d(m.net,vd.input[i])
      pts = peak_local_max(res[0],threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
      score3  = match_unambiguous_nearestNeib(vd.gt[i],pts,dub=3,scale=config.rescale_for_matching)
      score10 = match_unambiguous_nearestNeib(vd.gt[i],pts,dub=10,scale=config.rescale_for_matching)
      s3  = [score3.n_matched,  score3.n_proposed,  score3.n_gt]
      s10 = [score10.n_matched, score10.n_proposed, score10.n_gt]
      print(ta.i,i,s3,s10)
      vs.append([s3,s10])
      # save(res[0],config.savedir / f"ta/pred/e{e}_i{i}.tif")
      # save(res[0,ta.patch_space[0]//2].astype(np.float16),config.savedir / f"ta/ms_z/e{ta.save_count:02d}_i{i}.tif")
      save(res[0].max(0).astype(np.float16),config.savedir / f"ta/mx_z/e{ta.save_count:02d}_i{i}.tif")
      # save(res[0].max(1).astype(np.float16),config.savedir / f"ta/mx_y/e{ta.save_count:02d}_i{i}.tif")
      # save(res[0].max(2).astype(np.float16),config.savedir / f"ta/mx_x/e{ta.save_count:02d}_i{i}.tif")
      # save(res[0].astype(np.float16),config.savedir / f"ta/vali_full/e{ta.save_count:02d}_i{i}.tif")
  ta.vali_scores.append(vs)

def _pts2target(list_of_pts,sh,config):

  s  = config.sigmas # np.array([1,3,3])   ## sigma for gaussian
  ks = config.kernel_shape # np.array([7,21,21]) ## kernel size. must be all odd
  
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

def content_sampler(ta,td,config):
  """
  sample near ground truth annotations (but flat over time)
  requires td.gt points
  """

  # w = np.array([x.shape[0] for x in td.gt])
  # w = 1/w
  # w = w/w.sum()
  # st = np.random.choice(np.arange(ta.dims['T']), p=w)
  st = np.random.randint(td.dims['T'])

  ## sample a region near annotations
  _ipt = np.random.randint(0,len(td.gt[st]))
  _pt = td.gt[st][_ipt] ## sample one centerpoint from the chosen time
  _pt = _pt + (2*np.random.rand(3)-1)*config.patch_space*0.1 ## jitter
  _pt = _pt - config.patch_space//2 ## center
  _pt = _pt.clip(min=[0,0,0],max=[td.in_space - config.patch_space])[0]
  _pt = _pt.astype(int)
  sz,sy,sx = [slice(_pt[i],_pt[i] + config.patch_space[i]) for i in range(3)]

  x = td.input[[st],:,sz,sy,sx]
  yt = td.target[[st],:,sz,sy,sx]
  w = weights(yt,ta,config)

  return x,yt,w

def flat_sampler(ta,td,config):
  "sample from everywhere independent of annotations"

  ## sample from anywhere
  _pt = np.floor(np.random.rand(3)*(td.in_space-config.patch_space)).astype(int)
  sz,sy,sx = [slice(_pt[i],_pt[i] + config.patch_space[i]) for i in range(3)]
  st = np.random.randint(td.dims['T'])

  x  = td.input[[st],:,sz,sy,sx]
  yt = td.target[[st],:,sz,sy,sx]
  w  = np.ones(x.shape)
  w  = weights(yt,ta,config)

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

## Prediction and evaluation

def predict_raw(config,net,img):
  img = config.norm(img)
  res = torch_models.apply_net_tiled_3d(net,img[None])[0]
  return res

def predict_pts(config,img):
  pts = peak_local_max(img,exclude_border=False,threshold_abs=config.threshold_abs,footprint=config.plm_footprint)
  return pts

def predict_matches(config, pts_gt, pts_pred):
  matches = match_unambiguous_nearestNeib(pts_gt,pts_pred,dub=config.dub,scale=config.rescale_for_matching)
  return matches


# def test_load_net_predict_and_evaluate_single_img():
#   _config = config(eg_img_meta())
#   _config.savedir = Path("../ex7/celegans_isbi/train1/7.0_1.0/01/").resolve()
#   _config.best_model = _config.savedir / 'm/net30.pt'
#   left  = "/projects/project-broaddus/rawdata/"
#   right = "MDA231/Fluo-C3DL-MDA231/01/t002.tif"
#   gtpts = load("/projects/project-broaddus/rawdata/MDA231/traj/Fluo-C3DL-MDA231/01_traj.pkl")[2]
#   load_net_predict_and_evaluate_single_img(_config,left,right,gtpts)
#   return _config

# def load_net_predict_and_evaluate_single_img(config, image_name_left, image_name_right, gtpts):
#   assert config.best_model
#   image_name_right = Path(image_name_right)
#   image_name = Path(image_name_left) / image_name_right

#   net = _load_net(config)
#   img = load(image_name)
#   # assert img.ndim>2 and np.sum(np.array(img.shape)>100)>1 ## should kinda look like a 3D image
#   img = config.norm(img)
#   res = torch_models.apply_net_tiled_3d(net,img[None])[0]
#   pts = peak_local_max(res,threshold_abs=0.1,exclude_border=False,footprint=config.plm_footprint)
#   # gt_pts = isbi_tools.mantrack2pts(load(gt_name))
#   unambiguous_matches = match_unambiguous_nearestNeib(gtpts,pts,dub=config.dub,scale=config.rescale_for_matching)
#   print("new {:6f} {:6f} {:6f}".format(unambiguous_matches.f1,unambiguous_matches.precision,unambiguous_matches.recall))

#   res = res.astype(np.float16)
#   save(res,                 config.savedir / 'pred' /    image_name_right.parent / image_name_right.name)
#   save(res.max(0),          config.savedir / 'mx_z' /    image_name_right.parent / image_name_right.name)
#   save(pts,                 config.savedir / 'pts' /     image_name_right.parent / image_name_right.name)
#   save(unambiguous_matches, config.savedir / 'matches' / image_name_right.parent / (image_name_right.stem + '.pkl'))


def total_matches(matchdir,name_total_scores,ptsdir,name_total_pts):
  o = evaluator.out
  match_list = [load(x) for x in matchdir.glob("t*.pkl")]
  match_scores = point_matcher.listOfMatches_to_Scores(match_list)
  save(match_scores, evaluator.name_total_scores)
  print("SCORES: ", match_scores)
  allpts = [load(x) for x in ptsdir.glob('t*.tif')]
  save(allpts, name_total_pts)


def rasterize_detections(config, traj, imgshape, savedir, pts_transform = lambda x: x,):
  for i in range(len(traj)):
    pts = pts_transform(traj[i])
    lab = pts2lab(pts,imgshape,config)
    save(lab, savedir / f'mask{i:03d}.tif')

def pts2lab(pts,shape,config):
  kerns = [np.zeros(3*config.sigmas) + j + 1 for j in range(len(pts))]
  lab   = conv_at_pts_multikern(pts,kerns,shape)
  lab   = lab.astype(np.uint16)
  return lab



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


"""









