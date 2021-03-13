# from segtools.ns2dir import load,save,flatten_sn,toarray
# from segtools import torch_models
# import torch
# from torch import nn
# from segtools.numpy_utils import normalize3, perm2, collapse2, splt

# from segtools import point_matcher
# from subprocess import run, Popen
# import shutil
# import json

# import tracking
# import denoiser, denoise_utils
# import detector #, detect_utils
# import detector2

# from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
# from glob import glob
# import os
# import re

# from scipy.ndimage.morphology import binary_dilation
# from skimage.util import view_as_windows
# from expand_labels_scikit import expand_labels
# from scipy.ndimage.morphology import distance_transform_edt
# from scipy.ndimage import label,zoom


# from datagen import * 
# mantrack2pts, place_gaussian_at_pts, normalize3, sample_flat, sample_content, sample_iterate, shape2slicelist, augment, weights
# from segtools.point_tools import trim_images_from_pts2


# import sys
# import itertools
# import warnings
# import os,shutil
# from time import time
# from  pathlib  import Path

# import tifffile
# import skimage.io    as io
# from scipy.ndimage        import zoom, label
# from skimage.segmentation import find_boundaries
# from scipy.ndimage        import convolve

# from segtools import torch_models

from experiments_common import rchoose, partition, shuffle_and_split, parse_pid, iterdims, savedir_global
import datagen as dgen

from segtools.render import rgb_max
from models import CenterpointModel, SegmentationModel, StructN2V
from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale
from tracking import nn_tracking_on_ltps, random_tracking_on_ltps
from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
from segtools.numpy_utils import normalize3, perm2, collapse2, splt, plotgrid
from segtools.ns2dir import load,save

from skimage.feature  import peak_local_max
import numpy as np
from numpy import r_,s_,ix_

import torch
from subprocess import run, Popen
from scipy.ndimage import zoom
import json
from types import SimpleNamespace
from glob import glob
from math import floor,ceil
import re

from e21_common import *


try:
    import gputools
except ImportError as e:
    print("Can't import gputools on non-gpu node...\n", e)

import ipdb
from time import time



savedir = savedir_global()
print("savedir:", savedir)


def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy("/projects/project-broaddus/devseg_2/src/e24_isbidet_AOT.py", "/projects/project-broaddus/devseg_2/src/e24_isbidet_AOT_copy.py")
  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 1 -t 6:00:00 --mem 128000 "
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = 'sbatch -J e24_{pid:03d} {_resources} -o slurm/e24_pid{pid:03d}.out -e slurm/e24_pid{pid:03d}.err --wrap \'python3 -c \"import e24_isbidet_AOT_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
  slurm = slurm.replace("{_resources}",_gpu)
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)

def myrun_slurm_entry(pid=0):
  myrun(pid)

  # (p0,p1),pid = parse_pid(pid,[2,19,9])
  # myrun([p0,p1,p2])
  # for p2,p3 in iterdims([2,5]):
  #   try:
  #   except:
  #     print("FAIL on pids", [p0,p1,p2,p3])



## Program start here

def myrun(pid=0):
  """
  v01 : refactor of e18. make traindata AOT.
    add `_train` 0 = predict only, 1 = normal init, 2 = continue
    datagen generator -> CenterpointGen, customizable loss and validation metrics, and full detector2 refactor.
  """

  (p0,p1),pid = parse_pid(pid,[19,2])
  params = SimpleNamespace(p0=p0,p1=p1)

  savedir_local = savedir / f'e24_isbidet_AOT/v01/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  params.info = info

  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)

  """
  EVERYTHING BELOW THIS POINT IS LIKE A WORKSPACE THAT CHANGES RAPIDLY & USES FUNCTIONS DEFINED ABOVE.
  """

  print("Running e24 with savedir: \n", savedir_local, flush=True)

  # n_raw  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
  # n_lab  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track
  # n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  # n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  # filenames  = [(n_raw.format(time=i),n_lab.format(time=i)) for i in range(info.start,info.stop)]
  # pointfiles = f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl"

  data = build_and_save_training_data(info)
  save(data, savedir_local/'data.pkl')

  if 1:
    sampler = FullImageNormSampler(data)
    CPNet = CenterpointModel(savedir_local, sampler)
    CPNet._init_params(info.ndim)
    save([CPNet.sample(i) for i in range(3)], CPNet.savedir/"traindata_pts.pkl")
    CPNet.train_cfig.time_total = 1_000
    CPNet.train(_continue=1)

  # if 0:
  #   MySN2V = StructN2V(savedir_local / "sn2v", info.ndim)
  #   _rawnames = sorted(glob(f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/t*.tif"))
  #   gap = floor(len(_rawnames)/7)
  #   MySN2V.dataloader(_rawnames[::gap])
  #   MySN2V.train(_continue=1)

  # net = CPNet.getnet().cuda()
  # CPNet = CenterpointModel(savedir_local, info)
  # CPNet.net.load_state_dict(torch.load(CPNet.savedir / "m/best_weights_loss.pt"))

  # N   = 7
  # gap = floor((info.stop-info.start)/N)
  # predict_times = range(info.start,info.stop,gap)
  # savetimes = predict_times

  # ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  # tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  # scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()






def build_training_data(info,max_size_in_MB=2_000):
  params = init_params(info.ndim)
  params = merge(params,cpnet_data_specialization(info))

  def _f():
    a = info.stop-info.start     ## N images
    b = np.prod(params.patch)    ## N pixels / patch
    c = max_size_in_MB*1_000_000 ## N Bytes
    d = 2 ## N float values / patch-pixel (raw + target)
    e = 2 ## N Bytes / float value
    f = c / (a*b*d*e) ## Avg number of patches / image
    return f
  n_patches_per_image = _f()

  n_raw  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
  n_lab  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track

  if info.index in [6,11,12,13,14,15,18]:
    n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
    n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"

  def f(i):
    print(i)
    raw = load(n_raw.format(time=i))[...]
    # raw = load(n_raw.format(time=i))
    raw = zoom(raw,params.zoom,order=1)
    # raw = gputools.scale(raw,params.zoom,interpolation='linear')
    p2,p99 = np.percentile(raw[::4,::4,::4],[2,99.4]) ## for speed
    raw = (raw-p2)/(p99-p2)
    # raw = normalize3(raw[::4,::4,::4],2,99.4,clip=False)
    pts = dgen.mantrack2pts(load(n_lab.format(time=i)))
    pts = (np.array(pts) * params.zoom).astype(np.int)
    n_total_objects = len(pts)
    target = dgen.place_gaussian_at_pts(pts,raw.shape,params.kern)
    slices = dgen.shape2slicelist(raw.shape,params.patch,divisible=(1,8,8)[-info.ndim:])
    N = min(floor(n_patches_per_image), len(slices))
    np.random.shuffle(slices)
    slices = slices[:N]
    # idx_slices = np.random.choice(len(slices), N, replace=False)
    # slices = slices[idx_slices]
    def g(s):
      tmean = target[s].mean()
      tmax  = target[s].max()
      return SimpleNamespace(tmean=tmean,tmax=tmax,raw=raw[s].copy().astype(np.float16),target=target[s].copy().astype(np.float16),slice=s)
    samples = [g(s) for s in slices]

    res = SimpleNamespace(samples=samples,n_total_objects=n_total_objects,p2=p2,p99=p99,pts=pts)
    return res
  
  _ttimes = np.r_[info.start:info.stop]
  print("_ttimes: ", _ttimes)
  data = [f(i) for i in _ttimes]

  self = SimpleNamespace()
  self.data=data
  self.times=_ttimes
  self.type="full-img-norm"
  self.info=info
  self.max_size_in_MB = max_size_in_MB
  return self

class FullImageNormSampler(object):

  def __init__(self,_data):
    "path to fullImageNorm dataset"
    if hasattr(_data,'type') and _data.type == "full-img-norm":
      self.data = _data
    elif type(_data) is str:
      self.data = load(_data)
    self.all_samples = [s for _,d in enumerate(self.data.data) for _,s in enumerate(d.samples)]

  def __getitem__(self,idx):
    return self.all_samples[idx]

  def __len__(self):
    return len(self.all_samples)

