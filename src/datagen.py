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

import ipdb
import itertools
from math import floor,ceil
import numpy as np
# from scipy.ndimage import label,zoom
from skimage.feature  import peak_local_max
from skimage.measure  import regionprops
from pathlib import Path
from segtools.ns2dir import load,save,flatten_sn,toarray
from segtools import torch_models
from types import SimpleNamespace
import torch
from segtools.numpy_utils import normalize3, perm2, collapse2, splt
from segtools import point_matcher
from subprocess import run, Popen
import shutil
from segtools.point_tools import trim_images_from_pts2
from scipy.ndimage import zoom
import json
from scipy.ndimage.morphology import binary_dilation

import tracking
import denoiser, denoise_utils
import detector #, detect_utils
import detector2

from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
from glob import glob
import os
import re
from skimage.util import view_as_windows
from expand_labels_scikit import expand_labels


from scipy.ndimage.morphology import distance_transform_edt

from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.morphology import thin, binary_opening




def pts2target_gaussian(list_of_pts,sh,sigmas):
  target = np.array([place_gaussian_at_pts(pts,sh,sigmas) for pts in list_of_pts])
  return target

def pts2target_gaussian_sigmalist(list_of_pts,sh,list_of_sigmas):
  return np.array([pts2target_gaussian([x],sh,sig)[0] for x,sig in zip(list_of_pts,list_of_sigmas)])

def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)



# def sample(raw,lab):
#   ## make patches
#   if info.ndim==2:
#     a,b = raw.shape
#     if a<512:
#       _a,_b = 8*(a//8), 8*(b//8)
#       raw = raw[None,:_a,:_b]
#       lab = lab[None,:_a,:_b]
#     else:
#       # sa,sb = int((a-512)/(a/512-1)), int((b-512)//(b/512-1))
#       sa,sb = 512,512
#       raw = view_as_windows(raw,(512,512),(sa,sb)).reshape((-1,512,512))
#       lab = view_as_windows(lab,(512,512),(sa,sb)).reshape((-1,512,512))
    
#   if info.ndim==3:
#     a,b,c = raw.shape
#     _z = 16 if a>16 else a
#     raw = view_as_windows(raw,(_z,128,128),(_z,128,128)).reshape((-1,_z,128,128))
#     lab = view_as_windows(lab,(_z,128,128),(_z,128,128)).reshape((-1,_z,128,128))

#   ## remove the useless patches
#   dims = tuple(range(1,lab.ndim))
#   m = lab.sum(dims)>0
#   if m.sum()==0: return [],[]
#   raw_patches = raw[m]
#   lab_patches = lab[m]
#   return  raw_patches, lab_patches


# def lab2target(lab):
#   y = lab.copy()
#   pts = mantrack2pts(y)

#   if myname=="MSC":
#     y = y>0
#     for _ in [1,2,3,4,5]: y = binary_opening(y)
#     # y = y.astype(np.float32)
#     # y = convolve(y,np.ones([5,5]))
#     # y = gaussian_filter(y.astype(np.float32),sigma=3)
#     # y = gputools.convolve()
#     # for i in range(3): y=binary_dilation(y)
#     # y = expand_labels(y,3)>0
#     y = distance_transform_edt(y)
#   else:
#     y = detector2.place_gaussian_at_pts(pts,y.shape,res.kern)
#     # y = expand_labels(y,10)>0
#     # y = distance_transform_edt(y)
#   return y, pts


# def load_single_isbi(n,info):
#   raw = load(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=n))
#   lab = load(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}_GT/TRA/" + info.man_track.format(time=n))
#   if lab.sum()==0: return None
#   if res.zoom:
#     raw = zoom(raw, res.zoom, order=1)
#     lab = zoom(lab, res.zoom, order=0)
#   raw = normalize3(raw,2,99.4,clip=False)
#   raw_patches,lab_patches = sample(raw,lab)
#   if len(lab_patches)==0: return None
#   raw_patches, lab_patches = zip(*[augment(x,y) for x,y in zip(raw_patches,lab_patches)])
#   target_patches, gt_pts   = zip(*[lab2target(x) for x in lab_patches])
#   return raw_patches, target_patches, gt_pts


from segtools.point_tools import trim_images_from_pts2


# def load_single_goodBoundaries(rawname,labname,_zoom,_augment=False):
#   raw = load(rawname)
#   lab = load(labname)

#   if lab.sum()==0: return None
#   if _zoom:
#     raw = zoom(raw, _zoom, order=1)
#     lab = zoom(lab, _zoom, order=0)
#   # if pid==26: raw = normalize3(raw,2,100,clip=False)
#   else: raw = normalize3(raw,2,99.4,clip=False)
#   target, _ = lab2target(lab) ## ignore gt points for now!
#   raw_patches,lab_patches = sample(raw,lab)
#   if len(raw_patches)==0: return None
#   target_patches, _ = sample(target,lab)
#   if _augment:
#     raw_patches, lab_patches, target_patches = zip(*[augment(x,y,z) for x,y,z in zip(raw_patches,lab_patches,target_patches)])
#   _, gt_pts   = zip(*[lab2target(x) for x in lab_patches])
#   return raw_patches, target_patches, gt_pts


def place_gaussian_at_pts(pts,sh,sigmas):
  s  = np.array(sigmas)
  ks = np.ceil(s*7).astype(np.int) #*6 + 1 ## gaurantees odd size and thus unique, brightest center pixel
  ks = ks - ks%2 + 1## enfore ODD shape so kernel is centered! (grow even dims by 1 pix)
  # ks = (s*7).astype(np.int) ## must be ODD
  def f(x):
    x = x - (ks-1)/2
    return np.exp(-(x*x/s/s).sum()/2)
  kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
  kern = kern / kern.max()
  target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
  return target

def augment(x,*ys):
  noiselevel = 0.2
  ndim = x.ndim ## WARNING. SLIGHT DIFFERENCE FROM DETECTOR.PY.
  ## TODO: this only works when number of channels==1. probably want indep noise for each channel.
  # x += np.random.uniform(0,noiselevel,(1,)*ndim)*np.random.uniform(-1,1,x.shape)
  # x += np.random.uniform(0,noiselevel,(1,))*np.random.uniform(-1,1,x.shape)


  ## evenly sample all random flips and 90deg XY rotations (not XZ or YZ rotations)
  ## TODO: double check the groups here.
  ## TODO: maybe this could all be shorter with modular arithmetic. dim -2 is always Y, dim -1 is always X.
  if ndim==3:
    space_dims = {'Z':0,'Y':1,'X':2}
  elif ndim==2:
    space_dims = {'Y':0,'X':1}
  
  for d in space_dims.values():
    if np.random.rand() < 0.5:
      x  = np.flip(x,d)
      ys = tuple(np.flip(y,d) for y in ys)
  if np.random.rand() < 0.5 and x.shape[space_dims['Y']]==x.shape[space_dims['X']]:
    x  = x.swapaxes(space_dims['Y'],space_dims['X'])
    ys = tuple(y.swapaxes(space_dims['Y'],space_dims['X']) for y in ys)

  x = x.copy()
  ys = tuple(y.copy() for y in ys)
  return (x,) + ys

def augment2(x,*ys):
  noiselevel = 0.2
  ndim = x.ndim
  ## TODO: this only works when number of channels==1. probably want indep noise for each channel.
  # x += np.random.uniform(0,noiselevel,(1,)*ndim)*np.random.uniform(-1,1,x.shape)
  # x += np.random.uniform(0,noiselevel,(1,))*np.random.uniform(-1,1,x.shape)

  ## TODO: special "content aware" augmentation involving simplistic intensity-based segmentation techniques


  ## evenly sample all random flips and 90deg XY rotations (not XZ or YZ rotations)
  ## TODO: continuous rotations (in 3D!) (with anisotropic voxels!) (do we even need bounds checking?)
  ## TODO: maybe this could all be shorter with modular arithmetic. dim -2 is always Y, dim -1 is always X.
  if ndim==3:
    space_dims = {'Z':0,'Y':1,'X':2}
  elif ndim==2:
    space_dims = {'Y':0,'X':1}
  
  for d in space_dims.values():
    if np.random.rand() < 0.5:
      x  = np.flip(x,d)
      ys = tuple(np.flip(y,d) for y in ys)
  if np.random.rand() < 0.5 and x.shape[space_dims['Y']]==x.shape[space_dims['X']]:
    x  = x.swapaxes(space_dims['Y'],space_dims['X'])
    ys = tuple(y.swapaxes(space_dims['Y'],space_dims['X']) for y in ys)

  x = x.copy()
  ys = tuple(y.copy() for y in ys)
  return (x,) + ys


def sample_flat(data,_patch):
  _p1 = np.random.randint(len(data))
  ndim = data[_p1].target.ndim
  _p2 = np.floor(np.random.rand(ndim)*(data[_p1].target.shape - _patch)).astype(int).clip(min=[0,]*ndim)
  # ipdb.set_trace()
  ss  = tuple(slice(a,b) for a,b in zip(_p2,_p2+_patch))
  x  = data[_p1].raw[ss].copy()
  yt = data[_p1].target[ss].copy()
  return x,yt

def sample_content(data,_patch):
  _p1 = np.random.randint(len(data)) # timepoint
  _p2 = np.random.randint(len(data[_p1].pts)) # object center at timepoint
  pt  = data[_p1].pts[_p2]
  ndim = len(pt)

  pt  = pt + (2*np.random.rand(ndim))*_patch*0.1 ## jitter by 10%
  pt  = pt - _patch//2 ## center
  _max = np.clip([data[_p1].target.shape - _patch],a_min=[0]*ndim,a_max=None)
  pt  = pt.clip(min=[0]*ndim,max=_max)[0] ## clip to bounds
  pt  = pt.astype(int)
  ss  = tuple(slice(pt[i],pt[i] + _patch[i]) for i in range(len(pt)))

  x = data[_p1].raw[ss].copy()
  yt = data[_p1].target[ss].copy()
  return x,yt

def sample_iterate(data,_patch,n):
  totalsize = np.prod([d.target.size for d in data])
  totalpatches = ceil(totalsize / np.prod(_patch))
  pass



def weights(yt,time,thresh=1.0,decayTime=None,bg_weight_multiplier=1.0):
  "weight pixels in the slice based on pred patch content"
  
  w = np.ones(yt.shape)
  m0 = yt<thresh # background
  m1 = yt>thresh # foreground
  if 0 < m0.sum() < m0.size:
    # ws = 1/np.array([m0.mean(), m1.mean()]).astype(np.float) ## REMOVE. WE DON'T WANT TO BALANCE FG/BG. WE WANT TO EMPHASIZE FG.
    ws = [1,1] * np.array(1.)
    ws[0] *= bg_weight_multiplier
    ws /= ws.mean()
    if np.isnan(ws).any(): ipdb.set_trace()

    if decayTime:
      ## decayto1 :: linearly decay scalar x to value 1 after 3 epochs, then const
      decayto1 = lambda x: x*(1-time/decayTime) + time/decayTime if time<=decayTime else 1
      ws[0] = decayto1(ws[0])
      ws[1] = decayto1(ws[1])
      
    w[yt<thresh]  = ws[0]
    w[yt>=thresh] = ws[1]

  return w








