"""
Train CPNet on all ISBI datasets.
Build the training data ahead-of-time (AOT) as opposed to concurrently while training (just-in-time appraoch).
Also, use models.CenterpointModel for training, as opposed to custom `train()` method.
"""

import os
from pathlib import Path

import pandas
from collections import Counter

from experiments_common import rchoose, partition, shuffle_and_split, parse_pid, iterdims, savedir_global
import datagen as dgen

# from segtools.render import rgb_max
# from models import CenterpointModel
# from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale
# from tracking import nn_tracking_on_ltps, random_tracking_on_ltps
from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
# from segtools.numpy_utils import normalize3, perm2, collapse2, splt, plotgrid
from segtools.ns2dir import load,save

from skimage.feature  import peak_local_max
import numpy as np

from numpy import r_,s_,ix_
import torch

import pandas
from pandas import DataFrame


from subprocess import run, Popen
from scipy.ndimage import zoom
import json
from types import SimpleNamespace
from glob import glob
from math import floor,ceil
import re

# from e21_common import *
import shutil

try:
    import gputools
except ImportError as e:
    print("Can't import gputools on non-gpu node...\n", e)

import ipdb
from time import time

import matplotlib

from segtools import torch_models
import inspect

import matplotlib.pyplot as plt

savedir = savedir_global()
print("savedir:", savedir)



## stable. utility funcs

def sample2RawPng(sample):
  pngraw = _png(sample.raw)
  pnglab = _png(sample.lab)
  m = pnglab==0
  pngraw[~m] = (0.25*pngraw+0.75*pnglab)[~m]
  return pngraw

def _png(x):

  x = x.copy()

  def colorseg(seg):
    cmap = np.random.rand(256,3).clip(min=0.1)
    cmap[0] = (0,0,0)
    cmap = matplotlib.colors.ListedColormap(cmap)

    m = seg!=0
    seg[m] %= 254 ## we need to save a color for black==0
    seg[m] += 1
    rgb = cmap(seg)
    return rgb

  _dtype = x.dtype
  D = x.ndim

  if D==3:
    a,b,c = x.shape
    yx = x.max(0)
    zx = x.max(1)
    zy = x.max(2)
    x0 = np.zeros((a,a), dtype=x.dtype)
    x  = np.zeros((b+a+1,c+a+1), dtype=x.dtype)
    x[:b,:c] = yx
    x[b+1:,:c] = zx
    x[:b,c+1:] = zy.T
    x[b+1:,c+1:] = x0

  assert x.dtype == _dtype

  if 'int' in str(x.dtype):
    x = colorseg(x)
  else:
    x = norm_minmax01(x)
    x = plt.cm.gray(x)
  
  x = (x*255).astype(np.uint8)

  if D==3:
    x[b,:] = 255 # white line
    x[:,c] = 255 # white line

  return x

def norm_minmax01(x):
  """
  c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
  https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
  """
  mx = x.max()
  mn = x.min()
  if mx==mn: 
    return x-mx
  else: 
    return (x-mn)/(mx-mn)

def norm_affine01(x,lo,hi):
  if hi==lo: 
    return x-hi
  else: 
    return (x-lo)/(hi-lo)

def norm_percentile01(x,p0,p1):
  lo,hi = np.percentile(x,[p0,p1])
  return norm_affine01(x,lo,hi)

def take_n_evenly_spaced_items(item_list,N):
  M = len(item_list)
  # assert N<=M
  y = np.linspace(0,M-1,N).astype(np.int)
  # ss = [slice(y[i],y[i+1]) for i in range(M)]
  return np.array(item_list,dtype=np.object)[y]
  # return ss

def divide_evenly_with_min1(n_samples,n_bins):
  N = n_samples
  M = n_bins
  assert N>=M
  y = np.linspace(0,N,M+1).astype(np.int)
  ss = [slice(y[i],y[i+1]) for i in range(M)]
  return ss

def bytes2string(nbytes):
  if nbytes < 10**3:  return f"{nbytes} B"
  if nbytes < 10**6:  return f"{nbytes/10**3} KB"
  if nbytes < 10**9:  return f"{nbytes/10**6} MB"
  if nbytes < 10**12: return f"{nbytes/10**9} GB"

def file_size(root):
  "works for files or directories (recursive)"
  root = Path(root)
  # ipdb.set_trace()
  if root.is_dir():
    # https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python/1392549
    return sum(f.stat().st_size for f in root.glob('**/*') if f.is_file())
  else:
    # https://stackoverflow.com/questions/2104080/how-can-i-check-file-size-in-python
    return os.stat(fname).st_size

def strDiskSizePatchFrame(df):
  _totsize = (2+1+2) * df['shape'].apply(np.prod).sum()
  return bytes2string(_totsize)


## stable. job-specific utility funcs.

def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy("/projects/project-broaddus/devseg_2/src/e24_isbidet_AOT.py", "/projects/project-broaddus/devseg_2/src/temp/e24_isbidet_AOT_copy.py")
  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 1 -t 6:00:00 --mem 128000 "
  _cpu  = "-n 1 -t 1:00:00 -c 1 --mem 128000 "
  slurm = 'sbatch -J e24_{pid:03d} {_resources} -o slurm/e24_pid{pid:03d}.out -e slurm/e24_pid{pid:03d}.err --wrap \'python3 -c \"import temp.e24_isbidet_AOT_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
  slurm = slurm.replace("{_resources}",_cpu)
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)

def myrun_slurm_entry(pid=0):
  build_trainingdata(pid)

  # (p0,p1),pid = parse_pid(pid,[2,19,9])
  # myrun([p0,p1,p2])
  # for p2,p3 in iterdims([2,5]):
  #   try:
  #   except:
  #     print("FAIL on pids", [p0,p1,p2,p3])

def pid2params(pid):
  (p0,p1),pid = parse_pid(pid,[19,2])
  savedir_local = savedir / f'e24_isbidet_AOT/v01/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  return SimpleNamespace(**locals())

def isbiInfo_to_filenames(info):
  n_raw  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
  n_lab  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track
  # if info.index in [6,11,12,13,14,15,18]:
  n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  filenames_raw = [(n_raw.format(time=i),n_lab.format(time=i)) for i in range(info.start,info.stop)]
  filename_ltps = f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl"
  return filenames_raw, filename_ltps

def samples2pngSN(samples):
  s = SimpleNamespace()
  for sam in samples:
    s.__dict__[f't{sam.time}_r'] = sample2RawPng(sam)
    # s.__dict__[f't{sam.time}_r'] = _png(sam.raw.copy())
    # s.__dict__[f't{sam.time}_l'] = _png(sam.lab.copy())
    s.__dict__[f't{sam.time}_t'] = _png(sam.target.copy())
  return s

def patch_shapes_and_centers_from_slices(slices):
  starts = np.array([[s.start for s in _slices] for _slices in slices])
  stops  = np.array([[s.stop for s in _slices] for _slices in slices])
  patch_shapes   = stops-starts
  patch_centers  = starts + patch_shapes//2
  return patch_shapes, patch_centers

def norm_samples_raw(fullsamples):
  _raw = np.concatenate([s.raw.flatten() for s in fullsamples])
  p2,p99 = np.percentile(_raw,[2,99.4])
  for s in fullsamples:
    s.raw = norm_affine01(s.raw,p2,p99)
  return fullsamples



## infrequent updates. parameter memory.

def init_params(ndim):
  P = SimpleNamespace()
  if ndim==2:
    P.zoom  = (1,1) #(0.5,0.5)
    P.kern  = [5,5]
    P.patch = (256,256)
  elif ndim==3:
    P.zoom   = (1,1,1) #(1,0.5,0.5)
    P.kern   = [2,5,5]
    P.patch  = (16,128,128)
  P.nms_footprint = P.kern
  P.patch = np.array(P.patch)
  return P

def cpnet_data_specialization(info):
  isbiname = info.isbiname
  p = SimpleNamespace()

  if isbiname in ["Fluo-N3DH-CE", "Fluo-C3DH-A549", "Fluo-C3DH-A549-SIM", "Fluo-N3DH-CHO", "Fluo-N3DH-SIM+", ]:
    p.zoom = {3:(1,0.5,0.5), 2:(0.5,0.5)}[info.ndim]
  if isbiname=="Fluo-N3DL-TRIF":
    p.kern = [3,3,3]
    p.zoom = (0.5,0.5,0.5)
    p.patch = (64,64,64)
  if isbiname=="Fluo-C3DH-H157":
    p.zoom = (1/4,1/8,1/8)
  if isbiname=="Fluo-C2DL-MSC":
    a,b = info.shape
    p.zoom = {'01':(1/4,1/4), '02':(128/a, 200/b)}[info.dataset]
    ## '02' rescaling is almost exactly isotropic while still being divisible by 8.
  if isbiname=="DIC-C2DH-HeLa":
    # p.kern = [7,7]
    p.zoom = (0.5,0.5)
  if isbiname=="Fluo-N3DL-DRO":
    pass
    # p.kern = [1,3,3]
    # cfig.bg_weight_multiplier=0.0
    # cfig.weight_decay = False
  p.sparse  = True if info.isbiname in ["Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF",] else False
  return p




## Create Dataset Version 1.0. 

def filenames2garlist(filenames):
  "takes list tuples [(raw-name, lab-name)] "
  def f(a,b):
    raw  = load(a)
    lab  = load(b)
    time = int(re.search(r'(\d+)\.(tif|zar)', a).group(1))
    return SimpleNamespace(raw=raw,lab=lab,n_raw=a,n_lab=b,time=time)
  garlist = [f(*x) for x in filenames]
  return garlist

def gar2samples(gar, nsamples, params):
  print(f"running {inspect.currentframe().f_code.co_name}, T={gar.time}", flush=True)

  raw = gar.raw
  lab = gar.lab
  pts = gar.pts

  raw = zoom(raw,params.zoom,order=1)
  lab = zoom(lab,params.zoom,order=0)
  # raw = gputools.scale(raw,params.zoom,interpolation='linear')
  pts = (np.array(pts) * params.zoom).astype(np.int)

  _rawsmall = raw if raw.ndim==2 else raw[::2,::6,::6]
  p2,p99 = np.percentile(_rawsmall,[2,99.4]) ## for speed
  raw = (raw-p2)/(p99-p2)

  target = dgen.place_gaussian_at_pts(pts,raw.shape,params.kern)
  # pts = dgen.mantrack2pts(lab)

  ## subsample dataset. no overlapping patches.
  # slices = dgen.shape2slicelist(raw.shape,params.patch,divisible=(1,8,8)[-raw.ndim:])
  slices = dgen.tile_multidim(raw.shape, params.patch, (2,10,10)[-raw.ndim:])
  c = Counter([tuple(s.stop-s.start for s in x.a) for x in slices])
  output_slices = [x.b for x in slices]
  res = dgen.find_points_within_patches3(pts,output_slices)
  _ressum = res.sum(0) ## number of points per patch
  pts_per_slice = [pts[np.argwhere(res[:,i])[:,0]] for i in range(len(slices))]

  slices = [s for i,s in enumerate(slices) if _ressum[i]>0]
  # ipdb.set_trace()

  np.random.shuffle(slices)
  # slices = slices[:nsamples]

  def f(ss):
    r = raw[ss.a].copy().astype(np.float16)     ## use full 'input' slice.a , but use 'mask' slice.b in loss
    l = lab[ss.a].copy().astype(np.uint8)       ## use full 'input' slice.a , but use 'mask' slice.b in loss
    t = target[ss.a].copy().astype(np.float16)  ## use full 'input' slice.a , but use 'mask' slice.b in loss

    _fp = np.ones([3,5,5],dtype=np.bool) if r.ndim==3 else np.ones([5,5],dtype=np.bool)
    p = peak_local_max(t.astype(np.float32)[ss.c], threshold_abs=0.9, footprint=_fp)
    if p.shape[0]>0: ipdb.set_trace()
    return SimpleNamespace(raw=r,lab=l,target=t,pts=p,time=gar.time,slice=ss,)

  return [f(ss) for ss in slices]

def gar2samples_sampleFirst(gar, nsamples, params):
  # print("time, size, npts", gar.time, gar.raw.shape, gar.pts.shape[0])
  print(f"running {inspect.currentframe().f_code.co_name}, T={gar.time}", flush=True)
  raw = gar.raw
  lab = gar.lab
  pts = gar.pts

  # ## norm for speed ... Doesn't work. super slow.
  # _rawsmall = raw if raw.ndim==2 else raw[::2,::6,::6]
  # p2,p99 = np.percentile(_rawsmall,[2,99.4])
  # raw = (raw-p2)/(p99-p2)

  ## subsample dataset. no overlapping patches.
  patchsize = (np.array(params.patch) // params.zoom).astype(int)
  # slices = dgen.shape2slicelist(raw.shape,patchsize,divisible=(1,8,8)[-raw.ndim:])
  DD = int(8 // params.zoom[-1])

  _ftile  = lambda a,b,c : dgen.tile1d_predict(a,b,c,divisible=DD) #tile1d_predict(end,length,border,divisible=8)
  borders = (2,10,10)[-raw.ndim:]
  slices  = dgen.tile_multidim(raw.shape, patchsize, borders, f_singledim=_ftile) #, (4,10,10)[-raw.ndim:])

  # ipdb.set_trace()
  c = Counter([tuple(s.stop-s.start for s in x.a) for x in slices])
  # slices = [x.a for x in slices]
  output_slices = [x.b for x in slices]
  res = dgen.find_points_within_patches3(pts,output_slices)

  # shapes, centers = patch_shapes_and_centers_from_slices(slices)
  # pts_per_patch = dgen.find_points_within_patches(centers, pts, shapes[0])
  
  # slices = [s for i,s in enumerate(slices) if len(pts_per_patch[i])>0]
  _ressum = res.sum(0) ## number of points per patch
  slices = [s for i,s in enumerate(slices) if _ressum[i]>0]
  np.random.shuffle(slices)
  slices = slices[:nsamples]

  def f(ss):
    # print("XXX 1")
    r = raw[ss.a].copy() ## use full 'input' slice.a 
    l = lab[ss.a].copy() ## use full 'input' slice.a 
    
    ## NO MORE REFERENCES TO RAW OR LAB!

    # print("XXX 2")
    ## zoom
    r = zoom(r,params.zoom,order=1)
    l = zoom(l,params.zoom,order=0)

    ## build target
    # print("XXX 3")
    p = dgen.mantrack2pts(l)
    t = dgen.place_gaussian_at_pts(p,r.shape,params.kern)

    r = r.astype(np.float16)
    l = l.astype(np.uint8)
    t = t.astype(np.float16)

    # print("XXX 4")
    return SimpleNamespace(raw=r,lab=l,target=t,pts=p,time=gar.time,slice=ss,)

  return [f(ss) for ss in slices]

def build_trainingdata(pid=0):

  P = pid2params(pid)
  # print(json.dumps(P.info.__dict__,sort_keys=True, indent=2, default=str), flush=True)
  info = P.info
  
  print(f"""
    Begin pid {pid} ... {info.isbiname} / {info.dataset}.
    Savedir is {P.savedir_local}
    """)

  ## initial patch shit
  params  = init_params(info.ndim)
  _params = cpnet_data_specialization(info)
  for k in _params.__dict__.keys(): params.__dict__[k] = _params.__dict__[k]

  ## automatic patch sizing
  # params.patch = (256,256) 
  max_size_in_MB = 2000
  N = 100 if info.ndim==2 else 20 ## max number of images
  N_png_samples = 10

  print("Start to load images.")

  image_names,ltps_name = isbiInfo_to_filenames(info)
  _N = min(N,len(image_names))
  garlist = filenames2garlist(take_n_evenly_spaced_items(image_names,_N))
  ltps = load(ltps_name)
  for g in garlist: g.pts = ltps[g.time]

  print("Done loading images. Start to chop into patches.")

  ## now stop depending on info... only on virtual_img_data
  def _f():
    a = len(garlist)   ## N images
    b = np.prod(params.patch)    ## N pixels / patch
    c = max_size_in_MB*1_000_000 ## N Bytes
    d = 2 ## N float values / patch-pixel (raw + target)
    e = 2 ## N Bytes / float value
    f = int(c / (b*d*e)) ## Total number of patches
    g = f/a ## desired patches per image
    h = np.prod(garlist[0].raw.shape) ## N image pixels
    i = h / b ## actual number of patches per image
    # print('g',g,'i',i,)
    # print('a',a,'b',b,'c',c,'d',d,'e',e,'f',f,)
    return f, a

  n_patches, n_images = _f()
  ## goal is divide n_patches roughly evenly among all n_images
  n_patches_per_img = [x.stop-x.start for x in divide_evenly_with_min1(n_patches, n_images)]
  # print(np.unique(n_patches_per_img, return_counts=True))
  # print(n_patches,n_images,)

  _f_gar2samp = gar2samples_sampleFirst if params.sparse else gar2samples
  fullsamples = [_f_gar2samp(gar,nsamples,params) for (gar,nsamples) in zip(garlist,n_patches_per_img)]
  fullsamples = np.array([x for stime in fullsamples for x in stime])

  if params.sparse: fullsamples = norm_samples_raw(fullsamples)

  print("Done creating samples. Now save to disk.")

  save(samples2pngSN(take_n_evenly_spaced_items(fullsamples,N_png_samples)), P.savedir_local / 'sample_pngs')
  res = SimpleNamespace(samples=fullsamples, params=params)
  save(res, P.savedir_local / 'fullsamples')

  _n1 = len(res.samples)
  _n2 = res.samples[0].raw.shape
  _n3 = (2+1+2) * _n1 * np.prod(_n2) / 1_000_000
  print(f"""
  Working on {info.isbiname} / {info.dataset}.
  In total we saved {_n1} samples, each with shape {_n2}.
  This makes a total of {_n3:0.3f} MB of data.
  """)
  return res



## WIP. Create Dataframe of Virtual Samples, then fill them in.

def build_trainingdata2(pid=0):
  P = pid2params(pid)
  info = P.info
  
  print(f"""
    Begin pid {pid} ... {info.isbiname} / {info.dataset}.
    Savedir is {P.savedir_local}
    """)

  params  = init_params(info.ndim)
  _params = cpnet_data_specialization(info)
  for k in _params.__dict__.keys(): params.__dict__[k] = _params.__dict__[k]  

  image_names,ltps_name = isbiInfo_to_filenames(info)
  raw_names,lab_names = zip(*image_names)
  ltps  = load(ltps_name)
  times = ltps.keys() if type(ltps) is dict else np.arange(len(ltps))
  ltps  = [ltps[k] for k in times]
  imgFrame = DataFrame(dict(rawname=raw_names,labname=lab_names,pts=ltps,time=times))
  imgFrame['times2'] = imgFrame.rawname.apply(lambda x: int(re.search(r'(\d+)\.(tif|zar)', x).group(1)))
  N = len(image_names)
  imgFrame['shape']  = [info.shape] * N

  ## subsample images to manage data size
  ## PARAM
  # imgFrame = imgFrame.iloc[::15]

  df = pandas.concat([virtualPatches(x, params) for x in imgFrame.iloc])
  df['shape'] = df['ss_main'].apply(lambda ss: tuple(s.stop-s.start for s in ss))
  df['npts']  = df['pts'].apply(len)
  df['ss_startPt'] = df['ss_main'].apply(lambda ss: tuple(s.start for s in ss))

  df['ptsRel'] = df['pts'] - df['ss_startPt']


  print(f"""
  Working on {info.isbiname} / {info.dataset}.
  imgsize {imgFrame['shape'][0]}
  
  {df['shape'].describe()}
  """)

  N = len(df)
  print(f"Full Size {N} -- {strDiskSizePatchFrame(df)}")

  df = df[df.npts>0]
  N = len(df)
  print(f"Obj Only Size {N} -- {strDiskSizePatchFrame(df)}")

  idx = np.linspace(0,len(df)-1, 100)
  df  = df.iloc[idx]
  N = len(df)
  print(f"Subsampled Size {N} -- {strDiskSizePatchFrame(df)}")

  df = df.sample(n=10)

  # 
  #  Now below we begin actually loading the data. This is the time consuming part.
  # 

  tic = time()
  params.imgNorm = not params.sparse
  df = addRawLabTarget(df,imgFrame,params)

  df['raw']    = df['raw'].apply(   lambda x: zoom(x,params.zoom,order=1).astype(np.float16))
  df['target'] = df['target'].apply(lambda x: zoom(x,params.zoom,order=1).astype(np.float16))
  df['lab']    = df['lab'].apply(   lambda x: zoom(x,params.zoom,order=0).astype(np.uint8))

  print("Done creating samples. Now save to disk.")

  print("TIME: ", time()-tic)

  df.to_pickle(P.savedir_local / 'patchFrame.pkl')
  imgFrame.to_pickle(P.savedir_local / 'imgFrame.pkl')
  idxs = np.linspace(0,len(df)-1,10).astype(int) ## evenly sample items
  save(samples2pngSN(df.iloc[idxs].iloc), P.savedir_local / 'sample_pngs')

def addRawLabTarget(patchFrame,imgFrame,params):
  """
  Load images from disk and crop out patch data.
  """

  def f(df):
    time = df.name
    imfr = imgFrame.loc[time]

    raw = load(imfr['rawname'])
    if params.imgNorm: raw = norm_percentile01(raw,2,99.4)
    df['raw'] = [raw[ss].copy() for ss in df.ss_main]
    if not params.imgNorm:
      norm = lambda x: norm_percentile01(x,2,99.4)
      df['raw'] = df['raw'].apply(norm)

    lab = load(imfr['labname'])
    df['lab'] = [lab[ss].copy() for ss in df.ss_main]

    if np.prod(raw.shape) < 2 * df['shape'].apply(np.prod).sum():
      target = dgen.place_gaussian_at_pts(imfr['pts'], lab.shape, kern)
      df['target'] = [target[ss].copy() for ss in df.ss_main]
    else:
      df['target'] = [dgen.place_gaussian_at_pts(x.ptsRel,x.raw.shape,params.kern) for x in df.iloc]

    return df

  patchFrame = patchFrame.groupby('time').apply(f)
  return patchFrame

def get_slices2(pts, imshape, params):

  patchsize = (np.array(params.patch) // params.zoom).astype(int)
  slices  = dgen.tileND_random(imshape, (patchsize*1.1).astype(int), patchsize) #, (4,10,10)[-raw.ndim:])
  # ipdb.set_trace()

  return slices , 'inner'

def virtualPatches(imgFrameRow, params):
  ifr = imgFrameRow
  slices, ss_main = get_slices2(ifr.pts,ifr['shape'],params)
    
  df = DataFrame(
    dict(
      outer=[x.outer for x in slices],
      inner=[x.inner for x in slices],
      inner_rel=[x.inner_rel for x in slices],
      )
    )
  df['ss_main'] = df[ss_main] ## varies depending on get_slices()
  df['time'] = ifr.time

  res = dgen.find_points_within_patches3(ifr.pts, df['ss_main'])
  df['pts'] = [ifr.pts[np.argwhere(res[:,i])[:,0]] for i in range(len(slices))]
  
  return df


# def get_slices1(pts, imshape, params):
#   ## subsample dataset. no overlapping patches.
#   patchsize = (np.array(params.patch) // params.zoom).astype(int)  
#   DD = int(8 // params.zoom[-1])
#   _ftile  = lambda end,length,border : dgen.tile1d_predict(end,length,border,divisible=DD) #tile1d_predict(end,length,border,divisible=8)
#   borders = (2,10,10)[-len(imshape):]
#   slices  = dgen.tile_multidim(imshape, patchsize, borders, f_singledim=_ftile) #, (4,10,10)[-raw.ndim:])
#   # slices  = [SimpleNamespace(outer=x.outer,inner=x.inner,inner_rel=x.inner_rel) for x in slices]
#   return slices , 'outer'


# def addRawLab(patchFrame,imgFrame,kern=None):
#   patchFrame['raw'] = None
#   patchFrame['lab'] = None
#   patchFrame['target'] = None
#   for row in imgFrame.iloc:
#     m = patchFrame['time']==row['time']
#     if not np.any(m): continue
#     print(row['time'])

#     ipdb.set_trace()

#     raw = load(row['rawname'])
#     patchFrame.loc[m,'raw'] = np.array([raw[ss] for ss in patchFrame.loc[m,'ss_main']], dtype=np.object)
    
#     lab = load(row['labname'])
#     patchFrame.loc[m,'lab'] = np.array([lab[ss] for ss in patchFrame.loc[m,'ss_main']], dtype=np.object)
    
#     if kern:
#       target = dgen.place_gaussian_at_pts(row['pts'],row['shape'],kern)
#       patchFrame.loc[m,'target'] = np.array([target[ss] for ss in patchFrame.loc[m,'ss_main']], dtype=np.object)


## Experimental. Highly Unstable.

def get_dataset_balance():
  for pid in range(19*2):
    _params = pid2params(pid)
    I = _params.info

    fullsamples = load(savedir / f"e24_isbidet_AOT/v01/pid{pid:03d}/fullsamples2")
    counts = Counter([s.pts.shape[0] for s in fullsamples.samples])

    print(f"""
    pid{pid:03d} on dataset {I.isbiname} / {I.dataset}
    sample object Counts: 0:{counts[0]} nonzero:{sum(counts)-counts[0]}
    """)

def get_all_dataset_sizes():
  def f(pid):
    _params = pid2params(pid)
    myname = _params.info.myname
    isbiname = _params.info.isbiname
    dataset = _params.info.dataset
    ltps = load(f"/projects/project-broaddus/rawdata/{myname}/traj/{isbiname}/{dataset}_traj.pkl")
    if type(ltps) is dict: ltps = list(ltps.values())
    npts = [len(x) for x in ltps]

    n_imgs = (_params.info.stop - _params.info.start)
    n_pix_per_img = np.prod(_params.info.shape.shape)
    n_pixels = n_imgs * n_pix_per_img
    n_total_objects = sum(npts)
    n_pix_per_obj = n_pixels / n_total_objects

    s = dict(
             n_pix_per_obj=n_pix_per_obj,
             isbiname=isbiname,
             dataset=dataset,
             n_imgs=n_imgs,
             n_pix_per_img=n_pix_per_img,
             n_total_objects=n_total_objects,
             n_pixels=n_pixels,
             myname=myname,
             )

    return s
    # return dict({k:locals()[k] for k in set(locals()) - {'_params','npts'}})
    # return dict(**locals())
    # return dict(n_pixels=n_pixels, n_total_objects=n_total_objects, n_pix_per_obj=n_pix_per_obj)
    # return SimpleNamespace(n_imgs=n_imgs,n_pix_per_img=n_pix_per_img,n_pixels=n_pixels,myname=myname,isbiname=isbiname,dataset=dataset,npts=npts,)
  res = pandas.DataFrame([f(i) for i in range(19*2)])
  # sizes = np.array([x.n_pix_per_img for x in res])
  # print(sorted(sizes) / sizes.min())
  return res




def get_all_fullsamples_sizes():
  def f(pid):
    _params = pid2params(pid)
    myname = _params.info.myname
    isbiname = _params.info.isbiname
    dataset = _params.info.dataset
    ltps = load(f"/projects/project-broaddus/rawdata/{myname}/traj/{isbiname}/{dataset}_traj.pkl")
    if type(ltps) is dict: ltps = list(ltps.values())
    npts = [len(x) for x in ltps]


    fname = _params.savedir_local / "fullsamples2"
    fullsamples = load(fname)
    sizes  = Counter([x.raw.shape for x in fullsamples.samples])
    counts = Counter([s.pts.shape[0] for s in fullsamples.samples])

    sortedcounts = sorted(counts.keys())[:10]
    r1 = ("{:>6}"*len(sortedcounts)).format(*[str(k) for k in sortedcounts])
    r2 = ("{:>6}"*len(sortedcounts)).format(*[str(counts[k]) for k in sortedcounts])
    # ipdb.set_trace()

    print(f"""
    Pid {pid} Isbiname {isbiname} / {dataset}
    contains {len(fullsamples.samples)} samples with sizes {sizes}
    --- Object Counts per Patch ---
    N objects: {r1} ... 
    N patches: {r2} ... 
    total size: {bytes2string(file_size(fname))}
    """)

    # return s
    # return dict({k:locals()[k] for k in set(locals()) - {'_params','npts'}})
    # return dict(**locals())
    # return dict(n_pixels=n_pixels, n_total_objects=n_total_objects, n_pix_per_obj=n_pix_per_obj)
    # return SimpleNamespace(n_imgs=n_imgs,n_pix_per_img=n_pix_per_img,n_pixels=n_pixels,myname=myname,isbiname=isbiname,dataset=dataset,npts=npts,)
  for i in range(19*2): f(i)
  # for i in [18]: f(i)
  # res = pandas.DataFrame([f(i) for i in range(19*2)])
  # sizes = np.array([x.n_pix_per_img for x in res])
  # print(sorted(sizes) / sizes.min())
  # return res  





