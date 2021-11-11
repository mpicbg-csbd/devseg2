"""
Generate CP-Net training data for all ISBI datasets, training on both 01/02 acquisitions.
Build pandas dataframes for samples ahead of time (before training).
Only augmentation is JIT.
"""

import os
from pathlib import Path

import pandas
from collections import Counter

from experiments_common import rchoose, partition, shuffle_and_split, parse_pid, iterdims, savedir_global
import datagen as dgen

# from segtools.render import rgb_max
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


import ipdb
from time import time

from segtools import torch_models
import inspect

from e26_utils import *


import joblib
from joblib import Memory


# savedir = savedir_global()
# print("savedir:", savedir)




## stable. job-specific utility funcs.

# def myrun_slurm(pids):
#   ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
#   ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

#   shutil.copy(__file__, Path("/projects/project-broaddus/devseg_2/src/temp/") / str(Path(__file__).stem + "_copy.py"))
#   _gpu  = "-p gpu --gres gpu:1 -n 1 -c 1 -t 6:00:00 --mem 128000 "
#   _cpu  = "-n 1 -t 1:00:00 -c 1 --mem 128000 "
#   slurm = 'sbatch -J e26_{pid:03d} {_resources} -o slurm/e24_pid{pid:03d}.out -e slurm/e24_pid{pid:03d}.err --wrap \'python3 -c \"import temp.e24_isbidet_AOT_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
#   slurm = slurm.replace("{_resources}",_cpu)
#   for pid in pids: Popen(slurm.format(pid=pid),shell=True)

# def myrun_slurm_entry(pid=0):
#   build_trainingdata(pid)

#   # (p0,p1),pid = parse_pid(pid,[2,19,9])
#   # myrun([p0,p1,p2])
#   # for p2,p3 in iterdims([2,5]):
#   #   try:
#   #   except:
#   #     print("FAIL on pids", [p0,p1,p2,p3])


def pid2params(pid):
  (p0,p1),pid = parse_pid(pid,[19,2])
  # savedir_local = savedir / f'e24_isbidet_AOT/v01/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  return SimpleNamespace(**locals())

def isbiInfo_to_filenames(info):
  n_raw  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
  n_lab  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track
  # if info.index in [6,11,12,13,14,15,18]:

  ## MYPARAM undo this
  # n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  # n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"

  filenames_raw = [(n_raw.format(time=i),n_lab.format(time=i)) for i in range(info.start,info.stop)]
  filename_ltps = f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl"
  return filenames_raw, filename_ltps

def samples2pngSN(samples):
  s = SimpleNamespace()
  for sam in samples:
    s.__dict__[f't{sam.time}_r'] = sample2RawPng(sam)
    # s.__dict__[f't{sam.time}_r'] = img2png(sam.raw.copy())
    # s.__dict__[f't{sam.time}_l'] = img2png(sam.lab.copy())
    s.__dict__[f't{sam.time}_t'] = img2png(sam.target.copy())
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

## Advanced printing

def describe_virtual_samples(df):
  sizes  = Counter(df['shape'])
  counts = Counter(df['npts'])

  sortedcounts = sorted(counts.keys())
  ellipsis = "..." if len(sortedcounts) >= 10 else ""
  sortedcounts = sortedcounts[:10]

  r1 = ("{:>6}"*len(sortedcounts)).format(*[str(k) for k in sortedcounts])
  r2 = ("{:>6}"*len(sortedcounts)).format(*[str(counts[k]) for k in sortedcounts])

  print(f"""
  Contains {len(df)} samples with {df.npts.sum()} total objects from {len(df['time'].unique())} timepoints
  Patch Sizes --  {sizes}
  --- Object Counts per Patch ---
  N objects: {r1} {ellipsis} 
  N patches: {r2} {ellipsis} 
  Total size: {strDiskSizePatchFrame(df)}
  """)

def describe_samples(samples):
  sizes  = Counter([x.raw.shape for x in samples])
  counts = Counter([s.pts.shape[0] for s in samples])

  sortedcounts = sorted(counts.keys())[:10]
  r1 = ("{:>6}"*len(sortedcounts)).format(*[str(k) for k in sortedcounts])
  r2 = ("{:>6}"*len(sortedcounts)).format(*[str(counts[k]) for k in sortedcounts])

  print(f"""
  contains {len(samples)} samples with sizes {sizes}
  --- Object Counts per Patch ---
  N objects: {r1} ... 
  N patches: {r2} ... 
  total size: {bytes2string(file_size(fname))}
  """)



## Unstable. Dataset Parameters.

def init_params(ndim):
  P = SimpleNamespace()
  if ndim==2:
    P.zoom  = (1,1) #(0.5,0.5)
    P.kern  = [5,5]
    P.patch = (256,256)
    P.border = [2,2]
    P.match_dub = 10
    P.match_scale = [1,1]

  elif ndim==3:
    P.zoom   = (1,1,1)
    P.kern   = [3,5,5]
    P.patch  = (16,128,128)
    P.border = [1,2,2]
    P.match_dub = 10
    P.match_scale = [4,1,1]

  P.nms_footprint = P.kern
  P.patch = np.array(P.patch)

  return P

def cpnet_ISBIdata_specialization(params,info):
  isbiname = info.isbiname
  dataset = info.dataset
  p = params

  p.traintime = 200
  p.scale = info.scale
  p.sparse = True if info.isbiname in ["Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF",] else False

  if isbiname in ["Fluo-N3DH-CE" , "Fluo-N3DH-SIM+" , "PhC-C2DH-U373" , "Fluo-N2DH-GOWT1"]:
    p.zoom  = {3:(1,0.5,0.5), 2:(0.5,0.5)}[info.ndim]
  if isbiname == "Fluo-N3DH-CHO":
    p.zoom   = (1,0.25,0.25)
    p.border = (0,0,0)
  if isbiname=="Fluo-N3DL-TRIF":
    p.kern  = [3,3,3]
    # p.zoom  = (0.5,0.5,0.5) ## MYPARAM undo this
    p.patch = (64,64,64)
    p.match_scale = [1,1,1]
  if isbiname=="Fluo-C3DH-H157":
    p.zoom = (1/4,1/8,1/8)
    p.kern = (1,3,3)
    p.match_dub = 30 ## in full size space
  if isbiname in ["Fluo-C3DH-A549", "Fluo-C3DH-A549-SIM"]:
    # p.zoom = (1,1/4,1/4)
    p.zoom = (1,1/2,1/2)
  if isbiname=="Fluo-C2DL-MSC":
    # a,b = info.shape
    # p.zoom = {'01':(1/4,1/4), '02':(128/a, 200/b)}[info.dataset]
    p.zoom = 1/4, 1/4
    ## '02' rescaling is almost exactly isotropic while still being divisible by 8.
  if isbiname=="DIC-C2DH-HeLa":
    # p.kern = [7,7]
    p.zoom = (0.5,0.5)
  if isbiname=="Fluo-N3DL-DRO":
    p.kern = [1.5,3,3]
  if isbiname == 'PhC-C2DL-PSC':
    p.kern = [3,3]



  ## ignore errors when one object is within evalBorder of XY image boundary
  if isbiname in ["DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-C3DH-H157", "Fluo-N2DH-GOWT1", "Fluo-N3DH-CE", "Fluo-N3DH-CHO", "PhC-C2DH-U373",]:
    p.evalBorder = (0,50,50) if info.ndim==3 else (50,50)
  elif isbiname in ["BF-C2DL-HSC", "BF-C2DL-MuSC", "Fluo-C3DL-MDA231", "Fluo-N2DL-HeLa", "PhC-C2DL-PSC",]:
    p.evalBorder = (0,25,25) if info.ndim==3 else (25,25)
  elif isbiname in ["Fluo-C3DH-A549", "Fluo-N2DH-SIM+", "Fluo-N3DH-SIM+", "Fluo-C3DH-A549-SIM", "Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF"]:
    p.evalBorder = (0,0,0) if info.ndim==3   else (0,0)
  
  ## Sparsely Annotated datasets get: loss masking, patch-wise normalization and target creation (for speed), 

  p.subsample_traintimes = slice(0,None,None)

  if isbiname in ['BF-C2DL-HSC' , 'BF-C2DL-MuSC']:
    p.subsample_traintimes = slice(0,None,30)

  # if isbiname == 'Fluo-N3DH-CE': p.subsample_traintimes = slice(0,None,2) ## 22,23
  # if isbiname == 'Fluo-N3DH-SIM+': p.subsample_traintimes = slice(0,None,3)
  # if isbiname == 'Fluo-N3DL-DRO' and dataset == '02': p.subsample_traintimes = slice(0,None,3)
  # if isbiname == 'Fluo-N3DL-TRIC' and dataset == '01' : p.subsample_traintimes = slice(0,None,2) ## Fluo-N3DL-TRIC
  # if isbiname == 'Fluo-N3DL-TRIC' and dataset == '02' : p.subsample_traintimes = slice(0,None,3) ## Fluo-N3DL-TRIC
  # if isbiname == 'PhC-C2DH-U373': 
  #   p.subsample_traintimes = slice(0,None,4)
  # if isbiname == 'Fluo-N3DL-TRIF':
  #   p.subsample_traintimes = slice(0,None,4)

## Stable. Create Dataframe of Virtual Images. 

def build_imgFrame_and_params(pid=0):
  dPID = pid2params(pid)
  info = dPID.info
  
  print(f"""
    Begin `build_imgFrame_and_params()` on pid {pid} ... {info.isbiname} / {info.dataset}.
    """)
  # Savedir is IRRELEVANT ~~{dPID.savedir_local}~~

  imgFrame = pid2ImgFrame(info)
  params  = init_params(info.ndim)
  cpnet_ISBIdata_specialization(params,info,)

  ## train = 0; test = 1
  labels = np.ones(len(imgFrame),dtype=np.uint8)
  labels[params.subsample_traintimes] = 0
  imgFrame['labels'] = labels
  # ipdb.set_trace()
  
  imgFrame = addRescalingInfo2ImgFrame(imgFrame,params.zoom)

  # imgFrame.to_pickle(dPID.savedir_local / 'imgFrame.pkl')
  # save(params, dPID.savedir_local / 'params.pkl')
  return imgFrame, params

#### Used by build_imgFrame_and_params

def pid2ImgFrame(info):
  image_names,ltps_name = isbiInfo_to_filenames(info)
  raw_names,lab_names = zip(*image_names)
  
  ltps  = load(ltps_name)
  times = ltps.keys() if type(ltps) is dict else np.arange(len(ltps))
  ltps  = [ltps[k] for k in times]
  imgFrame = DataFrame(dict(rawname=raw_names,labname=lab_names,pts=ltps,time=times))
  imgFrame['times2'] = imgFrame.rawname.apply(lambda x: int(re.search(r'(\d+)\.(tif|zar)', x).group(1)))
  N = len(image_names)
  imgFrame['shape']  = [info.shape] * N
  imgFrame['npts']   = imgFrame['pts'].apply(len)
  return imgFrame

def addRescalingInfo2ImgFrame(imgFrame,_zoom):
  
  ## WARNING: this must exactly replicate the way that zoom() acts on the image shape.
  imgFrame['shape_rescaled'] = imgFrame['shape'].apply(lambda x: tuple((x*np.array(_zoom)).astype(np.uint32)))
  imgFrame['zoom_effective'] = imgFrame['shape_rescaled'] / imgFrame['shape'].apply(np.array)  
  imgFrame['pts_rescaled']   = imgFrame.apply(lambda x: zoom_pts(x.pts, x.zoom_effective), axis=1) ## WARNING: consistent img + pts rescaling is nontrivial!

  return imgFrame



## Stable. Create DataFrame of Virtual Patches, and reify them.

# @memory.cache
def build_patchFrame(pid=0):
  dPID = pid2params(pid)
  info = dPID.info

  imgFrame, params = build_imgFrame_and_params(dPID.pid)
  imgFrame = imgFrame[imgFrame.labels==0]

  def build_patches(time,pts,sz_img):
    df = virtualPatches(pts, sz_img, params.patch)
    df['time'] = time
    return df

  df = pandas.concat([build_patches(time,pts,sz_img) for time,pts,sz_img in imgFrame[['time','pts_rescaled','shape_rescaled']].iloc])
  df['shape'] = df['ss_main'].apply(lambda ss: tuple(s.stop-s.start for s in ss))
  df['npts']  = df['pts'].apply(len)
  df['ss_startPt'] = df['ss_main'].apply(lambda ss: tuple(s.start for s in ss))
  df['ptsRel'] = df['pts'] - df['ss_startPt']

  if params.sparse: df = df[df.npts>0].reset_index()

  desc = f"""
  times considered -- {repr(list(imgFrame.times2))[:50]}
  orig img shape   -- {info.shape}
  zoom factor      -- {params.zoom}
  new  img shape   -- {Counter(imgFrame['shape_rescaled'])}
  orig patch shape -- {params.patch}
  actual patch sha -- {Counter(df['shape'])}
  pixel fraction   -- {df['shape'].apply(np.prod).sum() / (imgFrame['shape_rescaled'].apply(np.prod).sum())}
  object fraction  -- {df['npts'].sum()} / {imgFrame['npts'].sum()} ({df['npts'].sum() / imgFrame['npts'].sum()})
  """
  print(desc)
  describe_virtual_samples(df)

  df.desc = desc

  # ipdb.set_trace()
  # return
  # df = df.sample(n=4)

  tic = time()
  f = lambda timegroup : processOneTimepoint(timegroup.name, timegroup, imgFrame, params)
  df = df.groupby('time').apply(f)
  print("Total patch creation time: ", time()-tic)

  # df.to_pickle(dPID.savedir_local / 'patchFrame.pkl')
  idxs = np.linspace(0,len(df)-1,min(len(df),10)).astype(int) ## evenly sample items
  pngs = samples2pngSN(df.iloc[idxs].iloc)
  # save(pngs, dPID.savedir_local / 'sample_pngs')
  return df, params, pngs

#### Used by build_patchFrame

def processOneTimepoint(time,patchFrame,imgFrame,params):
  """
  Load images from disk and crop out patch data.
  """
  print(f"Processing {time} ...",end="\r")
  df = patchFrame
  # ipdb.set_trace()
  imfr = imgFrame.set_index('time').loc[time]

  raw = load(imfr['rawname'])
  lab = load(imfr['labname'])

  if not params.sparse:
    raw = myzoom(raw,params.zoom)
    lab = myzoom(lab,params.zoom) ## some thin annotations may disappear!
    raw = norm_percentile01(raw,2,99.4)
    target = dgen.place_gaussian_at_pts(imfr['pts_rescaled'], raw.shape, params.kern)
    df['raw']    = [raw[ss].copy().astype(np.float16) for ss in df.ss_main]
    df['lab']    = [lab[ss].copy().astype(np.uint8) for ss in df.ss_main]
    df['target'] = [target[ss].copy().astype(np.float16) for ss in df.ss_main]
  else:
    # border = (6,6,6)
    ss_rescaled = [tuple(slice(floor(s.start/z),floor(s.stop/z)) for s,z in zip(ss,params.zoom)) for ss in df.ss_main]
    _raw = [raw[ss].copy() for ss in ss_rescaled]
    p0,p1 = np.percentile(np.array([r.flatten() for r in _raw]), [2,99.4]) ## combine all pixels for stats
    norm = lambda x: norm_affine01(x,p0,p1)
    df['raw'] = [norm(r) for r in _raw]
    df['lab'] = [lab[ss].copy() for ss in ss_rescaled]
    df['raw']    = df['raw'].apply(   lambda x: myzoom(x,params.zoom).astype(np.float16))
    df['lab']    = df['lab'].apply(   lambda x: myzoom(x,params.zoom).astype(np.uint8))
    # ipdb.set_trace()
    df['target'] = [dgen.place_gaussian_at_pts(x.ptsRel,x.raw.shape,params.kern).astype(np.float16) for x in df.iloc]
    # df['target'] = df['target'].apply(lambda x: myzoom(x,params.zoom).astype(np.float16))

  return df

def conform_szPatch(sz_img,sz_patch,divisible=8):
  """
  Ensure that patch size and box size are smaller than image and that patch_size is divisible by 8
  """
  sz_patch = np.minimum(sz_patch, sz_img).astype(int)
  sz_patch = (np.floor(sz_patch/divisible)*divisible).astype(int)
  sz_box   = np.minimum(sz_patch*1.2, sz_img).astype(int)
  return sz_patch,sz_box

def get_slices2(pts, sz_img, sz_patch):

  sz_patch_resolved,sz_box = conform_szPatch(sz_img, sz_patch, divisible=(1,8,8)[-len(sz_img):])
  slices  = dgen.tileND_random(sz_img, sz_box, sz_patch_resolved) #, (4,10,10)[-raw.ndim:])

  return slices , 'inner'

def get_slices1(pts, imshape, params):
  ## subsample dataset. no overlapping patches.
  patchsize = (np.array(params.patch) // params.zoom).astype(int)  
  DD = int(8 // params.zoom[-1])
  _ftile  = lambda end,length,border : dgen.tile1d_predict(end,length,border,divisible=DD) #tile1d_predict(end,length,border,divisible=8)
  borders = (2,10,10)[-len(imshape):]
  slices  = dgen.tile_multidim(imshape, patchsize, borders, f_singledim=_ftile) #, (4,10,10)[-raw.ndim:])
  # slices  = [SimpleNamespace(outer=x.outer,inner=x.inner,inner_rel=x.inner_rel) for x in slices]
  return slices , 'outer'

def virtualPatches(pts, sz_img, sz_patch):
  slices, ss_main = get_slices2(pts, sz_img, sz_patch)

  df = DataFrame([x.__dict__ for x in slices])
  df['ss_main'] = df[ss_main] ## varies depending on get_slices()

  res = dgen.find_points_within_patches3(pts, df['ss_main'])
  df['pts'] = [pts[np.argwhere(res[:,i])[:,0]] for i in range(len(slices))]

  # if pts.shape != df['pts'].iloc[0].shape: ipdb.set_trace()

  return df





