"""
Training detection for the ISBI datasets in a data-generator / sampler agnostic way.
Derived from e24_trainer.py

Works with "ahead of time" sampler.
"""


import matplotlib
import pandas

from time import time
_start_time = time()

from itertools import islice
from subprocess import run, Popen
import json
from types import SimpleNamespace
from glob import glob
from math import floor,ceil
import re
from pathlib import Path
import ipdb

import shutil
from skimage.feature  import peak_local_max
from scipy.ndimage import zoom
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

try:
    import gputools
except Exception as e:
    print("GPUTOOLS ERROR \n", e)

from experiments_common import rchoose, partition, shuffle_and_split, parse_pid, iterdims, savedir_global
import datagen
from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
from segtools.ns2dir import load, save
import augmend
from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale,IntensityScaleShift,Identity
from segtools import torch_models
from tqdm import tqdm
from segtools.point_matcher import match_unambiguous_nearestNeib
from segtools.point_tools import trim_images_from_pts2

from segtools.math_utils import conv_at_pts4, conv_at_pts_multikern

from collections import Counter

import augmend
from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale,IntensityScaleShift,Identity

import datagen as dgen

import e26_isbidet_dgen
memory = e26_isbidet_dgen.memory

from e26_utils import *

import tracking
from segtools import scores_dense

savedir = savedir_global()
print("savedir:", savedir)

print(f"Total Import Time: {time()-_start_time}")

import matplotlib.pyplot as plt

from segtools.math_utils import conv_at_pts_multikern
from expand_labels_scikit import expand_labels



## Stable. Utils.


def wipedir(path):
  path = Path(path)
  if path.exists(): shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)

def memoize(closure,filename,force=False):
  """
  Memoization at point of call (less useful)
  """
  filename = Path(filename)
  if filename.exists() and not force:
    return load(filename)
  else:
    res = closure()
    save(res,filename)
    return res

@DeprecationWarning
def mycache(file_name):
  # varname = ".cached_" + file_name + ".pkl"    
  def decorator(original_func):

      # try:
      #     # cache = json.load(open(file_name, 'r'))
      #     cache = load(file_name)
      #     cache_vars = load(varname)
      # except (IOError, ValueError):
      #     cache = {}

      def new_func(param):
        file_name = file_name.format(**locals())
        if param not in cache:
            cache[param] = original_func(param)
            save
            json.dump(cache, open(file_name, 'w'))
        return cache[param]

      return new_func

  return decorator



# Stable training functions.

def _init_unet_params(ndim):
  T = SimpleNamespace()
  if ndim==2:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
  elif ndim==3:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
  torch_models.init_weights(T.net)
  return T

def build_augmend(ndim):

  """
  Three args to aug(raw,target,weights).
  """

  aug = Augmend()
  ax = {2:(0,1), 3:(1,2)}[ndim]
  if ndim==3:
    aug.add([FlipRot90(axis=0), FlipRot90(axis=0), FlipRot90(axis=0),], probability=1)
    aug.add([FlipRot90(axis=(1,2)), FlipRot90(axis=(1,2)), FlipRot90(axis=(1,2))], probability=1)
  else:
    aug.add([FlipRot90(axis=(0,1)), FlipRot90(axis=(0,1)), FlipRot90(axis=(0,1)),], probability=1)

  aug.add([augmend.IntensityScaleShift(), augmend.Identity(), augmend.Identity()], probability=1)

  # aug.add([Rotate(axis=ax, order=1),
  #          Rotate(axis=ax, order=1),],
  #         probability=0.5)
  return aug


## Unstable. Main Training function.


"""
Train a new model with either 01,02,or both datasets.
The specific images used are determined by `params.subsample_traintimes`
"""
def train(pid,continue_training=False):

  (p1,p0,),pid = parse_pid(pid,[3,19])
  ## p1 : 0 = both, 1 = 01, 2 = 02

  savedir_local = savedir / f'e26_isbidet/train/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05

  if p1==0:
    info01 = get_isbi_info(myname,isbiname,"01")
    info02 = get_isbi_info(myname,isbiname,"02")
    ds1,params1,_pngs = e26_isbidet_dgen.build_patchFrame([p0,0])
    ds2,params2,_pngs = e26_isbidet_dgen.build_patchFrame([p0,1])
    df = pandas.concat([ds1,ds2])
    info = info01
    params = params1
  if p1==1:
    info = get_isbi_info(myname,isbiname,"01")
    df,params,_pngs = e26_isbidet_dgen.build_patchFrame([p0,0])
  if p1==2:
    info = get_isbi_info(myname,isbiname,"02")
    df,params,_pngs = e26_isbidet_dgen.build_patchFrame([p0,1])


  print(f"""
    Begin pid {pid} ... {info.isbiname}
    p0 = {p0} 
    =>  p1 : 0 = both, 1 = 01, 2 = 02
    Savedir is {savedir_local}
    """)

  save(df,savedir_local / 'patchFrame.pkl')
  save(params, savedir_local / 'params.pkl')

  # df = load(savedir_local / "patchFrame.pkl")
  # params = load(savedir_local / "params.pkl")
  e26_isbidet_dgen.describe_virtual_samples(df)

  P = params

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = _init_unet_params(info.ndim).net
  net = net.to(device)

  ## MYPARAM continue training existing dataset
  CONTINUE = continue_training
  print("CONTINUE ? : ", bool(CONTINUE))
  
  if CONTINUE:
    labels = load(savedir_local / "labels.pkl")
    net.load_state_dict(torch.load(savedir_local / f'm/best_weights_latest.pt')) ## MYPARAM start off from best_weights ?
    history = load(savedir_local / 'history.pkl')
  else:
    # if p1!=0:
    if True:
      N = len(df)
      a,b = N*5//8,N*7//8  ## MYPARAM train / vali / test fractions
      labels = np.zeros(N,dtype=np.uint8)
      labels[a:b]=1; labels[b:]=2 ## 0=train 1=vali 2=test
      np.random.shuffle(labels)
      save(labels, savedir_local / "labels.pkl")
    else:
      _,pid1 = parse_pid([1,p0],[3,19])
      _,pid2 = parse_pid([2,p0],[3,19])
      labels1 = load(savedir / f'e26_isbidet/train/pid{pid1:03d}/labels.pkl')
      labels2 = load(savedir / f'e26_isbidet/train/pid{pid2:03d}/labels.pkl')
      labels = np.concatenate([labels1,labels2]) ## first 01 then 02
      save(labels, savedir_local / "labels.pkl")
      # ipdb.set_trace()

    history = SimpleNamespace(lossmeans=[],valimeans=[],)
    wipedir(savedir_local/'m')
    wipedir(savedir_local/"glance_output_train/")
    wipedir(savedir_local/"glance_output_vali/")

  ## post-load configuration
  assert len(labels)>8
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)
  df['labels'] = labels

  f_aug = build_augmend(df.raw.iloc[0].ndim)

  def addweights(s):
    if P.sparse:
      w0 = dgen.weights__decaying_bg_multiplier(s.target,0,thresh=np.exp(-0.5*(3)**2),decayTime=None,bg_weight_multiplier=0.0)
    else:
      w0 = np.ones(s.target.shape,dtype=np.float32)
    return w0
  df['weights'] = df.apply(addweights,axis=1)


  def mse_loss(x,yt,w):

    x  = torch.from_numpy(x.copy() ).float().to(device, non_blocking=True)
    yt = torch.from_numpy(yt.copy()).float().to(device, non_blocking=True)
    w  = torch.from_numpy(w.copy() ).float().to(device, non_blocking=True)

    y  = net(x[None,None])[0,0]

    ## Introduce ss.b masking to ensure that backproped pixels do not overlap between train/vali/test
    
    loss = torch.abs((w*(y-yt)**2)).mean()
    return y,loss

  # trainset = df[(df.labels==0) & (df.npts>0)] ## MYPARAM subsample trainset ?
  trainset = df[df.labels==0]
  N_train  = len(trainset)

  def backprop_n_samples_into_net():
    _losses = []
    idxs = np.arange(N_train)
    np.random.shuffle(idxs)
    tic = time()
    for i in range(N_train):
      s  = trainset.iloc[idxs[i]]
      x  = s.raw.copy()
      yt = s.target.copy()
      w  = s.weights.copy()
      x,yt,w = f_aug([x,yt,w])
      y,l = mse_loss(x,yt,w)
      l.backward()
      opt.step()
      opt.zero_grad()
      _losses.append(float(l.detach().cpu()))
      dt = time()-tic; tic = time()
      print(f"it {i}/{N_train}, dt {dt:5f}, max {float(y.max()):5f}", end='\r',flush=True)

    history.lossmeans.append(np.nanmean(_losses))

  def validate_single(sample):
    s = sample

    x  = s.raw.copy()
    yt = s.target.copy()
    w  = s.weights.copy()

    with torch.no_grad(): y,l = mse_loss(x,yt,w)

    y = y.cpu().numpy()
    l = float(l.cpu().numpy())
    
    _peaks = y #.copy() #y/y.max()
    pts      = peak_local_max(_peaks,threshold_abs=.5,exclude_border=False,footprint=np.ones(P.nms_footprint))
    s_pts    = peak_local_max(s.target.astype(np.float32),threshold_abs=.5,exclude_border=False,footprint=np.ones(P.nms_footprint))

    ## filter border points
    patch  = np.array(s.raw.shape)
    pts2   = [p for p in pts if np.all(p%(patch-P.border) > P.border)]
    s_pts2 = [p for p in s_pts if np.all(p%(patch-P.border) > P.border)]

    matching = match_unambiguous_nearestNeib(s_pts2,pts2,dub=P.match_dub,scale=P.match_scale)
    return SimpleNamespace(pred=y, scores=(l,matching.f1,y.max()))

  def validate_many():
    _valiscores = []

    Nsamples = len(df[df.labels==1])
    # idxs = np.arange(Nsamples)
    # np.random.shuffle(idxs)

    for i in range(Nsamples):
      s = df[df.labels==1].iloc[i] ## no idxs
      _scores = validate_single(s).scores
      _valiscores.append(_scores)
      # if i%10==0: print(f"_scores",_scores, end='\n',flush=True)

    history.valimeans.append(np.nanmean(_valiscores,0))

    ## now save (new) best weights

    torch.save(net.state_dict(), savedir_local / f'm/best_weights_latest.pt')

    valikeys   = ['loss','f1','height']
    valiinvert = [1,-1,-1] # minimize, maximize, maximize
    valis = np.array(history.valimeans).reshape([-1,3])*valiinvert

    for i,k in enumerate(valikeys):
      if np.nanmin(valis[:,i])==valis[-1,i]:
        torch.save(net.state_dict(), savedir_local / f'm/best_weights_{k}.pt')

  def pred_glances(time):
    ids = [0,N_train//2,N_train-1]
    for i in ids:
      # enumerate(trainset.iloc[ids]):
      pred = validate_single(trainset.iloc[i]).pred
      save(img2png(pred),savedir_local/f'glance_output_train/a{time}_{i}.png')

    N_vali = len(df[df.labels==1])
    ids = [0,N_vali//2,N_vali-1]
    for i in ids:
      pred = validate_single(df[df.labels==1].iloc[i]).pred
      save(img2png(pred),savedir_local/f'glance_output_vali/a{time}_{i}.png')

  tic = time()
  n_pix = trainset['shape'].apply(np.prod).sum() / 1_000_000 ## Megapixels
  N_epochs=300 ## MYPARAM
  print(f"\nBegin training for {N_epochs} epochs...\n\n")
  for ep in range(N_epochs):
    backprop_n_samples_into_net()
    validate_many()
    save(history, savedir_local / "history.pkl")
    if ep in range(10) or ep%10==0: pred_glances(ep)
    
    dt  = time() - tic
    tic = time()

    print("\033[F",end='') ## move cursor UP one line 
    print(f"epoch {ep}/{N_epochs}, loss={history.lossmeans[-1]:4f}, dt={dt:4f}, rate={n_pix/dt:5f} Mpix/s", end='\n',flush=True)


## Unstable. predict and evaluate on patches

def evaluate(pid=0):

  t0 = time()

  (p1,p0,),pid = parse_pid(pid,[3,19])
  savedir_local = savedir / f'e26_isbidet/evaluate/pid{pid:03d}/'

  myname, isbiname = isbi_datasets[p0] # v05

  info = get_isbi_info(myname,isbiname,"01") ## Just pick "01" ... it doesn't matter...

  print(f"""
    Begin evaluate() on pid {pid} ... {isbiname}
    Savedir is {savedir_local}
    """)

  traindir = savedir / f"e26_isbidet/train/pid{pid:03d}/"   ## we can use the same PID for train() and evaluate()  

  def build_samples():
    (_p1,_1,),_pid = parse_pid([1,p0],[3,19]) ## get traindir for dataset "01"
    _traindir = savedir / f"e26_isbidet/train/pid{_pid:03d}/"
    samples1  = load(_traindir / "patchFrame.pkl")
    samples1.reset_index(inplace=True)
    samples1['labels'] = load(_traindir / "labels.pkl").astype(np.uint8) #[::-1]
    params1 = load(_traindir / "params.pkl")
    samples1['dataset'] = "01"

    (_p1,_1,),_pid = parse_pid([2,p0],[3,19]) ## get traindir for dataset "02"
    _traindir = savedir / f"e26_isbidet/train/pid{_pid:03d}/"
    samples2  = load(_traindir / "patchFrame.pkl")
    samples2.reset_index(inplace=True)
    samples2['labels'] = load(_traindir / "labels.pkl").astype(np.uint8) #[::-1]
    params2 = load(_traindir / "params.pkl")
    samples2['dataset'] = "02"

    samples = pandas.concat([samples1,samples2])
    params = params1
    return samples, params

  samples, params = build_samples()

  ## filter out empty patches
  # samples = samples[samples.npts>0].reset_index() ## MYPARAM only evaluate patches with points?
  # scalars = [c for c,d in zip(samples.columns,samples.dtypes) if str(d) != 'object']

  print("How many pts per patch? ", Counter(samples['npts']))

  P = params

  def addweights(s):
    if P.sparse:
      w0 = dgen.weights__decaying_bg_multiplier(s.target,0,thresh=np.exp(-0.5*(3)**2),decayTime=None,bg_weight_multiplier=0.0)
    else:
      w0 = np.ones(s.target.shape,dtype=np.float32)
    return w0
  samples['weights'] = samples.apply(addweights,axis=1)

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = _init_unet_params(info.ndim).net
  net = net.to(device)
  net.load_state_dict(torch.load(traindir / f'm/best_weights_loss.pt'))  ## MYPARAM use loss, f1, or some other vali metric ?

  def net_sample(sample):
    s  = sample
    x  = torch.from_numpy(s.raw).float().to(device,  non_blocking=True)
    yt = torch.from_numpy(s.target).float().to(device, non_blocking=True)
    w  = torch.from_numpy(s.weights).float().to(device, non_blocking=True)

    with torch.no_grad(): 
      y  = net(x[None,None])[0,0]
      loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()

    y = y.cpu().numpy()
    loss = np.float(loss.cpu().numpy())
    return y,loss

  def eval_sample(sample):

    s = sample
    y,loss = net_sample(s)

    _peaks = y.copy() #/y.max()

    # if P.sparse:
    #   _peaks[~s.weights.astype(np.bool)] = 0 ## TODO should sparse datasets ignore masked regions?

    pts = peak_local_max(_peaks,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
    s_pts = peak_local_max(s.target.astype(np.float32), threshold_abs=0.9, footprint=np.ones(P.nms_footprint))
    
    patch  = np.array(s.raw.shape)
    pts2   = [p for p in pts if np.all(p%(patch-P.border) > P.border)]
    s_pts2 = [p for p in s_pts if np.all(p%(patch-P.border) > P.border)]
    
    match = match_unambiguous_nearestNeib(s_pts2,pts2,dub=P.match_dub,scale=P.match_scale)

    # _set =   samples['set']     = [['train','vali','test'][l] for l in samples.labels]

    res = dict(
     time = s.time,
     yPred =     y,
     yPts =      pts,
     ytPts =     s_pts,
     loss =      loss,
     logloss =   -np.log10(loss),
     f1 =        match.f1,
     recall =    match.recall,
     precision = match.precision,
     height =    y.max(),
     labels = s.labels,
     set = ['train','vali','test'][s.labels],
     dataset = s.dataset,
    )

    return res
  
  f = lambda : pandas.DataFrame([eval_sample(s) for s in samples.iloc])
  table = memoize(f,savedir_local/"scores_patchTable.pkl",force=1)

  metricNames = ['logloss','f1','recall','height']
  print('\n'+' '*15 + "Metric Means\n",table.groupby('set')[metricNames].mean())
  print('\n'+' '*15 + "Metric StdDev\n",table.groupby('set')[metricNames].std())

  samples = samples.merge(table[table.columns.difference(samples.columns)],left_index=True,right_index=True)
  # df = get_exemplar_rows(samples)

  exemplars = get_exemplars(samples)
  if exemplars: save(exemplars, savedir_local / 'exemplars')

def get_exemplar_rows(samples,metricNames=["f1","recall","precision","height","logloss",],N=3):
  cols1 = samples.columns.difference(metricNames)
  df = samples.melt(id_vars=cols1,var_name='metric',value_name='value')
  def f(group):
    g = group.dropna(subset=['value']).sort_values(by='value')
    ng = len(g)
    idx = np.linspace(0,ng-1,N).astype(int)
    g = g.iloc[idx]
    return g
  df = df.groupby(['set','metric']).apply(f)
  return df

def get_exemplars(table, metrics=['logloss','f1','recall']):
  """
  Time to compute exemplars!
  """

  exemplars = SimpleNamespace()

  for j,tvt in enumerate(['train','vali','test']):
    for metric in metrics:

      m = table.labels==j
      idxs = np.argsort(table[metric][m])
      N = len(idxs)
      if N==0: continue
      raws, labs, scores, yPred, yPts, ytPts = table[m].iloc[idxs].iloc[[0,N//2,N-1]].loc[:,['raw','lab',metric,'yPred','yPts','ytPts']].T.values
      # ipdb.set_trace()

      for j,p in enumerate(['low','median','high']):
        a = raws[j]
        e = labs[j]
        b = scores[j]
        c = yPred[j]
        f = ytPts[j]
        d = np.zeros(c.shape,dtype=np.uint8)
        if len(yPts[j]) > 0: d[tuple(np.array(yPts[j]).T)] = 1
        if len(ytPts[j]) > 0: d[tuple(np.array(ytPts[j]).T)] += 2
        keybase = f"{tvt}_{p}_{metric}_"
        exemplars.__dict__[keybase+'raw']   = blendRawLab(a,e)
        exemplars.__dict__[keybase+'pred']  = blendRawLab(c,d,colors=[(0,0,1),(0,1,0),(1,0,0)])
        exemplars.__dict__[keybase+'score'] = b

  return exemplars


## Unstable. predict and eval on full images with simple metrics (don't call ISBI binaries)

def evaluate_imgFrame(pid=0):

  t0 = time()

  (p1,p0,),pid = parse_pid(pid,[3,19])
  savedir_local = savedir / f'e26_isbidet/evaluate_imgFrame/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05
  # trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,"01") ## NOTE. we use both 01/02

  print(f"""
    Begin pid {pid} ... {info.isbiname} / {info.dataset}.
    Savedir is {savedir_local}
    """)


  def build_samples():
    samples1,params1  = e26_isbidet_dgen.build_imgFrame_and_params([pid,0])
    samples1['dataset'] = "01"
    samples2,params2  = e26_isbidet_dgen.build_imgFrame_and_params([pid,1])
    samples2['dataset'] = "02"

    samples = pandas.concat([samples1,samples2])
    params = params1
    return samples, params

  samples, params = build_samples()

  # samples, params = e26_isbidet_dgen.build_imgFrame_and_params([pid,0])
  # samples = samples.iloc[::10]
  samples['set']     = [['train','test'][l] for l in samples.labels]
  # samples.reset_index(inplace=True)

  # ipdb.set_trace()
  
  print("How many pts per patch? ", Counter(samples['npts']))
  
  P = params

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = _init_unet_params(info.ndim).net
  net = net.to(device)
  weights = f"/projects/project-broaddus/devseg_2/expr/e26_isbidet/train/pid{pid:03d}/m/best_weights_loss.pt"
  net.load_state_dict(torch.load(weights))  ## MYPARAM use loss, f1, or some other vali metric ?

  def eval_sample(sample,withImageData=True):

    s  = sample
    print(f"predict on time {s.time}",end="\r") 

    raw = load(s.rawname).astype(np.float)
    o_shape = raw.shape ## original shape

    ## crop larger datasets

    if P.sparse:
      ## Find bounding box around GT pts, then crop image to GT pts + border region
      pL = (s.pts.min(0) - (6,40,40)).clip(min=(0,0,0), max=raw.shape)
      pR = (s.pts.max(0) + (6,40,40)).clip(min=(0,0,0), max=raw.shape)
      gtpts = s.pts - pL
      ss  = tuple([slice(a,b) for a,b in zip(pL,pR)])
      raw = raw[ss]
    else:
      gtpts = s.pts
      # P.evalBorder = (6,40,40)
      # P.evalBorder = (4,30,30)

    ## rescale all datasets
    # if s.time == 16: ipdb.set_trace()

    _raw = zoom(raw,params.zoom,order=1)
    zoom_effective = _raw.shape / np.array(raw.shape)
    gtPts_small = zoom_pts(gtpts,zoom_effective)
    # P.evalBorder = (P.evalBorder * zoom_effective).astype(int)
    raw = _raw

    ## normalize

    raw = norm_percentile01(raw,2,99.4)
    percentiles = np.percentile(raw,np.linspace(0,100,101))

    ## run through net

    def f_net(patch):
      with torch.no_grad():
        patch = torch.from_numpy(patch).float().cuda()
        return net(patch[None])[0].cpu().numpy() ## add then remove batch dimension
    pred = dgen.apply_net_tiled_nobounds(f_net,raw[None],outchan=1,) # patch_shape=params.patch,border_shape=params.border
    pred = pred[0]

    ## renormalize predictions

    _peaks = pred/pred.max()

    # if P.sparse:
    #   _peaks[~s.weights.astype(np.bool)] = 0 ## TODO should sparse datasets ignore masked regions?
    # ipdb.set_trace()

    ## extract points from predicted image

    pts = peak_local_max(_peaks,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
    yPts_small = pts.copy()

    ## undo scaling

    pts = zoom_pts(pts,1/zoom_effective)

    ## undo cropping / translation for sparse datasets

    if params.sparse:
      pts = pts + pL

    ## filter out boundary points

    # sz_patch  = np.array(_peaks.shape) # / zoom_effective
    # sz_patch = 
    # pts2 = pts / zoom_effective
    # gtpts2  = gtpts / zoom_effective
    # pts2 = pts
    # gtpts2 = s.pts

    o_shape = np.array(o_shape)
    pts2   = [p for p in pts   if np.all(p%(o_shape - P.evalBorder) >= P.evalBorder)]
    # gtpts2 = [p for p in s.pts if np.all(p%(o_shape - P.evalBorder) >= P.evalBorder)]

    ## get matching score on filtered points

    match  = match_unambiguous_nearestNeib(s.pts,pts2,            dub=P.match_dub,scale=P.match_scale) ## filter out boundary points. big space.
    match2 = match_unambiguous_nearestNeib(gtPts_small,yPts_small,dub=P.match_dub,scale=P.match_scale) ## don't filter. small space. THIS MAKES THE DIFFERENCE.
    match3 = match_unambiguous_nearestNeib(s.pts,pts,             dub=P.match_dub,scale=P.match_scale) ## don't filter. big space.
    
    # ipdb.set_trace()

    print(f"Time {s.time} .. F1 {match.f1:.3f} .. {match2.f1:.3f} .. {match3.f1:.3f}",flush=True)

    res = dict(
     time  =       s.time,
     yPts  =       pts,
     ytPts =       s.pts,
     f1 =          match.f1,
     recall =      match.recall,
     precision =   match.precision,
     n_matched =   match.n_matched,
     n_proposed =  match.n_proposed,
     n_gt =        match.n_gt,
     height =      pred.max(), #y.max(),
     yPts_small  = yPts_small,
     gtPts_small = gtPts_small,
     percentiles = percentiles,
     dataset = s.dataset,
     # loss  =     loss,
     # zoom_effective = zoom_effective,
    )

    if withImageData==True:
     res['raw_small']   = raw.astype(np.float16)
     res['yPred_small'] = pred.astype(np.float16)

    if True:
      lab_gt = load(s.labname)
      # target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
      stack  = conv_at_pts4(pts,np.ones([1,]*lab_gt.ndim),lab_gt.shape).astype(np.uint16)
      stack  = expand_labels(stack,25)
      det_score, det_counts = scores_dense.det(lab_gt,stack,return_counts=True)
      res['DET'] = det_score
      res['det_counts'] = det_counts
    return res

  def f_table():
    table = pandas.DataFrame([eval_sample(s,withImageData=False) for s in samples.iloc])
    table = table.merge(samples,on='time',how='left') # left_index=True,right_index=True)
    return table
  
  table = memoize(f_table, savedir_local/f"scores_imgTable.pkl",force=1) ## MYPARAM predict on opposite dataset

  # samples['logloss'] = -np.log10(samples['loss'])
  metricNames = ['f1','recall','height',]
  print('\n'+' '*15 + "Metric Means\n",table.groupby('set')[metricNames].mean())
  print('\n'+' '*15 + "Metric StdDev\n",table.groupby('set')[metricNames].std())

  ## aggregate scores

  def agg_scores_SNN_matching(df):
    ## from my matching
    n_m  = df.n_matched.sum()
    n_p  = df.n_proposed.sum()
    n_gt = df.n_gt.sum()
    precision = n_m / n_p
    recall    = n_m / n_gt
    f1        = 2*n_m / (n_p + n_gt)
    ## from DET matching
    # DET       = scores_dense._det(pandas.DataFrame([x for x in df.det_counts]).sum())
    # return pandas.Series([f1,precision,recall,DET],index=["f1", "precision", "recall", "DET"])
    return pandas.Series([f1,precision,recall],index=["f1", "precision", "recall",])

  agg_scores = table.groupby('set').apply(agg_scores_SNN_matching)
  print('\n'+' '*15 + "Agg Scores \n", agg_scores)

  def f_exemplars():
    ssam = pandas.DataFrame([eval_sample(s,withImageData=True) for s in samples.sample(n=min(len(table),8)).iloc])
    ssam['png'] = ssam.apply(mkpng,axis=1)
    exemplars = SimpleNamespace()
    for row in ssam.iloc: exemplars.__dict__[f'time{row.time}'] = row.png
    return exemplars
  memoize(f_exemplars, savedir_local / f'exemplars_imgFrame',force=1) ## MYPARAM predict on opposite dataset

  # ipdb.set_trace()

  if params.sparse:
    print("SPARSE")

  # ipdb.set_trace()

  def det_tra():
    ltps = list(table.yPts)
    ipdb.set_trace()
    tb = tracking.nn_tracking_on_ltps(ltps=ltps, scale=info.scale, dub=60)
    tracking.eval_tb_isbi(tb,info,savedir_local)
    # ## add {ext} to 01_DET.txt 01_TRA.txt
    shutil.move(savedir_local / f"{info.dataset}_DET.txt", savedir_local / f"isbi_DET{ext}.txt")
    shutil.move(savedir_local / f"{info.dataset}_TRA.txt", savedir_local / f"isbi_TRA{ext}.txt")

    with open(savedir_local / f"isbi_DET{ext}.txt", 'r') as file:
      det = re.match(r'DET measure: (\d\.\d+)',file.read()).group(1)
    with open(savedir_local / f"isbi_TRA{ext}.txt", 'r') as file:
      tra = re.match(r'TRA measure: (\d\.\d+)',file.read()).group(1)
    return dict(det=det,tra=tra,ext=ext)

  # dDetTra = memoize(det_tra,savedir_local/f'dDetTra{ext}.pkl',force=1)
  # agg_scores['ISBI_DET'] = dDetTra['det']
  # agg_scores['ISBI_TRA'] = dDetTra['tra']

  agg_scores.to_pickle(savedir_local / f'agg_scores{ext}.pkl')


def test_outputs():
  img = load("/projects/project-broaddus/rawdata/trib_isbi/crops_2xDown/Fluo-N3DL-TRIF/01/t007.tif").astype(np.float)
  res = load("/projects/project-broaddus/rawdata/trib_isbi/crops_2xDown/Fluo-N3DL-TRIF/01_RES/mask007.tif")
  lab = load("/projects/project-broaddus/rawdata/trib_isbi/crops_2xDown/Fluo-N3DL-TRIF/01_GT/TRA/man_track007.tif")

  # ipdb.set_trace()
  print(scores_dense.det(lab,lab))
  print(scores_dense.det(res,lab))

  # ipdb.set_trace()
  # png = img2png(res)
  save(img2png(img),"/projects/project-broaddus/devseg_2/expr/e24_isbidet_AOT_on_both/v01/pid036/exemplars_imgFrame_1/t007.png")
  save(img2png(res),"/projects/project-broaddus/devseg_2/expr/e24_isbidet_AOT_on_both/v01/pid036/exemplars_imgFrame_1/mask007.png")
  save(img2png(lab),"/projects/project-broaddus/devseg_2/expr/e24_isbidet_AOT_on_both/v01/pid036/exemplars_imgFrame_1/man_track007.png")

# def fixit():
#   for pid in range(36):
#     dPID = pid2params(pid)
#     info = dPID.info
#     savedir_local = dPID.savedir_local
    
#     dataset = info.dataset

#     oldext = ''
#     newext = '_1'
      
#     shutil.move(savedir_local / f'scores_imgTable{oldext}.pkl',  savedir_local / f'scores_imgTable{newext}.pkl', )
#     shutil.move(savedir_local / f'exemplars_imgFrame{oldext}',   savedir_local / f'exemplars_imgFrame{newext}', )
#     shutil.move(savedir_local / f"{dataset}_DET{oldext}.txt",    savedir_local / f"isbi_DET{newext}.txt", )
#     shutil.move(savedir_local / f"{dataset}_TRA{oldext}.txt",    savedir_local / f"isbi_TRA{newext}.txt", )

def get_exemplars_imgFrame(table, metricNames = ['f1','recall','height']):
  """
  Time to compute exemplars!
  """

  exemplars = SimpleNamespace()

  for j,tvt in enumerate(['train','vali','test']):
    for metric in metricNames:

      m = table.labels==j
      idxs = np.argsort(table[metric][m])
      N = len(idxs)
      if N==0: continue
      for j,p in enumerate(['low','median','high']):
        pass

  return exemplars

def mkpng(imgFrameRow):
  R = imgFrameRow
  a = img2png(R.raw_small)
  b = img2png(R.yPred_small,colors=plt.cm.magma)
  c = np.zeros(R.raw_small.shape,dtype=np.uint8)
  if len(R.yPts_small) > 0: c[tuple(np.array(R.yPts_small).T)] = 1
  if len(R.gtPts_small) > 0: c[tuple(np.array(R.gtPts_small).T)] += 2
  c = img2png(c,colors=[(0,0,1),(0,1,0),(1,0,0)])
  a = (a + b)//2
  png = blendRawLabPngs(a,c)
  # png = blendRawLab(a,d,colors=[(0,0,1),(0,1,0),(1,0,0)])
  return png

## compile scores from all predictions

def compile_scores():

  # res = load(savedir / "e26_isbidet/compile_scores/scores_evaluate.pkl")
  # ipdb.set_trace()

  def load_df(name,pid,keepers=None):
    name = str(name).format(pid=pid)

    (p1,p0,),pid = parse_pid(pid,[3,19])

    try:
      df = load(name)
      if keepers: df = df[keepers]
      print("Worked:", [p1,p0])
    except:
      df = pandas.DataFrame([])

    df['my_id'] = p0
    df['trainset'] = ["01+02","01","02"][p1]
    df['pid'] = pid

    return df

  keepers = ['loss', 'logloss', 'f1', 'recall', 'precision', 'height', ## losses / metrics
            'time', 
            'labels', ## equivalent to `set`
            'set', ## train/vali/test
            'dataset', ## 01/02
            ]
  d_evaluate = savedir / "e26_isbidet/evaluate/pid{pid:03d}/scores_patchTable.pkl"
  scores_evaluate = [load_df(d_evaluate,pid,keepers=keepers) for pid in range(3*19)]
  # ipdb.set_trace()
  scores_evaluate = pandas.concat(scores_evaluate)
  scores_evaluate = scores_evaluate.reset_index()
  save(scores_evaluate, savedir/"e26_isbidet/compile_scores/scores_evaluate.pkl")
  
  # keepers = ["time", "f1", "recall", "precision", "height", "pid", "shape","npts","labels","zoom_effective","set",]
  # d_evaluate_imgFrame = savedir / "e26_isbidet/evaluate_imgFrame/pid{pid:03d}/agg_scores_1.pkl"
  # scores_evaluate_imgFrame = [load_df(d_evaluate_imgFrame,pid) for pid in range(19)]
  # scores_evaluate_imgFrame = pandas.concat(scores_evaluate_imgFrame)
  # scores_evaluate_imgFrame = scores_evaluate_imgFrame.reset_index()
  # save(scores_evaluate_imgFrame, savedir/"e26_isbidet/compile_scores/scores_evaluate_imgFrame.pkl")
  # # ipdb.set_trace()

def test_compile_scores():
  res = load(savedir/"e26_isbidet/compile_scores/scores_evaluate_imgFrame.pkl")
  ipdb.set_trace()










if __name__=='__main__':
  # for i in range(19*2):
  #   train(i)
  myrun_slurm(range(19*2))