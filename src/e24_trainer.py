"""
Training detection for the ISBI datasets in a data-generator / sampler agnostic way.
Works with e24_isbidet_AOT.py "ahead of time" sampler and JIT sampler.
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
except ImportError as e:
    print("Can't import gputools on non-gpu node...\n", e)

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

from collections import Counter

import augmend
from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale,IntensityScaleShift,Identity

import datagen as dgen
import e24_isbidet_AOT


savedir = savedir_global()
print("savedir:", savedir)

print(f"Total Import Time: {time()-_start_time}")

import matplotlib.pyplot as plt

## Stable. Utils.

def wipedir(path):
  path = Path(path)
  if path.exists(): shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)

def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy("/projects/project-broaddus/devseg_2/src/e24_trainer.py", "/projects/project-broaddus/devseg_2/src/temp/e24_trainer_copy.py")
  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 4 -t 2:00:00 --mem 128000 "    ## TODO: more cores?
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = 'sbatch -J e24_{pid:03d} {_resources} -o slurm/e24_pid{pid:03d}.out -e slurm/e24_pid{pid:03d}.err --wrap \'python3 -c \"import temp.e24_trainer_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
  slurm = slurm.replace("{_resources}",_gpu)
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)

def myrun_slurm_entry(pid=0):
  # import e24_isbidet_AOT
  # e24_isbidet_AOT.build_trainingdata(pid)
  train(pid)
  evaluate(pid)

def pid2params(pid):
  (p0,p1),pid = parse_pid(pid,[19,2])
  savedir_local = savedir / f'e24_isbidet_AOT/v01/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  return SimpleNamespace(**locals())

def norm_minmax01(x):
  return (x-x.min())/(x.max()-x.min())

def blendRawLab(raw,lab,labcolors=None):
  pngraw = _png(raw)
  pnglab = _png(lab,labcolors=labcolors)
  m = pnglab[:,:,:3]==0
  m = np.all(m,-1)
  pngraw[~m] = pnglab[~m]
  # pngraw[~m] = (0.5*pngraw+0.5*pnglab)[~m]
  return pngraw

def _png(x,labcolors=None):

  def colorseg(seg):
    if labcolors:
      cmap = np.array([(0,0,0)] + labcolors*256)[:256]
    else:
      cmap = np.random.rand(256,3).clip(min=0.2)
      cmap[0] = (0,0,0)

    cmap = matplotlib.colors.ListedColormap(cmap)

    m = seg!=0
    seg[m] %= 255 ## we need to save a color for black==0
    # seg[m] += 1
    seg[seg==0] = 255
    seg[~m] = 0
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

def uniqueNdim(a):
  assert a.ndim==2
  uniq   = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
  counts = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1]))),return_counts=True)[1] #.view(a.dtype).reshape(-1, a.shape[1])
  return uniq, counts



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


## Unstable. Training functions.

def train(pid=0):

  P = pid2params(pid)

  info = P.info
  savedir_local = P.savedir_local
  wipedir(savedir_local/'m')

  print(f"""
    Begin pid {pid} ... {info.isbiname} / {info.dataset}.
    Savedir is {savedir_local}
    """)

  df = load(savedir_local / "patchFrame.pkl")
  params = load(savedir_local / "params.pkl")
  e24_isbidet_AOT.describe_virtual_samples(df)

  P = params
  P.scale = info.scale

  
  CONTINUE = 0 ## MYPARAM continue training existing dataset
  print("CONTINUE ? : ", bool(CONTINUE))

  
  N = len(df)
  if CONTINUE:
    labels = load(savedir_local / "labels.pkl")
  else:
    a,b = N*5//8,N*7//8  ## MYPARAM train / vali / test fractions
    labels = np.zeros(N,dtype=np.uint8)
    labels[a:b]=1; labels[b:]=2 ## 0=train 1=vali 2=test
    np.random.shuffle(labels)
    save(labels,savedir_local / "labels.pkl")


  df['labels'] = labels
  assert len(labels)>8

  f_aug = build_augmend(df.raw.iloc[0].ndim)

  def addweights(s):
    if P.sparse:
      w0 = dgen.weights__decaying_bg_multiplier(s.target,0,thresh=np.exp(-0.5*(3)**2),decayTime=None,bg_weight_multiplier=0.0)
    else:
      w0 = np.ones(s.target.shape,dtype=np.float32)
    return w0
  df['weights'] = df.apply(addweights,axis=1)

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = _init_unet_params(info.ndim).net
  net = net.to(device)
  
  if CONTINUE:
    net.load_state_dict(torch.load(savedir_local / f'm/best_weights_latest.pt')) ## MYPARAM start off from best_weights ?
    history = load(savedir_local / 'history.pkl')
  else:
    history = SimpleNamespace(lossmeans=[],valimeans=[],)
    wipedir(savedir_local/'m')
    wipedir(savedir_local/"glance_output_train/")
    wipedir(savedir_local/"glance_output_vali/")

  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)

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
    _fp = np.ones([3,5,5],dtype=np.bool) if x.ndim==3 else np.ones([5,5],dtype=np.bool)
    pts      = peak_local_max(_peaks,threshold_abs=.5,exclude_border=False,footprint=_fp)
    s.pts    = peak_local_max(s.target.astype(np.float32),threshold_abs=.5,exclude_border=False,footprint=_fp)

    ## filter border points
    patch  = np.array(s.raw.shape)
    border = (2,5,5)[-x.ndim:]
    pts2   = [p for p in pts if np.all(p%(patch-border) > border)]
    s.pts2 = [p for p in s.pts if np.all(p%(patch-border) > border)]

    if info.ndim==3: P.scale = P.scale*(0.5,1,1) ## to account for low-resolution and noise along z dimension (for matching)
    matching = match_unambiguous_nearestNeib(s.pts2,pts2,dub=100,scale=P.scale)
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
      save(_png(pred),savedir_local/f'glance_output_train/a{time}_{i}.png')

    N_vali = len(df[df.labels==1])
    ids = [0,N_vali//2,N_vali-1]
    for i in ids:
      pred = validate_single(df[df.labels==1].iloc[i]).pred
      save(_png(pred),savedir_local/f'glance_output_vali/a{time}_{i}.png')

  tic = time()
  N_epochs=500 ## MYPARAM
  print(f"\nBegin training for {N_epochs} epochs...\n\n")
  for ep in range(N_epochs):
    backprop_n_samples_into_net()
    validate_many()
    save(history, savedir_local / "history.pkl")
    pred_glances(ep)
    dt  = time() - tic
    tic = time()

    print("\033[F",end='') ## move cursor UP one line 
    print(f"epoch {ep}/{N_epochs}, loss={history.lossmeans[-1]:4f}, dt={dt:4f}, rate={N_train/dt:5f} samples/s", end='\n',flush=True)


## predict and evaluate on patches

def evaluate(pid=0):

  t0 = time()

  P = pid2params(pid)

  info = P.info
  savedir_local = P.savedir_local

  print(f"""
    Begin pid {pid} ... {info.isbiname} / {info.dataset}.
    Savedir is {savedir_local}
    """)

  samples = load(savedir_local / "patchFrame.pkl")
  samples.reset_index(inplace=True)
  samples['labels'] = load(savedir_local / "labels.pkl").astype(np.uint8)
  params = load(savedir_local / "params.pkl")

  ## filter out empty patches
  samples = samples[samples.npts>0].reset_index() ## MYPARAM only evaluate patches with points?

  scalars = [c for c,d in zip(samples.columns,samples.dtypes) if str(d) != 'object']

  print("How many pts per patch? ", Counter(samples['npts']))

  P = params
  P.scale = info.scale

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
  net.load_state_dict(torch.load(savedir_local / f'm/best_weights_loss.pt'))  ## MYPARAM use loss, f1, or some other vali metric ?

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

    _fp = np.ones([5,5,5],dtype=np.bool) if sample.raw.ndim==3 else np.ones([5,5],dtype=np.bool)
    pts = peak_local_max(_peaks,threshold_abs=.2,exclude_border=False,footprint=_fp)
    s_pts = peak_local_max(s.target.astype(np.float32), threshold_abs=0.9, footprint=_fp)
    # s_pts = 
    
    patch = np.array(s.raw.shape)
    border = (1,1,1)[-s.raw.ndim:]
    pts2   = [p for p in pts if np.all(p%(patch-border) > border)]
    s_pts2 = [p for p in s_pts if np.all(p%(patch-border) > border)]
    
    _dub = 100 if P.sparse else 100
    if info.ndim==3: P.scale = P.scale*(0.5,1,1) ## to account for low-resolution and noise along z dimension (for matching)
    # match = match_unambiguous_nearestNeib(s_pts2,pts2,dub=_dub,scale=P.scale)
    match = match_unambiguous_nearestNeib(s_pts2,pts2,dub=_dub,scale=(1,1,1))

    res = dict(
     yPred =     y,
     yPts =      pts,
     ytPts =     s_pts,
     loss =      loss,
     f1 =        match.f1,
     recall =    match.recall,
     precision = match.precision,
     height =    y.max(),
    )

    return res
  

  if (savedir_local/"pred_table.pkl").exists():
    table = pandas.DataFrame([eval_sample(s) for s in samples.iloc])
    table.to_pickle(savedir_local/"pred_table.pkl")
  else:
    table = load(savedir_local/"pred_table.pkl")

  # ipdb.set_trace()
  samples = samples.merge(table,left_index=True,right_index=True)

  samples['logloss'] = -np.log10(samples['loss'])
  samples['set']     = [['train','vali','test'][l] for l in samples.labels]

  metricNames = ['logloss','f1','recall','height']

  ipdb.set_trace()

  print('\n'+' '*15 + "Metric Means\n",samples.groupby('set')[metricNames].mean())
  print('\n'+' '*15 + "Metric StdDev\n",samples.groupby('set')[metricNames].std())

  ipdb.set_trace()
  exemplars = get_exemplars(samples)
  if exemplars: save(exemplars, savedir_local / 'exemplars')


def get_exemplars2(samples):
  samples.groupby('set')[metricNames].sample(n=3)

def get_exemplars(table):
  """
  Time to compute exemplars!
  """

  exemplars = SimpleNamespace()
  # ipdb.set_trace()

  for j,tvt in enumerate(['train','vali','test']):
    for metric in ['logloss','f1','recall']:

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
        exemplars.__dict__[keybase+'pred']  = blendRawLab(c,d,labcolors=[(0,0,1),(0,1,0),(1,0,0)])
        exemplars.__dict__[keybase+'score'] = b

  return exemplars


if __name__=='__main__':
  # for i in range(19*2):
  #   train(i)
  myrun_slurm(range(19*2))