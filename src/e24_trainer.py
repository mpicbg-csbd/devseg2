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


savedir = savedir_global()
print("savedir:", savedir)

print(f"Total Import Time: {time()-_start_time}")

import matplotlib.pyplot as plt

## Stable. Utils.

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

def _png(x,labcolor=None):

  def colorseg(seg):
    if labcolor:
      cmap = np.full((256,3), labcolor)
    else:
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




# Stable training functions.

def _init_unet_params(ndim):
  T = SimpleNamespace()
  if ndim==2:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
  elif ndim==3:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
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
  # print(json.dumps(P.info.__dict__,sort_keys=True, indent=2, default=str), flush=True)
  info = P.info
  savedir_local = P.savedir_local
  (savedir_local / 'm').mkdir(exist_ok=True,parents=True) ## place to save models

  print(f"""
    Begin pid {pid} ... {info.isbiname} / {info.dataset}.
    Savedir is {savedir_local}
    """)

  # ## initial patch shit
  # params  = init_params(info.ndim)
  # _params = cpnet_data_specialization(info)
  # for k in _params.__dict__.keys(): params.__dict__[k] = _params.__dict__[k]

  fulldata = load(savedir_local / "fullsamples2")
  # fulldata = pandas.read_pickle(savedir_local / "patchFrame.pkl")

  print(Counter([s.pts.shape[0] for s in fulldata.samples]))
  fulldata.samples = [s for s in fulldata.samples if s.pts.shape[0]>0]
  print(Counter([s.pts.shape[0] for s in fulldata.samples]))
  # ipdb.set_trace()

  P = fulldata.params
  P.scale = info.scale
  if info.ndim==3: P.scale = P.scale*(0.5,1,1) ## to account for low-resolution and noise along z dimension (for matching)

  N = len(fulldata.samples)
  a,b = N*5//8,N*7//8
  labels = np.zeros(N)
  labels[a:b]=1; labels[b:]=2 ## 0=train 1=vali 2=test
  np.random.shuffle(labels)
  save(labels,savedir_local / "labels.pkl")

  inds = np.r_[:N]
  split = SimpleNamespace(train=inds[:a],vali=inds[a:b],test=inds[b:])

  trainloader  = [fulldata.samples[i] for i in split.train]
  valiloader   = [fulldata.samples[i] for i in split.vali]
  testloader   = [fulldata.samples[i] for i in split.test]
  glance_train = trainloader[:3]
  glance_vali  = valiloader[:3]

  f_aug = build_augmend(trainloader[0].raw.ndim)

  def addweights(s):
    if P.sparse:
      w0 = dgen.weights__decaying_bg_multiplier(s.target,0,thresh=np.exp(-0.5*(3)**2),decayTime=None,bg_weight_multiplier=0.0)
    else:
      w0 = np.ones(s.target.shape,dtype=np.float32)
    # w = np.zeros(s.target.shape)
    # w[s.slice.c] = w0[s.slice.c]
    s.weights = w0

  for s in trainloader: addweights(s)
  for s in valiloader: addweights(s)

  tic = time()

  def mse_loss(net,sample,augment=False):
    s  = sample

    x  = s.raw.copy()
    yt = s.target.copy()
    w  = s.weights.copy()
    
    if augment: x,yt,w = f_aug([x,yt,w])

    x  = x.copy()
    yt = yt.copy()
    w  = w.copy()

    x  = torch.from_numpy(x ).float().to(device, non_blocking=True)
    yt = torch.from_numpy(yt).float().to(device, non_blocking=True)
    w  = torch.from_numpy(w ).float().to(device, non_blocking=True)

    y  = net(x[None,None])[0,0]

    ## Introduce ss.b masking to ensure that backproped pixels do not overlap between train/vali/test
    
    loss = torch.abs((w*(y-yt)**2)).mean() 
    return y,loss

  def validate(net,sample):
    s = sample
    with torch.no_grad(): y,l = mse_loss(net,s)

    y = y.cpu().numpy()
    l = l.cpu().numpy()
    _peaks = y.copy() #y/y.max()
    _fp = np.ones([3,5,5],dtype=np.bool) if sample.raw.ndim==3 else np.ones([5,5],dtype=np.bool)
    pts      = peak_local_max(_peaks,threshold_abs=.5,exclude_border=False,footprint=_fp)
    s.pts    = peak_local_max(s.target.astype(np.float32),threshold_abs=.5,exclude_border=False,footprint=_fp)

    ## filter border points
    patch  = np.array(s.raw.shape)
    border = (2,5,5)
    pts2   = [p for p in pts if np.all(p%(patch-border) > border)]
    s.pts2 = [p for p in s.pts if np.all(p%(patch-border) > border)]

    matching = match_unambiguous_nearestNeib(s.pts2,pts2,dub=100,scale=P.scale)
    return SimpleNamespace(pred=y, scores=(l,matching.f1,y.max()))

  def pred_glances(net,time):
    for i,s in enumerate(glance_train):
      with torch.no_grad(): y,l = mse_loss(net,s)
      y = y.cpu().numpy()
      l = l.cpu().numpy()
      if y.ndim==3: y = y.max(0)
      y = (norm_minmax01(y)*255).astype(np.uint8)
      save(y,savedir_local/f'glance_output_train/a{time}_{i}.png')

    for i,s in enumerate(glance_vali):
      with torch.no_grad(): y,l = mse_loss(net,s)
      y = y.cpu().numpy()
      l = l.cpu().numpy()
      if y.ndim==3: y = y.max(0)
      y = (norm_minmax01(y)*255).astype(np.uint8)
      save(y,savedir_local/f'glance_output_vali/a{time}_{i}.png')

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = _init_unet_params(info.ndim).net
  net = net.to(device)
  
  net.load_state_dict(torch.load(savedir_local / f'm/best_weights_latest.pt'))
  history = load(savedir_local / 'history.pkl')

  history = train_net(net,mse_loss,validate,trainloader,valiloader,N_vali=len(valiloader),N_train=len(trainloader),N_epochs=20,history=history,pred_glances=pred_glances,savedir=savedir_local)

  toc = time()
  print("TRAIN TIME:", toc-tic)

  ## Training
  return history

def train_net(net,f_loss,f_vali,trainloader,valiloader,N_vali=20,N_train=100,N_epochs=3,history=None,pred_glances=None,savedir=None):
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)
  if history is None: 
    history = SimpleNamespace()
    history.lossmeans = []
    history.valimeans = []

  tic = time()

  # trainloader = iter(trainloader)
  # valiloader = iter(valiloader)

  for i in range(N_epochs):
    loss = backprop_n_samples_into_net(net,opt,f_loss,trainloader,N_train)
    history.lossmeans.append(loss)
    vali = validate_me_baby(net,f_vali,valiloader,N_vali)
    history.valimeans.append(vali)
    save(history, savedir / "history.pkl")
    pred_glances(net,i)
    save_best_weights(net,vali,savedir)
    
    dt  = time() - tic
    tic = time()
    print(f"epoch {i}/{N_epochs}, loss={history.lossmeans[-1]:4f}, dt={dt:4f}, rate={N_train/dt:5f} samples/s", end='\r',flush=True)

  return history

def save_best_weights(net,_vali,savedir):
  torch.save(net.state_dict(), savedir / f'm/best_weights_latest.pt')

  valikeys   = ['loss','f1','height']
  valiinvert = [1,-1,-1] # minimize, maximize, maximize
  valis = np.array(_vali).reshape([-1,3])*valiinvert

  for i,k in enumerate(valikeys):
    if np.nanmin(valis[:,i])==valis[-1,i]:
      torch.save(net.state_dict(), savedir / f'm/best_weights_{k}.pt')

def validate_me_baby(net,f_vali,valiloader,nsamples):
  valis = []
  # for j,vs in tqdm(enumerate(valiloader),total=N_vali,ascii=True):
  for i in range(nsamples):
    s = valiloader[i]
    _scores = f_vali(net,s).scores
    valis.append(_scores)
    print(f"_scores",_scores, end='\n',flush=True)

  return np.nanmean(valis,0)

def backprop_n_samples_into_net(net,opt,f_loss,trainloader,nsamples):
  _losses = []
  tic = time()
  print()
  for i in range(nsamples):
    s = trainloader[i]
    y,l = f_loss(net,s,augment=True)
    l.backward()
    opt.step()
    opt.zero_grad()
    _losses.append(float(l.detach().cpu()))
    dt = time()-tic; tic = time()
    print(f"it {i}/{nsamples}, dt {dt:5f}, max {float(y.max()):5f}", end='\r',flush=True)
  # print("\033[F",end='')
  return np.mean(_losses)



## predict

def evaluate(pid=0):

  t0 = time()
  # print(f"XXX1 time is {time()-t0}")

  P = pid2params(pid)
  # print(json.dumps(P.info.__dict__,sort_keys=True, indent=2, default=str), flush=True)
  info = P.info
  savedir_local = P.savedir_local
  (savedir_local / 'm').mkdir(exist_ok=True,parents=True) ## place to save models

  print(f"""
    Begin pid {pid} ... {info.isbiname} / {info.dataset}.
    Savedir is {savedir_local}
    """)

  fulldata = load(savedir_local / "fullsamples2")
  
  samples = pandas.DataFrame([x.__dict__ for x in fulldata.samples])

  scalars = ['pts', 'time', 'slice', 'npts']

  samples['npts'] = samples['pts'].apply(len)
  print(samples.groupby('npts')[scalars].describe())
  split = load(savedir_local / "split.pkl")
  # samples['label']  = [['train','vali','test'][i] for i in split]
  # samples['label2'] = [[0,1,2][i] for i in split]

  print(Counter([s.pts.shape[0] for s in samples.iloc]))

  splitLabels = np.zeros(len(fulldata.samples),dtype=np.uint)
  splitLabels[split.vali] = 1
  splitLabels[split.test] = 2


  # print(f"XXX3 time is {time()-t0}")

  P = fulldata.params
  P.scale = info.scale
  if info.ndim==3: P.scale = P.scale*(0.5,1,1) ## to account for low-resolution and noise along z dimension (for matching)

  def addweights(s):
    if P.sparse:
      # ipdb.set_trace()
      s.weights = dgen.weights__decaying_bg_multiplier(s.target,0,thresh=np.exp(-0.5*(3)**2),decayTime=None,bg_weight_multiplier=0.0)
    else:
      s.weights = np.ones_like(s.target)
  for s in fulldata.samples: addweights(s)
  # for s in valiloader: addweights(s)

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  net = _init_unet_params(info.ndim).net
  net = net.to(device)
  net.load_state_dict(torch.load(savedir_local / f'm/best_weights_loss.pt'))

  # print(f"XXX4 time is {time()-t0}")

  def eval_sample(net,sample):

    s = sample
    y,loss = net_sample(net,s)

    _peaks = y.copy() #/y.max()

    # if P.sparse:
    #   _peaks[~s.weights.astype(np.bool)] = 0

    _fp = np.ones([3,5,5],dtype=np.bool) if sample.raw.ndim==3 else np.ones([5,5],dtype=np.bool)
    pts = peak_local_max(_peaks,threshold_abs=.2,exclude_border=False,footprint=_fp)
    s.pts = peak_local_max(s.target.astype(np.float32), threshold_abs=0.9, footprint=np.ones(P.nms_footprint))
    patch = np.array(s.raw.shape)
    border = (2,5,5)
    pts2   = [p for p in pts if np.all(p%(patch-border) > border)]
    s.pts2 = [p for p in s.pts if np.all(p%(patch-border) > border)]
    _dub  = 3 if P.sparse else 100
    match = match_unambiguous_nearestNeib(s.pts2,pts2,dub=_dub,scale=P.scale)
    if len(s.pts2)>0:
      print(s.pts, s.pts2, pts, pts2)
      print(_peaks.shape)
      # ipdb.set_trace()

    return y, pts, loss, match.f1, match.recall, y.max()

  
  table = pandas.DataFrame([eval_sample(net,s) for s in fulldata.samples],
                            columns=['yPred','yPts','loss','f1','recall','height'],)
                            # dtype=[np.ndarray,np.ndarray,np.float,np.float,np.float,np.float])
  # ipdb.set_trace()
  table['samples'] = fulldata.samples
  table['logloss'] = -np.log10(table['loss'])
  # table['yPts'] = allscores[:,1]
  # table['yPred'] = allscores[:,0]

  # table.metricNames = metricNames
  table['labels'] = splitLabels
  table['set']    = [['train','vali','test'][l] for l in splitLabels]
  table['npts']   = [s.pts.shape[0] for s in fulldata.samples]
  # table['ytPts']  = [s.pts for s in fulldata.samples]

  metricNames = ['logloss','f1','recall','height']

  print('\n'+' '*15 + "Metric Means\n",table.groupby('set')[metricNames].mean())
  print('\n'+' '*15 + "Metric StdDev\n",table.groupby('set')[metricNames].std())

  exemplars = get_exemplars(net,table)
  if exemplars: save(exemplars, savedir_local / 'exemplars')

def net_sample(net,sample):
  s  = sample
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  x  = torch.from_numpy(s.raw).float().to(device,  non_blocking=True)
  yt = torch.from_numpy(s.target).float().to(device, non_blocking=True)
  # w  = torch.from_numpy(s.weights).float().to(device,  non_blocking=True)
  w  = torch.from_numpy(s.weights).float().to(device, non_blocking=True)

  with torch.no_grad(): 
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()

  y = y.cpu().numpy()
  loss = np.float(loss.cpu().numpy())
  return y,loss

def get_exemplars(net,table):
  """
  Time to compute exemplars!
  """

  exemplars = SimpleNamespace()

  for j,tvt in enumerate(['train','vali','test']):
    for metric in ['logloss','f1','recall']:

      m = table.labels==j
      idxs = np.argsort(table[metric][m])
      N = len(idxs)
      if N==0: continue
      samples, scores, yPred, yPts = table[m].iloc[idxs].iloc[[0,N//2,N-1]].loc[:,['samples',metric,'yPred','yPts']].T.values

      for j,p in enumerate(['low','median','high']):
        a = samples[j]
        b = scores[j]
        c = yPred[j]
        d = np.zeros(c.shape,dtype=np.uint8)
        if len(yPts[j]) > 0: d[tuple(np.array(yPts[j]).T)] = 1
        keybase = f"{tvt}_{p}_{metric}_"
        exemplars.__dict__[keybase+'raw']   = blendRawLab(a.raw,a.lab)
        exemplars.__dict__[keybase+'pred']  = blendRawLabPred(c,d)
        exemplars.__dict__[keybase+'score'] = b

  return exemplars


# def print_table_scores(table):
#   # dk = scores.__dict__
#   print(" "*10 + " loss | f1  | height")
#   for k in dk.keys():
#     a = f"{k:<10}"
#     b = "{:.3f} {:.3f} {:.3f}".format(*dk[k].mean(0))
#     c = " ± "
#     d = "{:.3f} {:.3f} {:.3f}".format(*dk[k].std(0))
#     print(a,b,c,d)

def blendRawLab(raw,lab):
  pngraw = _png(raw)
  pnglab = _png(lab)
  m = pnglab[:,:,:3]==0
  m = np.all(m,-1)
  pngraw[~m] = (0.5*pngraw+0.5*pnglab)[~m]
  return pngraw

def blendRawLabPred(raw,lab):
  pngraw = _png(raw)
  pnglab = _png(lab,labcolor=(1,0,1))
  m = pnglab[:,:,:3]==0
  m = np.all(m,-1)
  pngraw[~m] = pnglab[~m] #(0.5*pngraw+0.5*pnglab)[~m]
  return pngraw



if __name__=='__main__':
  # for i in range(19*2):
  #   train(i)
  myrun_slurm(range(19*2))