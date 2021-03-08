from time import time
_start_time = time()

from itertools import islice
from experiments_common import rchoose, partition, shuffle_and_split, parse_pid, iterdims, savedir_global
import datagen as dgen

# from segtools.render import rgb_max
from models import * #CenterpointModel, SegmentationModel, StructN2V
from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale
# from tracking import nn_tracking_on_ltps, random_tracking_on_ltps
from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
# from segtools.numpy_utils import normalize3, perm2, collapse2, splt, plotgrid
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

# import e24_isbi_datagen
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path

try:
    import gputools
except ImportError as e:
    print("Can't import gputools on non-gpu node...\n", e)

import ipdb
from time import time

savedir = savedir_global()
print("savedir:", savedir)

print(f"Total Import Time: {time()-_start_time}")


def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy("/projects/project-broaddus/devseg_2/src/e21_isbidet.py", "/projects/project-broaddus/devseg_2/src/e21_isbidet_copy.py")
  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 16 -t 2:00:00 --mem 128000 "
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = 'sbatch -J e21_{pid:03d} {_resources} -o slurm/e21_pid{pid:03d}.out -e slurm/e21_pid{pid:03d}.err --wrap \'python3 -c \"import e21_isbidet_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
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


def _init_unet_params(ndim):
  T = SimpleNamespace()
  if ndim==2:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
    T.nms_footprint = [9,9]
  elif ndim==3:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
    T.nms_footprint = [3,9,9]
  return T

def isbiInfo_to_filenames(info):
  n_raw  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
  n_lab  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track
  if info.index in [6,11,12,13,14,15,18]:
    n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
    n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  filenames  = [(n_raw.format(time=i),n_lab.format(time=i)) for i in range(info.start,info.stop)]
  pointfiles = f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl"
  return filenames, pointfiles

class FullDynamicSampler(IterableDataset):
  def __init__(self,data,train_mode=1,N_samples=1_000,savefile=None):
    super().__init__()

    self.data = data #[f(a,b) for a,b in self.filenames][self.timeslice]
    # self.ltps = ltps #load(self.pointfiles)[self.timeslice] if self.pointfiles else None
    self.N_samples = N_samples
    self.timeslice  = slice(0,len(data))
    self.train_mode = train_mode
    self.savefile   = savefile
    if savefile is None or not Path(savefile).exists():
      self.reified_samples = []
    else:
      self.reified_samples = load(savefile)
      print(f"Savefile Loaded with {len(self.reified_samples)} samples...")

  def _init_params(self,ndim):
    if ndim==2:
      self.zoom  = (1,1) #(0.5,0.5)
      self.kern  = [5,5]
      self.patch = (512,512)
    elif ndim==3:
      self.zoom   = (1,1,1) #(1,0.5,0.5)
      self.kern   = [2,5,5]
      self.patch  = (16,128,128)
    self.patch = np.array(self.patch)

  def __iter__(self):
    self.ndim = self.data[0].raw.ndim
    self._init_params(self.ndim)
    self.counter = 0
    self.f_augment = self.augmenter(self.ndim)
    
    while True:
      if len(self.reified_samples) >= self.N_samples:
        s = self.reified_samples[self.counter % self.N_samples]
      else:
        s = self.sample(self.counter,self.train_mode)
        self.reified_samples.append(s)
      yield s
      self.counter += 1

  def __len__(self):
    return self.N_samples

  def sample(self,time,train_mode=1):
    d = rchoose(self.data)
    # t = ((time//40)*3)%len(self.data)
    # d   = self.data[t]

    pts = d.pts
    pt  = rchoose(pts)
    _patchsize  = (self.patch / self.zoom).astype(int)
    ss  = dgen.jitter_center_inbounds(pt,_patchsize,d.raw.shape,jitter=0.1)

    raw,lab = d.raw[ss].copy(), d.lab[ss].copy()
    p2,p99  = np.percentile(raw,[2,99.4]) ## for speed
    raw     = (raw-p2)/(p99-p2)

    if train_mode:
      raw,lab = self.f_augment([raw,lab])

    raw = zoom(raw,self.zoom,order=1)
    # raw = gputools.scale(raw,self.zoom,interpolation='linear')
    # p2,p99 = np.percentile(raw[::4,::4,::4],[2,99.4]) ## for speed
    pts     = dgen.mantrack2pts(lab)
    pts     = (np.array(pts) * self.zoom).astype(np.int)
    target  = dgen.place_gaussian_at_pts(pts,raw.shape,self.kern)
    weights = np.ones_like(target)

    return SimpleNamespace(x=raw,yt=target,w=weights,yt_pts=pts,zdim=0)

  def reaugment_sample(self,sample,time,train_mode=1):
    s=sample
    if train_mode:
      x,yt,y,w = s.x,ys.t,s.y,s.w,
      s.x,ys.t,s.y,s.w, = self.f_re_augment(s.x,ys.t,s.y,s.w,)

  def augmenter(self,ndim):
    aug = Augmend()
    ax = {2:(0,1), 3:(1,2)}[ndim]
    if ndim==3:
      aug.add([FlipRot90(axis=0), FlipRot90(axis=0),], probability=1)
      aug.add([FlipRot90(axis=(1,2)), FlipRot90(axis=(1,2)),], probability=1)
    else:
      aug.add([FlipRot90(axis=(0,1)), FlipRot90(axis=(0,1)),], probability=1)
    aug.add([Rotate(axis=ax, order=1),
             Rotate(axis=ax, order=1),],
            probability=0.5)
    return aug

def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset  # the dataset copy in this worker process 
  N = len(dataset.data) 
  M = worker_info.num_workers
  a = ceil(N/M)
  _id = worker_info.id
  s = slice(_id*a,(_id+1)*a)
  dataset.timeslice = s
  # print(locals())

def compress(sample_list):
  s = SimpleNamespace()
  _s = s.__dict__
  norm = lambda x: (x-x.min())/(x.max()-x.min())
  for i,x in enumerate(sample_list):
    for k,v in x.__dict__.items():
      if k=='x':
        _s[f'x{i}']  = (norm(v.max(0))*255).astype(np.uint8)
      if k=='yt':
        _s[f'yt{i}'] = (norm(v.max(0))*255).astype(np.uint8)
  return s



def myrun(pid=0):
  """
  v01 : refactor of e18. make traindata AOT.
    add `_train` 0 = predict only, 1 = normal init, 2 = continue
    datagen generator -> CenterpointGen, customizable loss and validation metrics, and full detector2 refactor.
  v02 : optimizing hyperparams. in [2,19,5,2] or [2,19,2,5,2] ? 
    p0: dataset in [01,02]
    p1: isbi dataset in [0..18]
    p2: sample flat vs content [0,1] ## ?WAS THIS REALLY HERE?
    p3: random variation [0..4]
    p4: pixel weights (should be per-dataset?) [0,1]
  v03 : explore kernel size. [2,50]
    p0: two datasets
    p1: kernel size
  v04 : unbiased jitter! [2,50]
    p0: two datasets (GOWT1/p)
    p1: noise_level: noise_level for random jitter
    we want to show performance vs jitter. [curve]. shape. 
  v05 : redo celegans. test training data size. [10,5]:
    p0 : n training images from _early times_
    p1 : repeats ?
  v06 : redo celegans, just the basic training on single images. NO xy SCALING!
    p0 : timepoint to use
    p1 : repeats
  v07 : sampling methods: does it matter what we use? [2,5]
    p0 : [iterative sampling, content sampling]
    p1 : repeats
  v08 : segmentation
    p0 : dataset in 0..18
    p1 : acquisition in 0,1
  """

  (p0,p1),pid = parse_pid(pid,[19,2])
  params = SimpleNamespace(p0=p0,p1=p1)

  savedir_local = savedir / f'e21_isbidet/v08/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  params.info = info

  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)

  """
  EVERYTHING BELOW THIS POINT IS LIKE A WORKSPACE THAT CHANGES RAPIDLY & USES FUNCTIONS DEFINED ABOVE.
  """

  print("Running e21 with savedir: \n", savedir_local, flush=True)

  ## lazy load ZARR files
  def _builddata():
    filenames,pointfiles, = isbiInfo_to_filenames(info)
    def f(a,b):
      raw = load(a)
      lab = load(b)
      return SimpleNamespace(raw=raw,lab=lab)
    imgs = [f(*x) for x in filenames]
    ltps = load(pointfiles)
    for i,d in enumerate(imgs): d.pts = ltps[i]
    return imgs
  data = _builddata()

  ## build train/vali dataloaders
  
  N_vali   =   200 # 20+ it/sec max speed
  N_train  = 1_000 # 4.5 it/sec max speed
  N_epochs = 10
  td,vd = shuffle_and_split(data,valifrac=1/8)

  valiloader = DataLoader(
              dataset=FullDynamicSampler(vd,
                savefile=savedir_local / "mydata_vali.pkl",
                N_samples=N_vali,
                train_mode=0),
              batch_size=None,
              # shuffle=False,
              num_workers=16,
              #os.cpu_count(),
              worker_init_fn=worker_init_fn,
          )

  trainloader = DataLoader(
              dataset=FullDynamicSampler(td,
                savefile=savedir_local / "mydata_train.pkl",
                N_samples=N_train,
                train_mode=1),
              batch_size=None,
              # shuffle=False,
              num_workers=16,
              #os.cpu_count(),
              worker_init_fn=worker_init_fn,
          )
    
  save(compress([x for x in islice(trainloader,10)]), savedir_local/'simple_sampler_train')
  save(compress([x for x in islice(valiloader,10)]), savedir_local/'simple_sampler_vali')

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  def mse_loss(net,sample):
    s  = sample

    x  = torch.from_numpy(s.x).float().to(device,  non_blocking=True)
    yt = torch.from_numpy(s.yt).float().to(device, non_blocking=True)
    w  = torch.from_numpy(s.w).float().to(device,  non_blocking=True)
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss
  net = _init_unet_params(info.ndim).net
  net = net.to(device)

  ## Training
  observer = TrainerObserver()
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)
  
  for i,s in tqdm(enumerate(trainloader),total=N_train,ascii=True,):
    if i==N_train*N_epochs: break
    y,l = mse_loss(net,s)
    l.backward()
    opt.step()
    opt.zero_grad()

    if i%N_train==N_train-1:
      if observer:
        observer.add(i,y,l,net,trainloader,)
      _vl = 0
      with torch.no_grad():
        for j,vs in tqdm(enumerate(valiloader),total=N_vali,ascii=True):
          if j==N_vali: break
          _y,_l = mse_loss(net,vs)
          _vl = _vl + _l
      observer.vali_loss.append(float(_vl/N_vali))


  ## Post-Training
  if trainloader.dataset.savefile: save(trainloader.dataset.reified_samples, trainloader.dataset.savefile)
  if valiloader.dataset.savefile:  save(valiloader.dataset.reified_samples, valiloader.dataset.savefile)

  return observer

if __name__=='__main__':
  myrun([18,0])