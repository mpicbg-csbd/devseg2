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
from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale
from segtools import torch_models
from tqdm import tqdm
from segtools.point_matcher import match_unambiguous_nearestNeib
from segtools.point_tools import trim_images_from_pts2



savedir = savedir_global()
print("savedir:", savedir)

print(f"Total Import Time: {time()-_start_time}")


def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy("/projects/project-broaddus/devseg_2/src/e21_isbidet.py", "/projects/project-broaddus/devseg_2/src/e21_isbidet_copy.py")
  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 4 -t 2:00:00 --mem 128000 "
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



def _init_params(ndim):
  P = SimpleNamespace()
  if ndim==2:
    P.zoom  = (1,1) #(0.5,0.5)
    P.kern  = [5,5]
    P.patch = (512,512)
  elif ndim==3:
    P.zoom   = (1,1,1) #(1,0.5,0.5)
    P.kern   = [2,5,5]
    P.patch  = (16,128,128)
  P.patch = np.array(P.patch)
  return P

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
  # if info.index in [6,11,12,13,14,15,18]:
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
    # self.timeslice  = slice(0,len(data))
    self.train_mode = train_mode
    self.savefile   = savefile
    if savefile is None or not Path(savefile).exists():
      self.reified_samples = []
    else:
      self.reified_samples = load(savefile)
      print(f"Savefile Loaded with {len(self.reified_samples)} samples...")
    self.data_resized = False


  def __iter__(self):
    self.ndim = self.data[0].raw.ndim
    P = _init_params(self.ndim)
    for k,v in P.__dict__.items(): self.__dict__[k] = v
    self.counter = 0
    self.f_augment = self.augmenter(self.ndim)

    while True:
      if self.N_samples and len(self.reified_samples) >= self.N_samples:
        s = self.reified_samples[self.counter % self.N_samples]
        if self.counter % self.N_samples == -1 % self.N_samples:
          np.random.shuffle(self.reified_samples)
      else:
        s = self.sample(self.counter)
        if self.N_samples: self.reified_samples.append(s)
      yield s

      self.counter += 1

  def resize_data(self):

    def resize(datum):
      d = datum
      pts2,ss = trim_images_from_pts2(d.pts,border=(10,10,10))
      d2 = SimpleNamespace()
      d2.pts = pts2
      d2.raw = d.raw[ss]
      d2.lab = d.lab[ss]
      return d2

    if np.prod(self.data[0].raw.shape)/1e9 > 1:
      self.data = [resize(d) for d in self.data]
    self.data_resized = True


  def sample(self,time):
    # while 1: 
    d = rchoose(self.data) #[self.timeslice])
    # if d is not None: break

    # t = ((time//40)*3)%len(self.data)
    # d   = self.data[t]

    pts = d.pts
    pt  = rchoose(pts)
    _patchsize  = (self.patch / self.zoom).astype(int)
    ss  = datagen.jitter_center_inbounds(pt,_patchsize,d.raw.shape,jitter=0.1)

    raw,lab = d.raw[ss].copy(), d.lab[ss].copy()
    p2,p99  = np.percentile(raw,[2,99.4]) ## for speed
    raw     = (raw-p2)/(p99-p2)

    if self.train_mode:
      raw,lab = self.f_augment([raw,lab])

    raw = zoom(raw,self.zoom,order=1)
    # raw = gputools.scale(raw,self.zoom,interpolation='linear')
    # p2,p99 = np.percentile(raw[::4,::4,::4],[2,99.4]) ## for speed
    pts     = datagen.mantrack2pts(lab)
    pts     = (np.array(pts) * self.zoom).astype(np.int)
    target  = datagen.place_gaussian_at_pts(pts,raw.shape,self.kern)
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
    # aug.add([Rotate(axis=ax, order=1),
    #          Rotate(axis=ax, order=1),],
    #         probability=0.5)
    return aug

def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset  # the dataset copy in this worker process 
  N = len(dataset.data) 
  M = worker_info.num_workers
  a = ceil(N/M)
  _id = worker_info.id
  s = slice(_id*a,(_id+1)*a)
  # dataset.timeslice = s
  dataset.data = dataset.data[s]
  print(f"Hello From Worker {_id}!!!")
  print("Data Shape: ", len(dataset.data))
  # _t1 = time()
  # print("Length of data: ", len(dataset.data))
  # dataset.resize_data()
  # print("Initializing Data...", time()-_t1, " sec")

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


def dataloader(data,savefile=None,N_samples=100,num_workers=0,train_mode=1):
  return DataLoader(
              dataset=FullDynamicSampler(data,
                savefile=savefile,
                N_samples=N_samples,
                train_mode=train_mode),
              batch_size=None,
              # shuffle=False,
              num_workers=num_workers,
              #os.cpu_count(),
              worker_init_fn=worker_init_fn,
              # persistent_workers=True, ## doesn't exist in pytorch 1.2
          )
  


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
  v09 : total system refactor
    p0 : dataset in 0..18
    p1 : acquisition in 0,1    
  """

  (p0,p1),pid = parse_pid(pid,[19,2])
  savedir_local = savedir / f'e21_isbidet/v08/pid{pid:03d}/'
  (savedir_local / 'm').mkdir(exist_ok=True,parents=True) ## place to save models
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  P = _init_params(info.ndim)
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
    for i,d in enumerate(imgs): d.pts = ltps[i+info.start]
    return imgs
  data = _builddata()
  # data = data[::3]



  np.random.seed(0)
  td,vd    = shuffle_and_split(data,valifrac=1/8)
  max_total_samples = len(data)*np.prod(info.shape) / np.prod(P.patch)
  valiloader  = dataloader(vd,savefile="mydata_train.pkl",
                            N_samples=min(max_total_samples//8, 500),
                            num_workers=8,train_mode=0)
  trainloader = dataloader(td,savefile="mydata_vali.pkl",
                            N_samples=min(max_total_samples//8*7, 1000),
                            num_workers=8,train_mode=1)

  print("Sizes valiloader: ", max_total_samples//8//4)
  print("Sizes trainloader: ", max_total_samples//8*7//4)
  
  glance_train = [x for x in islice(trainloader,3)]
  glance_vali  = [x for x in islice(valiloader,3)]
  save(compress(glance_train), savedir_local/'glance_input_train')
  save(compress(glance_vali), savedir_local/'glance_input_vali')

  def mse_loss(net,sample):
    s  = sample
    x  = torch.from_numpy(s.x).float().to(device,  non_blocking=True)
    yt = torch.from_numpy(s.yt).float().to(device, non_blocking=True)
    w  = torch.from_numpy(s.w).float().to(device,  non_blocking=True)
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss

  def validate(net,sample):
    s = sample
    with torch.no_grad(): y,l = mse_loss(net,s)
    y = y.cpu().numpy()
    l = l.cpu().numpy()
    nms_footprint = [3,7,7][-info.ndim:]
    pts   = peak_local_max(y/y.max(),threshold_abs=.2,exclude_border=False,footprint=np.ones(nms_footprint))
    scale = [4,1,1][-info.ndim:]
    matching = match_unambiguous_nearestNeib(s.yt_pts,pts,dub=100,scale=scale)
    return y, SimpleNamespace(loss=l,f1=matching.f1,height=y.max())

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

  history = train(net,mse_loss,validate,trainloader,valiloader,N_vali=10,N_train=100,N_epochs=100,observer=None,pred_glances=pred_glances,savedir=savedir_local)
  post_train(history,trainloader,valiloader,savedir_local)
  ## Training
  return history



def norm_minmax01(x):
  return (x-x.min())/(x.max()-x.min())

def train(net,f_loss,f_vali,trainloader,valiloader,N_vali=20,N_train=100,N_epochs=3,observer=None,pred_glances=None,savedir=None):
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)
  if observer is None: observer = SimpleNamespace()
  observer.lossmeans = []
  valikeys   = ['loss','f1','height']
  valiinvert = [1,-1,-1] # minimize, maximize, maximize
  observer.valimeans = dict(loss=[],f1=[],height=[])
  
  _loss_temp = []
  for i,s in tqdm(enumerate(trainloader),total=N_train*N_epochs,ascii=True,):
    if i==N_train*N_epochs: break
    y,l = f_loss(net,s)
    l.backward()
    opt.step()
    opt.zero_grad()
    _loss_temp.append(float(l.detach().cpu()))

    if i%N_train==N_train-1:
      valis = []
      for j,vs in tqdm(enumerate(valiloader),total=N_vali,ascii=True):
        if j==N_vali: break
        valis.append(f_vali(net,vs)[1])
      if pred_glances and (i//N_train)%3==0: pred_glances(net,i//N_train)
      current_valimeans = {k:np.mean([sn.__dict__[k] for sn in valis]) for k in valikeys}
      torch.save(net.state_dict(), savedir / f'm/best_weights_latest.pt')
      for i,k in enumerate(valikeys):
        if len(observer.valimeans[k])==0 or current_valimeans[k]*valiinvert[i] < np.min(observer.valimeans[k]*np.array(valiinvert[i])):
          torch.save(net.state_dict(), savedir / f'm/best_weights_{k}.pt')
        observer.valimeans[k].append(current_valimeans[k])
      observer.lossmeans.append(np.mean(_loss_temp))
      save(observer,savedir / "history.pkl")
      _loss_temp = []

  return observer

def post_train(history,trainloader,valiloader,savedir_local):
  ## Post-Training
  save(history,savedir_local / "history.pkl")
  save(trainloader.dataset.reified_samples[:1000], savedir_local / "mydata_train.pkl")
  save(valiloader.dataset.reified_samples[:1000], savedir_local / "mydata_vali.pkl")



if __name__=='__main__':
  myrun([18,0])