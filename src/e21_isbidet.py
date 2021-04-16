"""
Train CP-Net to detect cells and nuclei in one of the 19*2 ISBI CTC datasets.
"""

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


savedir = savedir_global()
print("savedir:", savedir)

print(f"Total Import Time: {time()-_start_time}")


def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy("/projects/project-broaddus/devseg_2/src/e21_isbidet.py", "/projects/project-broaddus/devseg_2/src/temp/e21_isbidet_copy.py")
  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 4 -t 2:00:00 --mem 128000 "    ## TODO: more cores?
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = 'sbatch -J e21_{pid:03d} {_resources} -o slurm/e21_pid{pid:03d}.out -e slurm/e21_pid{pid:03d}.err --wrap \'python3 -c \"import temp.e21_isbidet_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
  slurm = slurm.replace("{_resources}",_gpu)
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)

def myrun_slurm_entry(pid=0):
  load_trainingdata(pid)

  # (p0,p1),pid = parse_pid(pid,[2,19,9])
  # load_trainingdata([p0,p1,p2])
  # for p2,p3 in iterdims([2,5]):
  #   try:
  #   except:
  #     print("FAIL on pids", [p0,p1,p2,p3])

def _init_unet_params(ndim):
  T = SimpleNamespace()
  if ndim==2:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
  elif ndim==3:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
  return T


# def isbiInfo_to_filenames2(info):
#   n_raw  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
#   n_lab  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track
#   # if info.index in [6,11,12,13,14,15,18]:
#   n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
#   n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
#   filenames_raw = {i:(n_raw.format(time=i),n_lab.format(time=i)) for i in range(info.start,info.stop)}
#   filename_ltps = f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl"
#   return filenames_raw, filename_ltps

# def dataset_description(info):
#   filenames_raw, filename_ltps = isbiInfo_to_filenames2(info)
#   ltps = load(filename_ltps)
#   def f(i):
#     slicelist = 
#   slices = []




## useful for turning images into pngs
import matplotlib
cmap = np.random.rand(256,3).clip(min=0.1)
cmap[0] = (0,0,0)
cmap = matplotlib.colors.ListedColormap(cmap)
def colorseg(seg):
  m = seg!=0
  seg[m] %= 254 ## we need to save a color for black==0
  seg[m] += 1
  rgb = cmap(seg)
  return rgb
def _png(x):
  if x.ndim==3:
    x = x.max(0)
  if 'int' in str(x.dtype):
    x = colorseg(x)
  else:
    norm = lambda x: (x-x.min())/(x.max()-x.min())
    x = norm(x)
  x = (x*255).astype(np.uint8)
  return x

def norm_minmax01(x):
  mx = x.max()
  mn = x.min()
  if mx==mn: 
    return x-mx
  else: 
    return (x-mn)/(mx-mn)

def save_input_glances(sample_list,filename):
  s = SimpleNamespace()
  _s = s.__dict__
  for i,x in enumerate(sample_list):
    for k in ['x', 'yt', 'w',]: # 'yt_pts',]:
      v = x.__dict__[k]
      if v.ndim==3: v = v.max(0)
      assert v.ndim==2
      if k=='x':
        _s[f'x{i}']  = (norm_minmax01(v)*255).astype(np.uint8)
      if k=='yt':
        _s[f'yt{i}'] = (norm_minmax01(v)*255).astype(np.uint8)
  save(s,filename)

def pid2params(pid):
  (p0,p1),pid = parse_pid(pid,[19,2])
  # savedir_local = savedir / f'e25_isbi_segment/v02/pid{pid:03d}/'
  savedir_local = savedir / f'e21_isbidet/v09/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)
  return SimpleNamespace(**locals())



## Functions below are necessary for generating training data... 
## seems like too much work. 
## 200 lines???? that's just too much...

def cpnet_data_specialization(info):
  myname = info.myname
  p = SimpleNamespace()
  if myname in ["celegans_isbi","A549","A549-SIM","H157","hampster","Fluo-N3DH-SIM+"]:
    p.zoom = {3:(1,0.5,0.5), 2:(0.5,0.5)}[info.ndim]
  if myname=="trib_isbi":
    p.kern = [3,3,3]
    p.zoom = (0.5,0.5,0.5)
  if myname=="MSC":
    a,b = info.shape
    p.zoom = {'01':(1/4,1/4), '02':(128/a, 200/b)}[info.dataset]
    ## '02' rescaling is almost exactly isotropic while still being divisible by 8.
  if info.isbiname=="DIC-C2DH-HeLa":
    # p.kern = [7,7]
    p.zoom = (0.5,0.5)
  if myname=="fly_isbi":
    p.sparsity = 2
    # cfig.bg_weight_multiplier=0.0
    # cfig.weight_decay = False
  if "trib" in myname:
    p.sparsity = 1
  return p

def _method_params(ndim):
  P = SimpleNamespace()
  if ndim==2:
    P.zoom  = (1,1) #(0.5,0.5)
    P.kern  = [5,5]
    P.patch = (512,512)
  elif ndim==3:
    P.zoom   = (1,1,1) #(1,0.5,0.5)
    P.kern   = [2,5,5]
    P.patch  = (16,128,128)
  P.nms_footprint = P.kern
  P.patch = np.array(P.patch)
  return P

def isbiInfo_to_filenames(info):
  n_raw  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
  n_lab  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track
  # if info.index in [6,11,12,13,14,15,18]:
  n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  filenames_raw = [(n_raw.format(time=i),n_lab.format(time=i)) for i in range(info.start,info.stop)]
  filename_ltps = f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl"
  return filenames_raw, filename_ltps

class FullDynamicSampler(IterableDataset):
  def __init__(self,info,data,train_mode=1,N_samples=1_000,savefile=None):
    super().__init__()

    self.info = info
    self.sparse = True if info.isbiname in ["Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF",] else False

    # if len(data)>30:

    self.data = data #[f(a,b) for a,b in self.filenames][self.timeslice]

    # self.ltps = ltps #load(self.pointfiles)[self.timeslice] if self.pointfiles else None
    self.N_samples = N_samples
    # self.timeslice  = slice(0,len(data))
    self.train_mode = train_mode
    self.savefile   = savefile
    if savefile is None or not Path(savefile).exists():
      self.reified_samples = []
      self.savedyet = False
    else:
      self.reified_samples = load(savefile)
      self.savedyet = True
      print(f"Savefile Loaded with {len(self.reified_samples)} samples...")
    self.data_resized = False
    self.ndim = self.data[0].raw.ndim
    P = _method_params(self.ndim)
    for k,v in P.__dict__.items(): self.__dict__[k] = v
    self.f_augment = self.augmenter(self.ndim)
    self.counter = 0

  def __iter__(self):
    while True:

      if self.N_samples and len(self.reified_samples) >= self.N_samples:
        if self.savedyet == False: save(self.reified_samples,self.savefile)
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

    # ipdb.set_trace()

    _patchsize  = (self.patch / self.zoom).astype(int)
    if self.sparse:
      pt  = rchoose(d.pts)
    else:
      pt  = np.random.rand(self.ndim)*(d.raw.shape - _patchsize)
    
    if time%100==0: print(pt, d.raw.shape, _patchsize,flush=True)
    ss  = datagen.jitter_center_inbounds(pt,_patchsize,d.raw.shape,jitter=0.1)

    raw,lab = d.raw[ss].copy(), d.lab[ss].copy()
    p2,p99  = np.percentile(raw,[2,99.4]) ## for speed
    raw     = (raw-p2)/(p99-p2)

    if self.train_mode:
      raw,lab = self.f_augment([raw,lab])

    raw = zoom(raw,self.zoom,order=1)
    # raw = gputools.scale(raw,self.zoom,interpolation='linear')
    # p2,p99 = np.percentile(raw[::4,::4,::4],[2,99.4]) ## for speed
    pts       = np.array(datagen.mantrack2pts(lab)).astype(np.int)
    if len(pts)>0:
      pts     = (pts * self.zoom).astype(np.int)
      target  = datagen.place_gaussian_at_pts(pts,raw.shape,self.kern)
    else:
      target  = np.zeros(raw.shape)
    
    if self.sparse:
      weights = weights__decaying_bg_multiplier(target,0,thresh=np.exp(-0.5*(1/3)**2),decayTime=None,bg_weight_multiplier=0.0)
    else:
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

    aug.add([augmend.IntensityScaleShift(), augmend.Identity()], probability=1.0)

    # aug.add([Rotate(axis=ax, order=1),
    #          Rotate(axis=ax, order=1),],
    #         probability=0.5)
    return aug

def divide_evenly_with_min1(n_samples,n_bins):
  N = n_samples
  M = n_bins
  assert N>=M, f"The problem is N {N}, M {M}..."
  y = np.linspace(0,N,M+1).astype(np.int)
  ss = [slice(y[i],y[i+1]) for i in range(M)]
  return ss

def dataloader(info,data,savefile=None,N_samples=100,num_workers=0,train_mode=1):
  return DataLoader(
              dataset=FullDynamicSampler(info,data,
                savefile=savefile,
                N_samples=int(N_samples),
                train_mode=train_mode),
              batch_size=None,
              # shuffle=False,
              num_workers=int(num_workers),
              #os.cpu_count(),
              worker_init_fn=worker_init_fn,
              # persistent_workers=True, ## doesn't exist in pytorch 1.2
          )

def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset  # the dataset copy in this worker process 
  N = len(dataset.data) 
  M = worker_info.num_workers
  _id = int(worker_info.id)
  print('l360',_id,N,M)
  ss = divide_evenly_with_min1(N,M)
  s = ss[_id]
  # print(s)
  # ipdb.set_trace()
  dataset.data = dataset.data[s]
  print(f"Worker {_id}, start {s.start}, stop {s.stop}")
  # print("Data Shape: ", len(dataset.data))

  # _t1 = time()
  # print("Length of data: ", len(dataset.data))
  # dataset.resize_data()
  # print("Initializing Data...", time()-_t1, " sec")
  # print(locals())

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
  v10 : train on entire acquisition: test on 2nd acquisition.
    p0 : dataset in 0..18
    p1 : acquisition in 0,1
  """
  P = pid2params(pid)
  info = P.info
  T = _method_params(info.ndim)
  (P.savedir_local / 'm').mkdir(exist_ok=True,parents=True) ## place to save models
  """
  EVERYTHING BELOW THIS POINT IS LIKE A WORKSPACE THAT CHANGES RAPIDLY & USES FUNCTIONS DEFINED ABOVE.
  """

  tic = time()

  print("Running e21 with savedir: \n", P.savedir_local, flush=True)

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
  np.random.seed(0)

  ## in this form data can be shuffled and permuted. or stripped to leave out test data.
  N = max(6,len(data)//20); gap=len(data)//N
  data = data[::gap]
  ## heuristics to determine train/vali split and the number of patches to save / iterate over.
  td,vd = shuffle_and_split(data,valifrac=1/8)
  print('Line440',len(td),len(vd))

  nw = floor(len(vd)/4)
  valiloader  = dataloader(info,vd,savefile="mydata_train.pkl", N_samples=200, num_workers=0, train_mode=0,)
  nw = floor(len(td)/4)
  trainloader = dataloader(info,td,savefile="mydata_vali.pkl", N_samples=1000, num_workers=0, train_mode=1,)

  # max_total_samples = ceil(len(data)*np.prod(info.shape) / np.prod(T.patch))
  # ns = min(max_total_samples//8, 500)    # num samples to loop over
  # nw = np.clip(len(vd)//7,a_min=0,a_max=16) # num workers
  # nw = 2
  # print(f"Vali: ns {ns}, nw {nw}")  
  
  # ns = min(max_total_samples//8*7, 1000) # num samples to loop over
  # nw = np.clip(len(td)//7,a_min=0,a_max=16)   # num workers
  # nw = 2
  # print(f"Train: ns {ns}, nw {nw}")


  _dsp = cpnet_data_specialization(info) # data-specific params

  for k,v in _dsp.__dict__.items():
    trainloader.dataset.__dict__[k] = v
    valiloader.dataset.__dict__[k]  = v
  _prediction_params = SimpleNamespace(zoom=trainloader.dataset.zoom,nms=trainloader.dataset.nms_footprint,kern=trainloader.dataset.kern)
  save(_prediction_params, P.savedir_local/"data_specific_params.pkl")
  
  glance_train = [x for x in islice(trainloader,3)]
  glance_vali  = [x for x in islice(valiloader,3)]
  save(glance_train, P.savedir_local/'glance_train.pkl')
  save(glance_vali,  P.savedir_local/'glance_vali.pkl')

  save_input_glances(glance_train,P.savedir_local/'glance_input_train')
  save_input_glances(glance_vali, P.savedir_local/'glance_input_vali')

  # save(trainloader.dataset, P.savedir_local/'trainloader.pkl')
  # save(valiloader.dataset, P.savedir_local/'valiloader.pkl')

  # P = pid2params(pid)
  # info = P.info
  # T = _method_params(info.ndim)
  # (P.savedir_local / 'm').mkdir(exist_ok=True,parents=True) ## place to save models

  ## load data from disk
  if False:
    trainloader  = dataloader(load(P.savedir_local/'trainloader.pkl'))
    valiloader   = dataloader(load(P.savedir_local/'valiloader.pkl'))
    glance_train = load(P.savedir_local/'glance_input_train')
    glance_vali  = load(P.savedir_local/'glance_input_vali')


  toc = time()
  print("TIME dataloader:", toc-tic)

  train(P,trainloader,valiloader,glance_train,glance_vali,)




def train(params,trainloader,valiloader,glance_train,glance_vali,):

  tic = time()
  P = params

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
    nms_footprint = valiloader.dataset.nms_footprint
    pts   = peak_local_max(y/y.max(),threshold_abs=.2,exclude_border=False,footprint=np.ones(nms_footprint))
    scale = np.array(P.info.scale)
    if P.info.ndim==3: scale = scale*(0.5,1,1) ## to account for low-resolution and noise along z dimension
    matching = match_unambiguous_nearestNeib(s.yt_pts,pts,dub=100,scale=scale)
    # return y, SimpleNamespace(loss=l,f1=matching.f1,height=y.max())
    return y, (l,matching.f1,y.max())

  def pred_glances(net,time):
    for i,s in enumerate(glance_train):
      with torch.no_grad(): y,l = mse_loss(net,s)
      y = y.cpu().numpy()
      l = l.cpu().numpy()
      if y.ndim==3: y = y.max(0)
      y = (norm_minmax01(y)*255).astype(np.uint8)
      save(y,P.savedir_local/f'glance_output_train/a{time}_{i}.png')

    for i,s in enumerate(glance_vali):
      with torch.no_grad(): y,l = mse_loss(net,s)
      y = y.cpu().numpy()
      l = l.cpu().numpy()
      if y.ndim==3: y = y.max(0)
      y = (norm_minmax01(y)*255).astype(np.uint8)
      save(y,P.savedir_local/f'glance_output_vali/a{time}_{i}.png')

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = _init_unet_params(P.info.ndim).net
  net = net.to(device)
  # net.load_state_dict(torch.load(P.savedir_local / f'm/best_weights_latest.pt'))

  toc = time()
  print("TIME init model:", toc-tic)
  history = train_net(net,mse_loss,validate,trainloader,valiloader,N_vali=10,N_train=100,N_epochs=100,observer=None,pred_glances=pred_glances,savedir=P.savedir_local)
  post_train(history,trainloader,valiloader,P.savedir_local)
  ## Training
  return history





def norm_minmax01(x):
  return (x-x.min())/(x.max()-x.min())

def train_net(net,f_loss,f_vali,trainloader,valiloader,N_vali=20,N_train=100,N_epochs=3,observer=None,pred_glances=None,savedir=None):
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)
  if observer is None: 
    observer = SimpleNamespace()
    observer.lossmeans = []
    observer.valimeans = []

  tic = time()

  trainloader = iter(trainloader)
  valiloader = iter(valiloader)

  for i in range(N_epochs):
    loss = backprop_n_samples_into_net(net,opt,f_loss,trainloader,N_train)
    observer.lossmeans.append(loss)
    vali = validate_me_baby(net,f_vali,valiloader,N_vali)
    observer.valimeans.append(vali)
    save(observer,savedir / "history.pkl")
    pred_glances(net,i)
    save_best_weights(net,vali,savedir)
    
    dt  = time() - tic
    tic = time()
    print(f"epoch {i}/{N_epochs}, loss={observer.lossmeans[-1]:4f}, dt={dt:4f}, rate={N_train/dt:5f} samples/s", end='\r',flush=True)

  return observer

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
    s = next(valiloader)
    valis.append(f_vali(net,s)[1])
  return np.nanmean(valis,0)

def backprop_n_samples_into_net(net,opt,f_loss,trainloader,nsamples):
  _losses = []
  tic = time()
  print()
  for i in range(nsamples):
    s = next(trainloader)
    y,l = f_loss(net,s)
    l.backward()
    opt.step()
    opt.zero_grad()
    _losses.append(float(l.detach().cpu()))
    dt = time()-tic; tic = time()
    print(f"it {i}/{nsamples}, dt {dt:5f}, max {float(y.max()):5f}", end='\r',flush=True)
  print("\033[F",end='')
  return np.mean(_losses)




def post_train(history,trainloader,valiloader,savedir_local):
  ## Post-Training
  save(history,savedir_local / "history.pkl")
  save(trainloader.dataset.reified_samples[:1000], savedir_local / "mydata_train.pkl")
  save(valiloader.dataset.reified_samples[:1000], savedir_local / "mydata_vali.pkl")



if __name__=='__main__':
  for i in range(19*2): load_trainingdata(i)