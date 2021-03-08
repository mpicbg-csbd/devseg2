# from experiments_common import *
# from pykdtree.kdtree import KDTree as pyKDTree

from segtools import scores_dense
import numpy as np
from numpy import r_,s_
import torch
import denoise_utils
from enum import Enum,IntEnum
from types import SimpleNamespace

import augmend
from augmend import Augmend, Elastic, FlipRot90
from experiments_common import rchoose, partition, shuffle_and_split
from datagen import shape2slicelist, jitter_center_inbounds, sample_slice_from_volume

from segtools import torch_models
from segtools.ns2dir import load,save
from segtools.numpy_utils import normalize3
from segtools.point_matcher import match_unambiguous_nearestNeib

from scipy.ndimage import zoom,label
from skimage.feature  import peak_local_max

import shutil
from time import time
import ipdb
from tqdm import tqdm



class TrainerObserver(object):
  def __init__(self,):
    self.savedir = None
    # self.time_total = 10_000
    self.losses = []
    self.vali_loss  = []
    self.timings = [time()]
    pass

  def add(self,i,y,l,model,sampler,):
    if i%20==0:
      ts = self.timings
      ts.append(time())
      self.losses.append(float(l.detach().cpu()))
      # tqdm.write(str(ts[-1]-ts[-2]))
      # tqdm.write(str(float(l.detach().cpu())))



class Trainer(object):
  def __init__(self):
    self.lr = 1e-4
    # self.time_total = 10_000
    pass

  def train(self,model,sampler,loss,n_samples=1_000,observer=None):
    
    self.opt = torch.optim.Adam(model.parameters(), lr = self.lr)

    for i,s in tqdm(enumerate(sampler),total=n_samples,ascii=True):
      if i==n_samples: break
      # ipdb.set_trace()
      y,l = loss(model,s)
      l.backward()
      self.opt.step()
      self.opt.zero_grad()
      if observer:
        observer.add(i,y,l,model,sampler,)


class BaseModel(object):

  def __init__(self, savedir):
    self.savedir = savedir
    # self.ndim = ndim
    # self._init_params(ndim)
    # self.extern  = extern
    cfig = SimpleNamespace()
    cfig.time_validate = 100
    cfig.time_total = 10_000 # if info.ndim==3 else 15_000 ## # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr on 3D data (10x faster on 2D?)
    cfig.n_vali_samples = 10
    cfig.save_every_n = 10
    cfig.lr = 4e-4
    cfig.savedir = self.savedir
    cfig.predict_and_compress = self.predict_and_compress
    self.train_cfig = cfig

  def train(self,sampler,_continue=1):
    
    # if _continue==TrainStart.FRESH:
    if _continue==1:
      # print(TrainStart.FRESH)
      print("Starting Fresh...")
      self._setup_dirs()
      self._init_tatdtv()
      # elif _continue==TrainStart.RELOAD:
    elif _continue==2:
      print("Reloading old TA...")
      # print(TrainStart.RELOAD)
      self._reload_tatdtv()

    ta = self.ta
    config = self.train_cfig
    _losses = []

    self.opt = torch.optim.Adam(self.net.parameters(), lr = config.lr)

    for ta.i in range(ta.i,config.time_total+1):

      # s = self.sample(ta.i)
      s = next(sampler)
      y,loss = self.loss(self.net,s) #config.datagen.sample(ta.i,train_mode=1))
      loss.backward()
      y = y.detach().cpu().numpy()

      if ta.i%5==0:
        self.opt.step()
        self.opt.zero_grad()
        _losses.append(float(loss.detach().cpu()))

      if ta.i%20==0:
        ta.timings.append(time())
        dt = 0 if len(ta.timings)==1 else ta.timings[-1]-ta.timings[-2]
        l = float(loss)
        ymax,ystd = float(y.max()), float(y.std())
        ytmax,ytstd = float(s.yt.max()), float(s.yt.std())
        print(f"i={ta.i:04d}, loss={l:4f}, dt={dt:4f}, y={ymax:4f},{ystd:4f} yt={ytmax:4f},{ytstd:4f}",end='\r',flush=True)

      if ta.i % config.time_validate == 0:
        n = ta.i // config.time_validate
        ta.losses.append(np.mean(_losses[-config.time_validate:]))
        save(ta , config.savedir/"ta/")
        self.validate(n)
        self.check_weights_and_save(n)
        if ta.i % (config.time_validate*config.save_every_n) == 0:
          self.predict_and_save_glances(n)

  def _setup_dirs(self):
    if self.savedir.exists(): shutil.rmtree(self.savedir)
    (self.savedir/'m').mkdir(exist_ok=True,parents=True)
    # shutil.copy("/projects/project-broaddus/devseg_2/src/detector2.py",savedir)

  def _reload_tatdtv(self):
    self.ta = load(self.savedir / "ta/")
    self.td = load(self.savedir / "glance_input_td.pkl") #config.sample(0)
    self.vd = load(self.savedir / "glance_input_vd.pkl") #config.sample(0,train_mode=0)

  def _init_tatdtv(self):
    torch_models.init_weights(self.net)
    print("Network params randomized...")
    
    self.td = [self.sample(i,train_mode=1) for i in range(3)]
    self.vd = [self.sample(i,train_mode=0) for i in range(3)]
    save(self.td, self.savedir / "glance_input_td.pkl")
    save(self.vd, self.savedir / "glance_input_vd.pkl")

    config = self.train_cfig
    self.ta = SimpleNamespace(i=1,losses=[],lr=config.lr,save_count=0,vali_loss=[],vali_scores=[],timings=[])
    self.ta.vali_names = [f.__name__ for f in config.vali_metrics]
    self.ta.best_weights_time = np.zeros(len(config.vali_metrics)+1)
    self.predict_and_save_glances(0)

  def validate(self,time):
    
    def f():
      s = self.sample(self.ta.i,train_mode=0)
      with torch.no_grad(): y,loss = self.loss(self.net,s)
      y = y.detach().cpu().numpy()
      loss = loss.cpu().numpy()
      res = [loss] + [f(y,s) for f in self.train_cfig.vali_metrics]
      return res

    allscores = np.stack([f() for _ in range(self.train_cfig.n_vali_samples)])  
    allscores = allscores.mean(0) ## mean over samples
    ta = self.ta
    ta.vali_loss.append(allscores[0])
    ta.vali_scores.append(allscores[1:])

    names  = ["val_loss"] + ta.vali_names
    scores = [ta.vali_loss[-1]] + list(ta.vali_scores[-1])
    print()
    for i in range(len(names)):
      print(f"{names[i]}: {scores[i]:6f}")

  def check_weights_and_save(self,n):

    ta = self.ta
    torch.save(self.net.state_dict(), self.savedir / f'm/best_weights_latest.pt')
    ta.best_weights_latest = n
    
    vl = np.array(ta.vali_loss)
    if vl.min()==vl[-1]:
      ta.best_weights_time[0] = n
      torch.save(self.net.state_dict(), self.savedir / f'm/best_weights_loss.pt')

    vs = np.array(ta.vali_scores)
    for i in range(vs.shape[1]):
      valiname = ta.vali_names[i]
      f_max = ta.vali_minmax[i]
      if f_max is None: continue
      if f_max(vs[:,i])==vs[-1,i]:
        ta.best_weights_time[i] = n
        torch.save(self.net.state_dict(), self.savedir / f'm/best_weights_{valiname}.pt')

  def predict_and_compress(self,sample,time,train=1):
    with torch.no_grad():
      y,l = self.train_cfig.loss(self.net,sample)
    y = y.cpu().numpy()
    l = l.cpu().numpy()
    if self.ndim==3:
      y = y.max(0)
    _dir = "glance_predict_td" if train else "glance_predict_vd"
    save(y.astype(np.float16),self.savedir / _dir / f"yp/a_{time:02d}.npy")

  def predict_and_save_glances(self,time):
    for x in self.td: self.predict_and_compress(x,time,train=1)
    for x in self.vd: self.predict_and_compress(x,time,train=0)

    # tds = [self.predict_and_compress(x) for x in self.td]
    # vds = [self.predict_and_compress(x) for x in self.vd]
    # save(tds,self.savedir / f"glance_predict_td/a{time}.pkl")
    # save(vds,self.savedir / f"glance_predict_vd/a{time}.pkl")

class CenterpointModel(BaseModel):

  def __init__(self, savedir):

    super().__init__(savedir)

    self.loss = self.mse_loss
    def height(y,sample): return y.max()
    self.train_cfig.vali_metrics = [height, self.point_match]
    self.train_cfig.vali_minmax  = [None, np.max]
    self.train_cfig.loss = self.mse_loss

  def _init_params(self,ndim):
    if ndim==2:
      # self.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
      self.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential).cuda()
      self.zoom  = (1,1) #(0.5,0.5)
      self.kern  = [5,5]
      self.patch = (512,512)
      self.nms_footprint = [9,9]
    elif ndim==3:
      self.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential).cuda()
      self.zoom   = (1,1,1) #(1,0.5,0.5)
      self.kern   = [2,5,5]
      self.patch  = (16,128,128)
      self.nms_footprint = [3,9,9]
    self.patch = np.array(self.patch)

  def mse_loss(self,net,sample):
    s  = sample
    x  = torch.from_numpy(s.x).float().cuda()
    yt = torch.from_numpy(s.yt).float().cuda()
    w  = torch.from_numpy(s.w).float().cuda()
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss

  def point_match(self,y,sample):
    s = sample
    pts   = peak_local_max(y,threshold_abs=.2,exclude_border=False,footprint=np.ones(self.nms_footprint))
    scale = [4,1,1][-self.ndim:]
    matching = match_unambiguous_nearestNeib(s.yt_pts,pts,dub=100,scale=scale)
    return matching.f1

  def match(self,yt_pts,pts,dub=100,scale=[4,1,1]):
    scale = [4,1,1][-self.ndim:]
    return match_unambiguous_nearestNeib(yt_pts,pts,dub=dub,scale=scale)

  def predict_and_compress(self,sample,time,train=1):
    with torch.no_grad():
      y,l = self.train_cfig.loss(self.net,sample)
    y = y.cpu().numpy()
    l = l.cpu().numpy()
    pts = peak_local_max(y,threshold_abs=.2,exclude_border=False,footprint=np.ones(self.nms_footprint))
    if self.ndim==3:
      y = y.max(0)
      pts = pts[:,[1,2]]
    # y = (255*y/y.max()).astype(np.uint8)
    _dir = "glance_predict_td" if train else "glance_predict_vd"
    save(y.astype(np.float16),self.savedir / _dir / f"yp/a_{time:02d}.npy")
    save(pts,self.savedir / _dir / f"pts/a_{time:02d}.npy")


  def predict_full(self,raw,dims="YX"):
    assert raw.ndim == self.ndim
    raw = normalize3(raw,2,99.4,clip=False)
    x   = zoom(raw,self.zoom,order=1)
    pred = torch_models.predict_raw(self.net,x,dims=dims).astype(np.float32)
    height = pred.max()
    pred = pred / pred.max() ## 
    pts = peak_local_max(pred,threshold_abs=.2,exclude_border=False,footprint=np.ones(self.nms_footprint))
    pts = pts/self.zoom
    pred = zoom(pred, 1/np.array(self.zoom), order=1)
    return SimpleNamespace(pred=pred,height=height,pts=pts)

class SegmentationModel(BaseModel):

  def __init__(self, savedir):
    super().__init__(savedir)

    self.loss = self.mse_loss
    def height(y,sample): return y.max()
    self.train_cfig.vali_metrics = [height,self.seg]
    self.train_cfig.vali_minmax  = [None,np.max]
    self.train_cfig.loss = self.mse_loss

  def _init_params(self,ndim):
    if ndim==2:
      # self.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
      self.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential).cuda()
      self.zoom  = (1,1) #(0.5,0.5)
      self.patch = (512,512)
    elif ndim==3:
      self.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential).cuda()
      self.zoom   = (1,1,1) #(1,0.5,0.5)
      self.patch  = (16,128,128)
    self.patch = np.array(self.patch)

  def mse_loss(self,net,sample):
    s  = sample
    x  = torch.from_numpy(s.x).float().cuda()
    yt = torch.from_numpy(s.yt).float().cuda()
    w  = torch.from_numpy(s.w).float().cuda()
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss

  @DeprecationWarning
  def sample(self,time,train_mode=True):
    N = len(self.data)
    Nvali  = ceil(N/8)
    Ntrain = N-Nvali
    n0,n1 = (0,Ntrain) if train_mode else (Ntrain,None)
    _data = self.data[n0:n1]
    for _ in range(10):
      _n = np.random.choice(r_[:len(_data)])
      d  = _data[_n]
      # _patch = (np.array(d.raw.shape)-P.patch).clip(min=0)

      x  = d.raw[ss].copy()
      yt = d.target[ss].copy()
      w  = d.weights[ss].copy()
      lab = d.lab[ss].copy()
      # ipdb.set_trace()
      s  = SimpleNamespace(x=x,yt=yt,w=w,ss=ss,pt=pt,lab=lab)
      if s.yt.sum() > 20: break
    s.x,s.yt = augment(s.x,s.yt)
    # print(s.pt,s.ss)
    return s

  def predict_full(self,raw,dims="YX"):
    raw  = normalize3(raw,2,99.4,clip=False)
    x    = zoom(raw,self.zoom,order=1)
    pred = torch_models.predict_raw(self.net,x,dims=dims).astype(np.float32)
    height = pred.max()
    pred = pred / pred.max() ## 
    pred = zoom(pred, 1/np.array(self.zoom), order=1)
    seg  = label(pred>0.5)[0]
    return SimpleNamespace(raw=raw,pred=pred,seg=seg,height=height)

  def seg(self,y,sample):
    lab = label(y>0.5)[0]
    score = scores_dense.seg(sample.lab,lab)
    return score

  def seg_score(self,lab_gt,lab):
    score = scores_dense.seg(lab_gt,lab)
    return score

class StructN2V(BaseModel):

  def __init__(self, savedir, ndim):

    super().__init__(savedir)

    self.ndim = ndim
    self.loss = self.mse_loss
    self.mask = np.zeros((ndim,)*ndim)
    self.mask[(1,)*ndim] = 1

    def height(y,sample): return y.max()
    self.train_cfig.vali_metrics = [height]
    self.train_cfig.vali_minmax  = [None]
    self.train_cfig.loss = self.mse_loss

    self._init_params(ndim)

  def dataloader(self,filenames):

    def f(i):
      raw = load(filenames[i])
      raw = zoom(raw,self.zoom,order=1)
      raw = normalize3(raw,2,99.4,clip=False)
      # slices = shape2slicelist(raw.shape,self.patch,divisible=(1,8,8)[-self.ndim:])
      return SimpleNamespace(raw=raw) #,slices=slices)

    self.data = [f(i) for i in range(len(filenames))]

  def _init_params(self,ndim):
    if ndim==2:
      # self.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
      self.net   = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential).cuda()
      self.zoom  = (1,1)
      self.patch = (512,512)
    elif ndim==3:
      self.net   = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential).cuda()
      self.zoom   = (1,1,1)
      self.patch  = (16,128,128)
    self.patch = np.array(self.patch)

  def sample(self,time,train_mode=True):

    # data = self.data_train if train_mode else self.data_vali
    d  = rchoose(self.data)
    ss = sample_slice_from_volume(self.patch, d.raw.shape)
    x  = d.raw[ss].copy()

    ## TODO: Do we want augmentation in StructN2V? Probably not.
    # if train_mode:
    #   x,yt = self.aug([x,yt])
    # x  = x.copy()
    # yt = yt.copy()

    s = SimpleNamespace(x=x,)
    return s

  def mse_loss(self,net,sample):
    s  = sample
    # x,w = denoise_utils.structN2V_masker(s.x.copy(),self.mask)
    x,w = denoise_utils.nearest_neib_masker(s.x.copy())
    s.w  = w.copy()
    s.yt = s.x.copy()
    s.x  = x.copy()
    x  = torch.from_numpy(x).float().cuda()
    yt = torch.from_numpy(s.x).float().cuda()
    w  = torch.from_numpy(w).float().cuda()
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean()/w.mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss



