from experiments_common import *
from pykdtree.kdtree import KDTree as pyKDTree

from segtools import scores_dense
import numpy as np
from numpy import r_,s_
from segtools import torch_models
import torch 


class BaseModel(object):

  def __init__(self, ndim, savedir, extern):
    self.ndim = ndim
    self._init_params(ndim)
    self.savedir = savedir
    self.extern  = extern

  def _add_train_vali_slices(self):
    ## train/vali for iterative sampling
    slicelist = [(i,ss) for i,d in enumerate(self.data) for ss in d.slices]
    # slicelist = [ss for ss in slicelist if (self._get_patch(ss).yt>0.5).sum()>0]
    # dist = [self._get_patch(ss).yt.sum() for ss in slicelist] #, [1,10,20,30,40,50,60,70,80,90,99]
    # print(dist)
    np.random.seed(0)
    np.random.shuffle(slicelist)
    N = len(slicelist)
    print("N = ", N)
    Nvali  = ceil(N/8)
    Ntrain = N-Nvali
    self.train_slices = slicelist[:Ntrain]
    self.vali_slices  = slicelist[Ntrain:]


class CenterpointModel(object):

  def __init__(self, ndim, savedir, extern):
    self.ndim = ndim
    self._init_params(ndim)
    self.savedir = savedir
    self.extern  = extern

  def load_data(self,filenames):

    def _f(i):
      raw = load(filenames[i][0])
      raw = zoom(raw,self.zoom,order=1)
      raw = normalize3(raw,2,99.4,clip=False)
      lab = load(filenames[i][1])
      lab = zoom(lab,self.zoom,order=0)
      pts = mantrack2pts(lab)
      target = place_gaussian_at_pts(pts,raw.shape,self.kern)
      slices = shape2slicelist(raw.shape,self.patch,divisible=(1,8,8)[-self.ndim:])
      return SimpleNamespace(raw=raw,lab=lab,pts=pts,target=target,slices=slices)

    self.data = [_f(i) for i in range(len(filenames))]
    self._add_train_vali_slices()


  def _add_train_vali_slices(self):
    ## train/vali for iterative sampling
    slicelist = [(i,ss) for i,d in enumerate(self.data) for ss in d.slices]
    # slicelist = [ss for ss in slicelist if (self._get_patch(ss).yt>0.5).sum()>0]
    # dist = [self._get_patch(ss).yt.sum() for ss in slicelist] #, [1,10,20,30,40,50,60,70,80,90,99]
    # print(dist)
    np.random.seed(0)
    np.random.shuffle(slicelist)
    N = len(slicelist)
    print("N = ", N)
    Nvali  = ceil(N/8)
    Ntrain = N-Nvali
    self.train_slices = slicelist[:Ntrain]
    self.vali_slices  = slicelist[Ntrain:]

    ## an index into data must be (i∈len(data),ss∈data[0].shape)
    ## we can make a list of non-overlapping slices, filter out slices based on data content?, split slices into train/vali, all without _actually_ shuffling patches or forgetting their orig spacetime location!

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

  def _get_patch(self,idx):
    i,ss = idx
    s = SimpleNamespace()
    s.x = self.data[i].raw[ss].copy()
    s.yt = self.data[i].target[ss].copy()
    return s

  def _sample_content(self,train_mode=True):
    N = len(self.data)
    # Nvali  = [1,1,1,3,3,4,4][p0] # ceil(N/8) v06
    Nvali  = 4 # v07
    Ntrain = N-Nvali
    idxs = np.r_[:Ntrain] if train_mode else np.r_[Ntrain:N]

    d  = np.random.choice(self.data[n0:n1])
    pt = d.pts[np.random.choice(d.pts.shape[0])]
    ss = jitter_center_inbounds(pt,self.patch,d.raw.shape,jitter=0.1)
    x  = d.raw[ss].copy()
    yt  = d.target[ss].copy()
    return x,yt

  def _sample_iterate(self,time,train_mode=True):
    
    sls = self.train_slices if train_mode else self.vali_slices
    ss  = sls[np.random.choice(len(sls))]
    s   = self._get_patch(ss)
    if train_mode:
      N = len(self.train_slices)
      if time%N==N-1: np.random.shuffle(self.train_slices)
    return s.x, s.yt

  def sample(self,time,train_mode=True):

    # if self.extern.p0==0:
    #   x,yt = self._sample_iterate(time,train_mode)
    # if self.extern.p0==1:
    #   x,yt = self._sample_content(train_mode)

    x,yt = self._sample_iterate(time,train_mode)

    if train_mode:
      x,yt = augment(x,yt)
    if self.extern.info.myname=='fly_isbi':
      w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=0.0)
    else:
      w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=1/5)
      # w = np.ones_like(yt)
    s = SimpleNamespace(x=x,yt=yt,w=w)
    ## prevent overdetect on peaks EVEN WITH FOOTPRINT because they have exactly the same value
    # s.yt_pts = peak_local_max(yt+np.random.rand(*yt.shape)*1e-5,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
    s.yt_pts = peak_local_max(yt,threshold_abs=.2,exclude_border=False,footprint=np.ones(self.nms_footprint))
    return s

  def sampleMax(self,t):
    s = self.sample(t)
    if s.x.ndim==3:
      s.x = s.x.max(0)
      s.yt = s.yt.max(0)
      s.w = s.w.max(0)
    return s

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

  def mse_loss(self,net,sample):
    s  = sample
    x  = torch.from_numpy(s.x).float().cuda()
    yt = torch.from_numpy(s.yt).float().cuda()
    w  = torch.from_numpy(s.w).float().cuda()
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss

  def height(self,y,sample):
    return y.max()

  def match(self,yt_pts,pts):
    scale = [3,1,1][-yt_pts.shape[1]:]
    return match_unambiguous_nearestNeib(yt_pts,pts,dub=100,scale=scale)

  def point_match(self,y,sample):
    s = sample
    pts   = peak_local_max(y,threshold_abs=.2,exclude_border=False,footprint=np.ones(self.nms_footprint))
    scale = [3,1,1][-self.ndim:]
    matching = match_unambiguous_nearestNeib(s.yt_pts,pts,dub=100,scale=scale)
    return matching.f1

  def train_config(self):
    cfig = SimpleNamespace()
    cfig.net = self.net
    cfig.sample = self.sample
    cfig.time_validate = 100
    ## 30_000 for C.Elegans ONLY v07/v06
    cfig.time_total = 10_000 # if info.ndim==3 else 15_000 ## # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr on 3D data (10x faster on 2D?)
    cfig.n_vali_samples = 10
    cfig.save_every_n = 5
    cfig.lr = 4e-4
    cfig.savedir = self.savedir
    cfig.loss = self.mse_loss
    cfig.vali_metrics = [self.height, self.point_match]
    cfig.vali_minmax  = [None,np.max]
    return cfig

  # def optimize_plm(net,data):
  #   def params2score(params):
  #     def f(x):
  #       (raw,gt) = x
  #       # pred = 
  #     avg_score = np.mean([f(x) for x in dataset])
  #     return avg_score
  #   params = optimize(params0,params2score,)
  #   return params


class SegmentationModel(object):

  def __init__(self, ndim, savedir, extern):
    self.ndim = ndim
    self._init_params(ndim)
    self.savedir = savedir
    self.extern  = extern

  def load_data(self,filenames):

    _time,_zpos = re.search(r"man_seg_?(\d{3,4})_?(\d{3,4})?\.tif",filenames[0][1]).groups()
    _d = -2 if _zpos is not None or self.ndim==2 else -3

    def _f(i):
      raw = load(filenames[i][0])
      raw = zoom(raw,self.zoom,order=1)
      raw = normalize3(raw,2,99.4,clip=False)
      lab = load(filenames[i][1])
      lab = zoom(lab,self.zoom[_d:],order=0)
      _time,_zpos = re.search(r"man_seg_?(\d{3,4})_?(\d{3,4})?\.tif",filenames[i][1]).groups()
      target = np.zeros(raw.shape)
      if _zpos:
        _zpos = int(_zpos)
        # print(_zpos)
        target[_zpos,lab>0] = 1
        weights = np.zeros(target.shape)
        weights[_zpos] = 1
      else:
        target[lab>0] = 1
        weights = np.ones(target.shape)
      return SimpleNamespace(raw=raw.astype(np.float16),lab=lab.astype(np.uint16),target=target,weights=weights,)

    self.data = [_f(i) for i in range(len(filenames))]

  def _init_params(self,ndim):
    if ndim==2:
      self.net = torch_models.Unet1(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential).cuda()
      self.zoom  = (1,1) #(0.5,0.5)
      # self.kern  = [5,5]
      self.patch = (512,512)
      # self.nms_footprint = [9,9]
    elif ndim==3:
      self.net = torch_models.Unet1(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential).cuda()
      self.zoom   = (1,1,1) #(1,0.5,0.5)
      # self.kern   = [2,5,5]
      self.patch  = (16,128,128)
      # self.nms_footprint = [3,9,9]
    self.patch = np.array(self.patch)

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
      _patch = np.minimum(self.patch,d.raw.shape)
      pt = (np.random.rand(self.ndim)*(d.raw.shape - _patch)).astype(int)
      ss = tuple([slice(pt[i],pt[i]+_patch[i]) for i in range(len(_patch))])
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

  def mse_loss(self,net,sample):
    s  = sample
    x  = torch.from_numpy(s.x).float().cuda()
    yt = torch.from_numpy(s.yt).float().cuda()
    w  = torch.from_numpy(s.w).float().cuda()
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss

  def height(self,y,sample):
    return y.max()

  # def point_match(self,y,sample):
  #   s = sample
  #   pts   = peak_local_max(y,threshold_abs=.2,exclude_border=False,footprint=np.ones(self.nms_footprint))
  #   score = match(s.yt_pts,pts)
  #   return score.f1

  def train_config(self):
    cfig = SimpleNamespace()
    cfig.net = self.net
    cfig.sample = self.sample
    cfig.time_validate = 100
    ## 30_000 for C.Elegans ONLY v07/v06
    cfig.time_total = 30_000 # if info.ndim==3 else 15_000 ## # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr on 3D data (10x faster on 2D?)
    cfig.n_vali_samples = 10
    cfig.save_every_n = 5
    cfig.lr = 4e-4
    cfig.savedir = self.savedir
    cfig.loss = self.mse_loss
    cfig.vali_metrics = [self.height, self.seg]
    cfig.vali_minmax  = [None, np.max]
    return cfig





