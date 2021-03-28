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

import models

from skimage.feature  import peak_local_max
from scipy.ndimage import zoom
from skimage.segmentation import find_boundaries
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

import matplotlib

savedir = savedir_global()
print("savedir:", savedir)

print(f"Total Import Time: {time()-_start_time}")


def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy("/projects/project-broaddus/devseg_2/src/e25_isbi_segment.py", "/projects/project-broaddus/devseg_2/src/temp/e25_isbi_segment_copy.py")
  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 4 -t 2:00:00 --mem 128000 " ## TODO: more cores?
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = 'sbatch -J e25_{pid:03d} {_resources} -o slurm/e25_pid{pid:03d}.out -e slurm/e25_pid{pid:03d}.err --wrap \'python3 -c \"import temp.e25_isbi_segment_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
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


def myrun(pid=0):
  (p0,p1),pid = parse_pid(pid,[19,2])
  savedir_local = savedir / f'e25_isbi_segment/v01/pid{pid:03d}/'
  (savedir_local / 'm').mkdir(exist_ok=True,parents=True) ## place to save models
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  # P = _init_params(info.ndim)
  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)

  print("Running e25 with savedir: \n", savedir_local, flush=True)

  if 0:
    SEGnet = SEGnetISBI(savedir_local, info)
    # save(dg.data[0].target.astype(np.float32), SEGnet.savedir / 'target_t_120.tif')  
    SEGnet.train_cfig.time_total = 30_000
    # SEGnet.net.load_state_dict(torch.load(SEGnet.savedir / "m/best_weights_latest.pt"))
    SEGnet.train(_continue=1)

  SEGnet = SEGnetISBI(savedir_local, info)
  SEGnet.net.load_state_dict(torch.load(SEGnet.savedir / "m/best_weights_seg.pt"))
  segscores = predict_and_eval_seg(SEGnet,info)

  # N = 7
  # gap = floor((info.stop-info.start)/N)
  # predict_times = range(info.start,info.stop,gap)
  # savetimes = predict_times

  # L    = SimpleNamespace(info=info,SEGnet=SEGnet,savetimes=savetimes,predict_times=predict_times,)
  # L.predict_times = range(info.start,info.stop,gap)
  # L.savetimes = predict_times
  # segs = predict_segments(L)



## SEGNet 

def segdata(info):
  labnames = sorted(glob(f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/SEG/*.tif"))
  def f(n_lab):
    _d = info.ndigits
    _time,_zpos = re.search(r"man_seg_?(\d{3,4})_?(\d{3,4})?\.tif",n_lab).groups()
    _time = int(_time)
    if _zpos: _zpos = int(_zpos)
    n_raw = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname.format(time=_time)
    return SimpleNamespace(raw=n_raw,lab=n_lab,time=_time,zpos=_zpos)
  return [f(x) for x in labnames]

def combine_segscores():
  def f(pid):
    try:
      d = list(load(f"/projects/project-broaddus/devseg_2/expr/e25_isbi_segment/v01/pid{pid:03d}/pred_all_seg/segscores.pkl"))
    except:
      d = None
    return d
  scores = [f(pid) for pid in range(19*2)]

  S = SimpleNamespace()
  S.scores = scores
  S.means = [np.mean(x) for x in scores if x is not None]
  S.stds  = [np.std(x) for x in scores if x is not None]
  S.min  = [np.min(x) for x in scores if x is not None]
  S.max  = [np.max(x) for x in scores if x is not None]

  # (p0,p1),pid = parse_pid(pid,[19,2])
  S.data = [isbi_datasets[i//2] for i in range(19*2) if scores[i]is not None]
  S.pids = [i for i in range(19*2) if scores[i] is not None]
  return S


class SEGnetISBI(models.SegmentationModel):

  def __init__(self,savedir,info):
    super().__init__(savedir)
    self._init_params(info.ndim)
    self.train_cfig.net = self.net
    self.train_cfig.sample = self.sample
    self.info = info
    self.ndim = info.ndim
    self.aug = self.augmenter()
    # TODO: cpnet_data_specialization(CPNet,info) ## specialization before dataloader!
    self.dataloader()

  # def _init_params(self,ndim):
  #   if ndim==2:
  #     # self.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
  #     self.net = torch_models.Unet3(4, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sigmoid).cuda()
  #     self.zoom  = (1,1) #(0.5,0.5)
  #     self.patch = (512,512)
  #   elif ndim==3:
  #     self.net = torch_models.Unet3(4, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sigmoid).cuda()
  #     self.zoom   = (1,1,1) #(1,0.5,0.5)
  #     self.patch  = (16,128,128)
  #   self.patch = np.array(self.patch)

  def augmenter(self):
    aug = Augmend()
    ax = {2:(-2,-1), 3:(-2,-1)}[self.ndim]
    # ax_target = {2:(0,1), 3:(1,2)}[self.ndim]
    aug.add([FlipRot90(axis=ax),
             FlipRot90(axis=ax),
             FlipRot90(axis=ax),
             FlipRot90(axis=ax),],
            probability=0.5)
    aug.add([Elastic(axis=ax, amount=5, order=1),
             Elastic(axis=ax, amount=5, order=1),
             Elastic(axis=ax, amount=5, order=1),
             Elastic(axis=ax, amount=5, order=0),],
            probability=0.5)
    aug.add([IntensityScaleShift(), Identity(), Identity(), Identity()], probability=1.0)

    return aug

  def dataloader(self):
    _segdata = segdata(self.info)
    def norm(x): 
      p0,p1 = np.percentile(x,[2,99.4])
      return (x-p0)/(p1-p0)

    def f(s):
      raw = load(s.raw)
      raw = zoom(raw,self.zoom,order=1)
      raw = norm(raw)
      lab = load(s.lab)
      _d  = lab.ndim
      lab = zoom(lab,self.zoom[-_d:],order=0).astype(np.uint16)
      assert lab.sum() > 100
      _bounds = find_boundaries(lab)

      z = s.zpos
      weights = np.ones(raw.shape)
      if z is not None:
        weights = np.zeros(raw.shape)
        weights[z] = 1
        lab2 = np.zeros(raw.shape)
        lab2[z] = lab.copy()
        lab = lab2.astype(np.uint16)
        bounds = np.zeros(raw.shape)
        bounds[z] = _bounds.copy()
        _bounds = bounds

      pts = datagen.mantrack2pts(lab)

      # # target = (lab>0).astype(np.float32)
      # target = np.zeros((3,) + lab.shape) 
      # target[0] = np.where(lab==0,1,0)
      # target[1] = np.where((lab>0) & (_bounds==0),1,0)
      # target[2] = np.where(_bounds==1,1,0)

      # target = (lab>0).astype(np.float32)
      target = lab.copy()
      target[lab>0] = 1
      target[_bounds] = 2
      target = target.astype(np.uint8)

      # ipdb.set_trace()
      # slices = datagen.patches_from_centerpoints(img, pts, _patchsize)
      return SimpleNamespace(raw=raw,lab=lab,target=target,weights=weights,pts=pts)

    _segdata = _segdata[::2] ## remove test data
    N = floor(len(_segdata)*7/8) ## t/v split
    self.data_train = [f(s) for s in _segdata[:N]]
    self.data_vali  = [f(s) for s in _segdata[N:]]

    def _g(x):
      x0 = (x==0).sum()
      x1 = (x==1).sum()
      x2 = (x==2).sum()
      return x0,x1,x2

    _ws = np.sum([_g(s.target) for s in self.data_train],axis=0)
    _ws = _ws / _ws.sum()
    self.ce_weights = 1/_ws

    save(self.data_train, self.savedir / 'data_train.pkl')
    save(self.data_vali, self.savedir / 'data_vali.pkl')

  def sample(self,time,train_mode=True):

    data = self.data_train if train_mode else self.data_vali
    d    = data[np.random.choice(len(data))] ## choose time
    pt   = d.pts[np.random.choice(d.pts.shape[0])] ## choose point
    ss   = datagen.jitter_center_inbounds(pt,self.patch,d.raw.shape,jitter=0.1)
    
    # _all = (slice(None),) + ss
    x  = d.raw[ss].copy()
    yt = d.target[ss].copy()
    w  = d.weights[ss].copy()
    l  = d.lab[ss].copy()

    if train_mode:
      # ipdb.set_trace()
      x,yt,w,l = self.aug([x,yt,w,l])
      l = np.round(l).astype(int)

    x  = x.copy()
    yt = yt.copy()
    w  = w.copy()
    l  = l.copy()

    # ipdb.set_trace()
    s = SimpleNamespace(x=x,yt=yt,w=w,lab=l)
    return s

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



def divide_evenly_with_min1(n_samples,n_bins):
  N = n_samples
  M = n_bins
  assert N>=M
  y = np.linspace(0,N,M+1).astype(np.int)
  ss = [slice(y[i],y[i+1]) for i in range(M)]
  return ss

def predict_and_eval_seg(SEGnet,info):
  savedir = SEGnet.savedir / "pred_all_seg"
  if savedir.exists():
    shutil.rmtree(savedir)
    savedir.mkdir()
  info = info
  dims = "ZYX" if info.ndim==3 else "YX"
  _segdata = segdata(info)
  def _save_ids():
    N = len(_segdata)
    return np.linspace(0,N-1,min(5,N)).astype(np.int)
  save_ids = _save_ids()
  print(save_ids)

  # ipdb.set_trace()

  def _single(i):
    s = _segdata[i]
    print(s.time, s.zpos)
    raw  = load(s.raw)
    lab_gt  = load(s.lab)
    res  = SEGnet.predict_full(raw,dims)
    _seg = res.seg[s.zpos] if s.zpos is not None else res.seg
    res.segscore = SEGnet.seg_score(lab_gt,_seg)
    print(res.segscore)
    # ipdb.set_trace()
    res.lab_gt = lab_gt
    return res

  def _save_preds(d,i):
    save(_png(d.raw), savedir/f"d{i:04d}/raw.png")
    # ipdb.set_trace()
    save(_png(d.pred[0]), savedir/f"d{i:04d}/pred0.png")
    save(_png(d.pred[1]), savedir/f"d{i:04d}/pred1.png")
    save(_png(d.pred[2]), savedir/f"d{i:04d}/pred2.png")
    save(_png(d.seg), savedir/f"d{i:04d}/seg.png")
    save(_png(d.lab_gt), savedir/f"d{i:04d}/lab_gt.png")

  def _f(i): ## i is time
    d = _single(i)
    if i in save_ids: _save_preds(d,i)
    return [d.segscore]

  segscores = list(map(list,zip(*[_f(i) for i in range(len(_segdata))])))
  save(segscores, savedir/"segscores.pkl")
  return segscores


def predict_segments(L):
  # gtpts = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
  savedir = L.SEGnet.savedir / "pred_all_01"
  info = L.info
  dims = "ZYX" if info.ndim==3 else "YX"

  def _single(i):
    "i is time"
    print(i)
    name = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname.format(time=i)
    raw  = load(name)
    res  = L.SEGnet.predict_full(raw,dims)
    return res

  def _save_preds(d,i):
    save(_png(d.raw), savedir/f"d{i:04d}/raw.png")
    save(_png(d.pred), savedir/f"d{i:04d}/pred.png")
    save(_png(d.seg), savedir/f"d{i:04d}/seg.png")

  def _f(i): ## i is time
    d = _single(i)
    if i in L.savetimes: _save_preds(d,i)
    return d.seg

  segs = map(list,zip(*[_f(i) for i in L.predict_times]))
  return segs


if __name__=='__main__':
  for i in range(25,19*2): 
    try: 
      myrun(i)
    except:
      print("Error in pid ",i)


