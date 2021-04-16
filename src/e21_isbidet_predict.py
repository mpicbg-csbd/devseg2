from segtools.point_matcher import match_unambiguous_nearestNeib

from experiments_common import rchoose, partition, shuffle_and_split, parse_pid, iterdims, savedir_global
from datagen import place_gaussian_at_pts

# from segtools.render import rgb_max
# from models import CenterpointModel, SegmentationModel, StructN2V
# from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale
# from tracking import nn_tracking_on_ltps, random_tracking_on_ltps
from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
from segtools.numpy_utils import normalize3, perm2, collapse2, splt, plotgrid
from segtools.ns2dir import load,save
from segtools import torch_models

import shutil
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
from pathlib import Path

import warnings
warnings.simplefilter("once")

try:
    import gputools
except ImportError as e:
    print("Can't import gputools on non-gpu node...\n", e)

import ipdb
from time import time

# from e24_isbi_datagen import TrainyDataThingy, FullImageNormSampler, FullDynamicSampler


savedir = savedir_global()
print("savedir:", savedir)


def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy("/projects/project-broaddus/devseg_2/src/e21_isbidet.py", "/projects/project-broaddus/devseg_2/src/e21_isbidet_copy.py")
  _gpu  = "-p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 "
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = 'sbatch -J e21_{pid:03d} {_resources} -o slurm/e21_seg_pid{pid:03d}.out -e slurm/e21_seg_pid{pid:03d}.err --wrap \'python3 -c \"import e21_isbidet_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
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

## Program start here

def _init_unet_params(ndim):
  T = SimpleNamespace()
  if ndim==2:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
    T.nms_footprint = [9,9]
  elif ndim==3:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
    T.nms_footprint = [3,9,9]
  return T

def myrun(pid=0):
  (p0,p1),pid = parse_pid(pid,[19,2])
  savedir_local = savedir / f'e21_isbidet_predict/v01/pid{pid:03d}/'
  (savedir_local / 'm').mkdir(exist_ok=True,parents=True) ## place to save models
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  # P = _init_params(info.ndim)
  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)

  """
  EVERYTHING BELOW THIS POINT IS LIKE A WORKSPACE THAT CHANGES RAPIDLY & USES FUNCTIONS DEFINED ABOVE.
  """

  print("Running e21 prediction with savedir: \n", savedir_local, flush=True)

  n_raw  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
  n_lab  = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track
  # if info.index in [6,11,12,13,14,15,18]:
  n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  filenames  = [(n_raw.format(time=i),n_lab.format(time=i)) for i in range(info.start,info.stop)]
  pointfiles = f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl"


  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = _init_unet_params(info.ndim).net
  net = net.to(device)
  old_weights = Path(f"/projects/project-broaddus/devseg_2/expr/e21_isbidet/v09/pid{pid:03d}/m/best_weights_loss.pt")
  net.load_state_dict(torch.load(old_weights))

  N   = 7
  gap = floor((info.stop-info.start)/N)
  predict_times = range(info.start,info.stop,gap)
  savetimes = predict_times
  
  D = info.ndim
  kwargs   = dict(ndim=D,zoom=(1,1,1)[-D:],dims="ZYX"[-D:],nms_footprint=(3,5,5)[-D:],kern=[3,5,5][-D:])
  _trainparams = load(Path(f"/projects/project-broaddus/devseg_2/expr/e21_isbidet/v09/pid{pid:03d}/data_specific_params.pkl"))
  for k,v in _trainparams.__dict__.items(): kwargs[k] = v
  print(kwargs)
  with warnings.catch_warnings():
    warnings.simplefilter("once")
    ltps = predict_and_eval_centers(net,savetimes,predict_times,info,savedir_local,kwargs)



  # ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  # tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  # scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()



## Prediction functions

def _png(x):
  if x.ndim==3:
    x = x.max(0)
  norm = lambda x: (x-x.min())/(x.max()-x.min())
  x = (norm(x)*255).astype(np.uint8)
  return x

def predict_full(net,raw,**params):
  p = params
  p = {**dict(ndim=2,zoom=(1,1),dims="YX",nms_footprint=(5,5)), **p}
  p = SimpleNamespace(**p)
  assert p.ndim == raw.ndim
  raw  = normalize3(raw,2,99.4,clip=False)
  x    = zoom(raw,p.zoom,order=1)
  pred = torch_models.predict_raw(net,x,dims=p.dims,).astype(np.float32)
  # with torch.no_grad():
  #   pred = net(torch.from_numpy(x[None,None]).float().cuda())[0,0].cpu().numpy()
  height = pred.max()
  pred = pred / pred.max() ## 
  pts = peak_local_max(pred,threshold_abs=.2,exclude_border=False,footprint=np.ones(p.nms_footprint))
  pts = pts/p.zoom
  pred = zoom(pred, 1/np.array(p.zoom), order=1)
  return SimpleNamespace(pred=pred,height=height,pts=pts)



def match(yt_pts,pts,dub=100,scale=[4,1,1]):
  ndim = pts.shape[1]
  scale = [4,1,1][-ndim:]
  return match_unambiguous_nearestNeib(yt_pts,pts,dub=dub,scale=scale)


def predict_and_eval_centers(net,savetimes,predict_times,info,savedir_local,kwargs):
  # savedir_local  = savedir_local / 'pred_all_01'
  pts_gt = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
  name   = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
  shutil.rmtree(savedir_local); Path(savedir_local).mkdir()

  def _single(i):
    "i is time"
    _time = predict_times[i]
    # name   = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname.format(time=predict_times[i])
    raw    = load(name.format(time=_time))
    S      = predict_full(net,raw,**kwargs)
    matching = match(pts_gt[_time],S.pts)
    # fp, fn = find_errors(raw,matching)
    # if savedir_local: save(pts,savedir_local / "predpts/pts{i:04d}.pkl")
    target = place_gaussian_at_pts(pts_gt[_time],raw.shape,kwargs['kern'])
    matching.mse  = np.mean((S.pred-target)**2)
    matching.time = _time
    matching.pred_height = S.height
    print(f"time {_time}, f1: {matching.f1}")
    # scores = dict(f1=matching.f1,precision=matching.precision,recall=matching.recall,mse=mse)
    return SimpleNamespace(raw=raw,matching=matching,target=target,pred=S.pred,)

  def _save_preds(d,i):
    # _best_f1_score = matching.f1
    # print("Saving ", i)
    save(_png(d.raw), savedir_local/f"d{i:04d}/raw.png")
    save(_png(d.pred), savedir_local/f"d{i:04d}/pred.png")
    save(_png(d.target), savedir_local/f"d{i:04d}/target.png")
    save(d.matching, savedir_local/f"d{i:04d}/matching.pkl")

    # save(d.pts, savedir_local/f"d{i:04d}/pts.pkl")
    # save(d.pts_gt, savedir_local/f"d{i:04d}/pts_gt.pkl")
    # save(d.scores, savedir_local/f"d{i:04d}/scores.pkl")
    # save(fp, savedir_local/f"errors/t{i:04d}/fp.pkl")
    # save(fn, savedir_local/f"errors/t{i:04d}/fn.pkl")

  def _f(i): ## i is time
    # if Path(name.format(time=predict_times[i])).exists(): return None,None,None
    d = _single(i)
    if predict_times[i] in savetimes: _save_preds(d,i)
    return d.matching.pts_yp


  ltps = map(list,zip(*[_f(i) for i in range(len(predict_times))]))
  save(ltps, savedir_local/f"ltps.pkl")

  return ltps



if __name__ == '__main__':
  for i in range(19*2): myrun(i)

