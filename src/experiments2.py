"""
## blocking cuda enables straightforward time profiling
export CUDA_LAUNCH_BLOCKING=1
ipython

import denoiser, detector
from segtools.ns2dir import load,save,toarray
import experiments2, analysis2, ipy
import numpy as np
%load_ext line_profiler
"""

"""
Each function of the form `eXX_name` is a self-contained experiment.
Each is parameterized by a single integer param ID (pid).
The remaining functions are helpers in some way: usually building method configurations.
The top level function `run_slurm` will submit All Experiments x All Parameters to SLURM for parallel running on the cluster.
The top level value `savedir` controls a single global output home for all experiments.
"""

"""
Detection and denoising.
"""

# import torch
# from torch import nn
# import torch_models
import itertools
# from math import floor,ceil
import numpy as np
# from scipy.ndimage import label,zoom
from skimage.feature  import peak_local_max
from skimage.measure  import regionprops
from pathlib import Path
from segtools.ns2dir import load,save,flatten_sn,toarray
from segtools import torch_models
from types import SimpleNamespace
import denoiser, denoise_utils
import detector #, detect_utils
import torch
from segtools.numpy_utils import normalize3, perm2, collapse2, splt
from segtools import point_matcher
from subprocess import run, Popen
import shutil
from segtools.point_tools import trim_images_from_pts2
from scipy.ndimage import zoom
import json

import tracking
from glob import glob
import os
import re


savedir = Path('/projects/project-broaddus/devseg_2/expr/')

def _parse_pid(pid_or_params,dims):
  if type(pid_or_params) in [list,tuple]:
    params = pid_or_params
    pid = np.ravel_multi_index(params,dims)
  elif type(pid_or_params) is int:
    pid = pid_or_params
    params = np.unravel_index(pid,dims)
  return params, pid

def iterdims(shape):
  return itertools.product(*[range(x) for x in shape])

def OLD_run_slurm():
  """
  Submit all experiments as independent jobs to the SLURM cluster job manager.
  Each job has specific SLURM params, and function params, identified with a certain pid (int).

  cmd = "sbatch -J {rule} -p {cluster.p} --gres {cluster.gres} -n {cluster.n} -t {cluster.t} -c {cluster.c} --mem {cluster.mem} -o slurm/slurm-%j.out -e slurm/slurm-%j.err "

  """

  shutil.copy("/projects/project-broaddus/devseg_2/src/experiments2.py", "/projects/project-broaddus/devseg_2/src/ex2copy.py")
  # shutil.copy("/projects/project-broaddus/devseg_2/src/experiments2.py", cfig.savedir)

  # ## job10_alex_retina_3D
  # cmd = 'sbatch -J e10-3D_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e10-3D_pid{pid:02d}.out -e slurm/e10-3D_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.job10_alex_retina_3D({pid})\"\' '
  # for pid in [1,2,3]: Popen(cmd.format(pid=pid),shell=True)

  # ## job10_alex_retina_2D
  # cmd = 'sbatch -J e10-2D_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e10-2D_pid{pid:02d}.out -e slurm/e10-2D_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.job10_alex_retina_2D({pid})\"\' '
  # for pid in [1]: Popen(cmd.format(pid=pid),shell=True)

  # ## job11_synthetic_membranes
  # cmd = 'sbatch -J e11_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e11_pid{pid:02d}.out -e slurm/e11_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.job11_synthetic_membranes({pid})\"\' '
  # for pid in [1,2,3]: Popen(cmd.format(pid=pid),shell=True)

  ## e08_horst
  # shutil.copy("/projects/project-broaddus/devseg_2/src/experiments2.py", "/projects/project-broaddus/devseg_2/src/ex2copy.py")
  # cmd = 'sbatch -J e08_{pid:03d} -p gpu --gres gpu:1 -n 1 -t 1:00:00 -c 1 --mem 128000 -o slurm/e08_pid{pid:03d}.out -e slurm/e08_pid{pid:03d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e08_horst({pid})\"\' '
  # for pid in range(50): Popen(cmd.format(pid=pid),shell=True)

  # ## job13_mangal
  # cmd = 'sbatch -J e13_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e13_pid{pid:02d}.out -e slurm/e13_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.job13_mangal({pid})\"\' '
  # for pid in range(1,9): Popen(cmd.format(pid=pid),shell=True)

  # e14_celegans
  # cmd = 'sbatch -J e14_{pid:03d} -p gpu --gres gpu:1 -n 1 -t 4:00:00 -c 1 --mem 128000 -o slurm/e14_pid{pid:03d}.out -e slurm/e14_pid{pid:03d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e14_celegans({pid})\"\' '
  # for pid in range(10): Popen(cmd.format(pid=pid),shell=True)

  # ## e15_celegans
  # cmd = 'sbatch -J e15_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e15_pid{pid:02d}.out -e slurm/e15_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e15_celegans({pid})\"\' '
  # for pid in range(3): Popen(cmd.format(pid=pid),shell=True)

  ## e16_ce_adapt
  # cmd = 'sbatch -J e16_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 2:00:00 -c 1 --mem 128000 -o slurm/e16_pid{pid:02d}.out -e slurm/e16_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e16_ce_adapt({pid})\"\' '
  # for pid in range(2*5): Popen(cmd.format(pid=pid),shell=True)

  # e18_trib
  # cmd = 'sbatch -J e18_{pid:03d} -p gpu --gres gpu:1 -n 1 -t 3:00:00 -c 1 --mem 128000 -o slurm/e18_pid{pid:03d}.out -e slurm/e18_pid{pid:03d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e18_trib({pid})\"\' '
  # for pid in range(19): Popen(cmd.format(pid=pid),shell=True)

  ## e19_tracking
  # cmd = 'sbatch -J e19_{pid:03d} -n 1 -t 1:00:00 -c 4 --mem 128000 -o slurm/e19_pid{pid:03d}.out -e slurm/e19_pid{pid:03d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e19_tracking({pid})\"\' '
  # for pid in range(3*19): Popen(cmd.format(pid=pid),shell=True)

slurm = SimpleNamespace()
slurm.e10_3D = 'sbatch -J e10-3D_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e10-3D_pid{pid:02d}.out -e slurm/e10-3D_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.job10_alex_retina_3D({pid})\"\' '
slurm.e10_2D = 'sbatch -J e10-2D_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e10-2D_pid{pid:02d}.out -e slurm/e10-2D_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.job10_alex_retina_2D({pid})\"\' '
slurm.e11 = 'sbatch -J e11_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e11_pid{pid:02d}.out -e slurm/e11_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.job11_synthetic_membranes({pid})\"\' '
slurm.e08 = 'sbatch -J e08_{pid:03d} -p gpu --gres gpu:1 -n 1 -t 1:00:00 -c 1 --mem 128000 -o slurm/e08_pid{pid:03d}.out -e slurm/e08_pid{pid:03d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e08_horst({pid})\"\' '
slurm.e13 = 'sbatch -J e13_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e13_pid{pid:02d}.out -e slurm/e13_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.job13_mangal({pid})\"\' '
slurm.e14 = 'sbatch -J e14_{pid:03d} -p gpu --gres gpu:1 -n 1 -t 4:00:00 -c 1 --mem 128000 -o slurm/e14_pid{pid:03d}.out -e slurm/e14_pid{pid:03d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e14_celegans({pid})\"\' '
slurm.e15 = 'sbatch -J e15_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 12:00:00 -c 1 --mem 128000 -o slurm/e15_pid{pid:02d}.out -e slurm/e15_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e15_celegans({pid})\"\' '
slurm.e16 = 'sbatch -J e16_{pid:02d} -p gpu --gres gpu:1 -n 1 -t 2:00:00 -c 1 --mem 128000 -o slurm/e16_pid{pid:02d}.out -e slurm/e16_pid{pid:02d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e16_ce_adapt({pid})\"\' '
slurm.e18 = 'sbatch -J e18_{pid:03d} -p gpu --gres gpu:1 -n 1 -t 3:00:00 -c 1 --mem 128000 -o slurm/e18_pid{pid:03d}.out -e slurm/e18_pid{pid:03d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e18_trib({pid})\"\' '
slurm.e19 = 'sbatch -J e19_{pid:03d} -n 1 -t 3:00:00 -c 4 --mem 128000 -o slurm/e19_pid{pid:03d}.out -e slurm/e19_pid{pid:03d}.err --wrap \'python3 -c \"import ex2copy; ex2copy.e19_tracking({pid})\"\' '

def run_slurm(cmd,pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.
  shutil.copy("/projects/project-broaddus/devseg_2/src/experiments2.py", "/projects/project-broaddus/devseg_2/src/ex2copy.py")
  for pid in pids: Popen(cmd.format(pid=pid),shell=True)

## Alex's retina 3D

def job10_alex_retina_3D(pid=1,img=None):
  if img is None: img = load("../raw/every_30_min_timelapse.tiff")

  cfig = _job10_alex_retina_3D(img,pid=pid)
  T = denoiser.train_init(cfig)

  denoiser.train(T)
  # torch_models.gc.collect()
  res = denoiser.predict_raw(T.m.net,img[33], dims="ZYX", ta=T.ta, D_zyx=(16,256,256)) # pp_zyx=(4,16,16), D_zyx=(16,200,200))
  save(res,T.config.savedir / 'pred_t33.tif')
  # res = denoiser.predict_raw(T.m.net, T.td.input[0,0],ta=T.ta)
  # save(res,T.config.savedir / 'td_pred3d.tif')

def _job10_alex_retina_3D(img,pid=1):
  """
  load image, train net, predict, save
  img data already normalized (and possibly even clipped poorly)
  """
  
  cfig = denoiser.config_example()

  # t=100
  # cfig.times = [1,10,t//10,t//4,t]
  t=10_000
  cfig.times = [10,100,t//20,t//10,t]
  cfig.lr = 1e-4
  cfig.batch_space  = np.array([16,128,128])
  cfig.batch_shape  = np.array([1,1,16,128,128])
  cfig.sampler      = denoiser.flat_sampler
  cfig.getnet       = lambda : torch_models.Unet2(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)

  if pid==1:
    # kern = np.zeros((1,1,1)) ## must be odd
    # kern[0,0,0]  = 2
    # cfig.mask = kern
    cfig.masker  = denoise_utils.nearest_neib_masker
  if pid==2:
    kern = np.zeros((40,1,1)) ## must be odd
    kern[:,0,0]  = 1
    kern[20,0,0] = 2
    cfig.mask = kern
    cfig.masker  = denoise_utils.structN2V_masker
  if pid==3:
    kern = np.zeros((40,1,1)) ## must be odd
    kern[:,0,0]  = 1
    kern[20,0,0] = 2
    cfig.mask = kern
    cfig.masker  = denoise_utils.footprint_masker

  cfig.savedir = savedir / f'e01_alexretina/v2_timelapse3d_{pid}/'


  def _ltvd(config):
    td = SimpleNamespace()    
    td.input  = img[[33],None]
    td.target = img[[33],None]
    denoiser.add_meta_to_td(td)
    vd = SimpleNamespace()
    vd.input  = img[[30],None]
    vd.target = img[[30],None]
    denoiser.add_meta_to_td(vd)
    return td,vd

  cfig.load_train_and_vali_data = _ltvd

  return cfig

## Alex's retina 2D

def job10_alex_retina_2D(pid=1):
  img = load("../raw/every_30_min_timelapse.tiff")
  cfig = _job10_alex_retina_2D(img,pid=pid)
  T = denoiser.train_init(cfig)
  denoiser.train(T)
  res = denoiser.predict_raw(T.m.net,img[33,:,None],dims="NCYX",ta=T.ta)
  save(res.astype(np.float16),T.config.savedir / 'pred_t33.tif')

def _job10_alex_retina_2D(img,pid=1):
  """
  load image, train net, predict, save
  img data already normalized (and possibly even clipped poorly)
  """
  
  cfig = denoiser.config_example()
  t=10_000
  cfig.times = [10,100,t//20,t//10,t]
  cfig.lr = 1e-4
  cfig.batch_space  = np.array([512,512])
  cfig.batch_shape  = np.array([1,1,512,512])
  cfig.sampler      = denoiser.flat_sampler
  cfig.getnet       = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=torch_models.nn.Sequential)

  ## 2
  if pid==1:
    cfig.masker  = denoise_utils.nearest_neib_masker

  cfig.savedir = savedir / f'e01_alexretina/v2_timelapse2d_{pid:d}/'

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = img[33,:,None]
    td.target = img[33,:,None]
    denoiser.add_meta_to_td(td)    
    vd = SimpleNamespace()
    vd.input  = img[30,:,None]
    vd.target = img[30,:,None]
    denoiser.add_meta_to_td(vd)

    return td,vd
  cfig.load_train_and_vali_data = _ltvd

  return cfig

## Alex's synthetic membranes

def job11_synthetic_membranes(pid=1):
  signal = load('/projects/project-broaddus/rawdata/synth_membranes/gt.npy')[:6000]

  from scipy.ndimage import convolve
  def f():
    kern  = np.array([[1,1,1]])/3
    a,b,c = signal.shape
    res = []
    for i,_x in enumerate(signal):
      noise = np.random.rand(b,c)
      noise = convolve(noise,kern)
      noise = noise-noise.mean()
      res.append(noise)
    return np.array(res)
  noise = f()
  X = signal + noise

  X = X[:,None]
  cfig = _job11_synthetic_membranes(X,pid=pid)
  T = denoiser.train_init(cfig)

  save(noise, T.config.savedir / 'noise.npy')

  denoiser.train(T)
  X = X.reshape([500,12,1,128,128])
  res = denoiser.predict_raw(T.m.net,X,dims="NBCYX",ta=T.ta)
  res = res.reshape([12*500,128,128])
  save(res[::10].astype(np.float16),T.config.savedir / 'pred.npy')

def _job11_synthetic_membranes(X,pid=1):
  """
  X is synthetic membranes with noisy values in [0,2]
  """
  
  cfig = denoiser.config_example()
  t=10_000
  cfig.times = [10,100,t//20,t//10,t]
  cfig.lr = 1e-4
  cfig.batch_space  = np.array([128,128])
  cfig.batch_shape  = np.array([1,1,128,128])
  cfig.sampler      = denoiser.flat_sampler
  cfig.getnet       = lambda : torch_models.Unet2(16, [[1],[1]], pool=(2,2), kernsize=(3,3), finallayer=torch_models.nn.Sequential)

  if pid==1:
    kern = np.array([[0,0,0,1,1,1,1,1,0,0,0]])
    cfig.mask = kern
    cfig.masker = denoise_utils.structN2V_masker
  if pid==2:
    kern = np.array([[0,0,0,0,1,1,1,0,0,0,0]])
    cfig.mask = kern
    cfig.masker = denoise_utils.structN2V_masker
  if pid==3:
    kern = np.array([[0,0,0,0,0,1,0,0,0,0,0]])
    cfig.mask = kern
    cfig.masker = denoise_utils.structN2V_masker

  cfig.savedir = savedir / f'e07_synmem/v2_t{pid:02d}/'

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = X[:-50]
    td.target = X[:-50]
    denoiser.add_meta_to_td(td)    
    vd = SimpleNamespace()    
    vd.input  = X[-50:]
    vd.target = X[-50:]
    denoiser.add_meta_to_td(vd)

    return td,vd

  cfig.load_train_and_vali_data = _ltvd
  return cfig

def j11_sm_analysis():
  signal = load('/projects/project-broaddus/rawdata/synth_membranes/gt.npy')[:6000:10]
  noise  = load(savedir / 'e07_synmem/v2_t01/noise.npy')[:6000:10]
  pred   = load(savedir / 'e07_synmem/v2_t01/pred.npy')
  diff   = signal + noise - pred
  print(signal.mean(), noise.mean(), pred.mean(), diff.mean())
  """
  most of the errors that i see are on membranes along the _horizontal_, i.e. the axis of the noise.
  This is totally 
  """

## horst's calcium images

horst_data = [
  '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00001.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00002.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00003.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00004.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00005.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00006.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/88592-openfield_00001_00001.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/88592-openfield_00001_00002.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/88592-openfield_00001_00003.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/88592-openfield_00001_00004.tif',
  '/projects/project-broaddus/rawdata/HorstObenhaus/img60480.npy',
  '/projects/project-broaddus/rawdata/HorstObenhaus/img88592.npy',
  ]

def e08_horst_predict():
  """
  run predictions for Horst on his 2nd round of long timeseries.
  2k  timepoints 1 zslice
  # TODO: how to upload these?
  """

  from segtools.numpy_utils import norm_to_percentile_and_dtype
  model = torch_models.Unet2(32, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=torch_models.nn.Sequential).cuda()
  print(torch_models.summary(model, (1,512,512)))

  for params in iterdims([1,10,5,1,1]):
    (p0_unet,p1_data,p2_repeat,p3_chan,p4_datasize), pid = _parse_pid(params,[1,10,5,1,1])
    if p2_repeat != 0: continue
    model.load_state_dict(torch.load(savedir / f"e08_horst/v02/pid{pid:03d}/m/best_weights_latest.pt"))
    
    name    = horst_data[p1_data]
    resname = name.replace("HorstObenhaus/","HorstObenhaus/pred/v02/pred_")
    x = load(name)[::2]
    x = x[:,None]

    res = denoiser.predict_raw(model,x,"NCYX")
    res = norm_to_percentile_and_dtype(res,x,2,99.5)
    res = ((res + 0.28*x)/1.28).astype(np.int16)
    save(res, resname)

def e08_horst(pid=1):
  """
  Let's try a few variations on the Horst data...
  - vary amount / window of training data sampling
  - Unet2 / Unet3
  - keep 60480/88592 separate
  - we already know the optimal mask size = [1,1,1]
  - odd frames of images are blank (seperate, noisy channel).
  - visually optimize the linear combination of RAW and DENOISED so it doesn't look too smooth.
  - Compare to previous denoising
  - Predict on all stacks (about 10GB) and upload to Dropbox.

  v01: explore many options
  v02: train for a long time on all the data we can use, use the bigger chan 32 nets. repeat 5 times.
  """


  # (p0_unet,p1_data,p2_repeat,p3_chan,p4_datasize), pid = _parse_pid(pid,[2,12,5,2,5])
  (p0_unet,p1_data,p2_repeat,p3_chan,p4_datasize), pid = _parse_pid(pid,[1,10,5,1,1])

  img  = load(horst_data[p1_data])
  img  = img[::2,None] ## remove the pure-noise channel
  unet = [torch_models.Unet2, torch_models.Unet3][p0_unet]
  # chan = [16,32][p3_chan]
  chan = 32

  _mask = np.ones(img.shape[0]).astype(np.bool); _mask[[0,img.shape[0]//2,-1]]=0
  img_vd = img[~_mask]
  img_td = img[_mask]
  np.random.seed(0)
  np.random.shuffle(img_td)
  # _n_patches = [1,10,50,100,500][p4_datasize]

  img_td = img_td #[:_n_patches] # v02: TAKE THEM ALL

  def _cfig():
    cfig = SimpleNamespace()
    t=10_000
    cfig.times = [10,100,t//20,t//10,t]
    # cfig.times = [1,10,10,50,1000]
    cfig.lr = 1e-4
    cfig.batch_space  = np.array([512,512])
    cfig.batch_shape  = np.array([1,1,512,512])
    cfig.sampler      = denoiser.flat_sampler
    cfig.getnet       = lambda : unet(chan, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=torch_models.nn.Sequential)
    kern = np.array([[1,1,1]])
    cfig.mask = kern
    cfig.masker = denoise_utils.structN2V_masker
    # cfig.continue_training = False
    return cfig

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = img_td
    td.target = img_td
    denoiser.add_meta_to_td(td)
    vd = SimpleNamespace()
    vd.input  = img_vd
    vd.target = img_vd
    denoiser.add_meta_to_td(vd)
    return td,vd

  cfig = _cfig()
  cfig.savedir = savedir / f'e08_horst/v02/pid{pid:03d}/'
  cfig.load_train_and_vali_data = _ltvd

  print(json.dumps(cfig.__dict__,sort_keys=True, indent=2, default=str))

  T = denoiser.train_init(cfig)
  save(img_vd, cfig.savedir / 'vali_raw.npy') ## WARNING: don't save anything in cfig.savedir _until_ train_init

  denoiser.train(T)

  # net, ta = T.m.net, T.ta
  # return T
  # ta = None
  # net = cfig.getnet().cuda()
  # net.load_state_dict(torch.load(savedir / f'e08_horst/v2_t{pid-4:02d}' / 'm/net049.pt'))
  # res = denoiser.predict_raw(net,img[:100].reshape([10,10,1,512,512]),dims="NBCYX",ta=ta)

## mangal's nuclei

def job13_mangal(pid=3):
  img = load('/projects/project-broaddus/rawdata/mangal_nuclei/2020_08_01_noisy_images_sd4_example1.tif')
  img = img[:,None]
  cfig = _job13_mangal(img,pid=pid)

  T = denoiser.train_init(cfig)
  denoiser.train(T)
  net, ta = T.m.net, T.ta
  # ta = None
  # net = cfig.getnet().cuda()
  # net.load_state_dict(torch.load(savedir / f'../e08_horst/v2_t{pid-4:02d}/' / 'm/net049.pt'))

  res = denoiser.predict_raw(net,img.reshape([25,4,1,1024,1024]),dims="NBCYX",ta=ta)
  save(res,cfig.savedir / 'pred.npy')

def _job13_mangal(data,pid):
  """
  data is nuclei images with most values in {0,1} and shape 1x100x1024x1024 with dims "SCYX"
  """
  
  cfig = denoiser.config_example()

  t=10_000
  cfig.times = [10,t//100,t//20,t//20,t]
  cfig.lr = 1e-4
  cfig.batch_space  = np.array([1024,1024])
  cfig.batch_shape  = np.array([1,1,1024,1024])
  cfig.sampler      = denoiser.flat_sampler
  cfig.getnet       = lambda : torch_models.Unet2(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=torch_models.nn.Sequential)

  if pid in [0,1,5]:
    kern = np.array([[1,1,1,1,1]])
    cfig.mask = kern
    cfig.masker = denoise_utils.structN2V_masker
  if pid in [2,6]:
    kern = np.array([[1,1,1]])
    cfig.mask = kern
    cfig.masker = denoise_utils.structN2V_masker
  elif pid in [3,7]:
    kern = np.array([[1]])
    cfig.mask = kern
    cfig.masker = denoise_utils.structN2V_masker
  elif pid in [4]:
    kern = np.array([[1,1,1,1,1,1,1,1,1]])
    cfig.mask = kern
    cfig.masker = denoise_utils.structN2V_masker    
  elif pid in [8]:
    cfig.masker = denoise_utils.nearest_neib_masker

  cfig.savedir = savedir / f'e09_mangalnuclei/v3_t{pid:02d}/'
  # cfig.best_model = savedir / f'../e08_horst/v2_t{pid-4:02d}/' / 'm/net049.pt'

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = data[:95]
    td.target = data[:95]
    denoiser.add_meta_to_td(td)
    vd = SimpleNamespace()
    vd.input  = data[95:]
    vd.target = data[95:]
    denoiser.add_meta_to_td(vd)
    return td,vd
  cfig.load_train_and_vali_data = _ltvd

  return cfig

## train c. elegans gaussian detector

def e14_celegans(pid=0):
  """
  train on a single timepoint with an appropriate kernel size.
  see how predictions decay at other times.
  v02 changed the size of the target kernel container, because the big kernel had square boundaries.
  v03 added augmentation
  v04 corrected 01/t006 annotations. [3,3,5].
  v05 refactor _ltvd to allow for mutliple timepoints (eliminating need for separate e16). [4,3,5]
  v06 do stratified sampling over train time. five repeats, sampling from 1..31 stacks...
  v07 continuously scale kernel size on three timepoints.
  v08 pick specific train and vali times and set bg_weight multiplier. try to recreate old experiment results. 10 identical runs.
  """

  # if type(pid) is list:
  #   p0,p1,p2 = pid
  #   pid = np.ravel_multi_index(pid,[30,1,5])
  # else:
  #   p0,p1,p2 = np.unravel_index(pid,[30,1,5]) ## train timepoint, kernel size, repeat n, 
  # print("params: ", p0, p1, p2)

  # random.choice samples without replacement, so it works for train/vali 
  def stratsampler(n):
    res   = np.array([np.random.choice(x,2,replace=False) for x in np.array_split(np.r_[:190],n)])
    train = res[:,0]
    vali  = res[:,1]
    return train,vali

  train_times, vali_times = [6,100,180], [7,101,181]

  ## v08 only: try to match the old experiments we did
  train_times = [0,5,33,100,189]
  vali_times  = [0,1,180]

  # train_times, vali_times = stratsampler(p0+1)
  print(train_times,vali_times)
  ## convert p's to meaningful params
  # train_times  = [[6],[100],[180],[6,100,180]][p0]
  # vali_times   = [[7],[101],[181],[7,101,181]][p0]
  # kernel_shape = [(1,3,3),(1.5,7,7),(2,11,11)][1]
  # kernel_shape = np.array([1.5,7,7])*np.linspace(1/4,2,30)[p0]
  kernel_shape = np.array([1.5,7,7])
  print(kernel_shape)
  trainset = "01"
  testset  = "01"
  
  ## load pts and apply corrections
  pts  = np.array(load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{trainset}_traj.pkl"))
  correction_6 = load("/projects/project-broaddus/devseg_2/raw/t006.npy")
  correction_6 = correction_6[:,[1,2,3]].astype(np.int)
  pts[6] = correction_6
  correction_7 = load("/projects/project-broaddus/devseg_2/raw/t007.npy")
  correction_7 = correction_7[:,[1,2,3]].astype(np.int)
  pts[7] = correction_7

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{trainset}/t{n:03d}.tif") for n in train_times])
    td.input  = td.input[:,None]  ## add channels
    td.input  = normalize3(td.input,2,99.4,clip=False)
    td.target = detector.pts2target(pts[train_times],td.input[0,0].shape,kernel_shape)
    td.target = td.target[:,None] ## add channels
    td.gt = pts[train_times]

    vd = SimpleNamespace()
    vd.input  = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{trainset}/t{n:03d}.tif") for n in vali_times])
    vd.input  = vd.input[:,None] ## add channels
    vd.input  = normalize3(vd.input,2,99.4,clip=False)
    vd.target = detector.pts2target(pts[vali_times],vd.input[0,0].shape,kernel_shape)
    vd.target = vd.target[:,None] ## add channels
    vd.gt = pts[vali_times]
    
    return td,vd

  cfig = _e14_celegans()
  cfig.load_train_and_vali_data = _ltvd
  cfig.savedir = savedir / f'e14_celegans/v08/pid{pid:03d}/'
  cfig.time_total = 20_000 ## special for e14-v08-pid009

  ## Train the net
  # T = detector.train_init(cfig)
  T = detector.train_continue(cfig,cfig.savedir / 'm/best_weights_f1.pt')
  T.ta.train_times = train_times
  T.ta.vali_times  = vali_times
  # shutil.copy("/projects/project-broaddus/devseg_2/src/experiments2.py", cfig.savedir)
  detector.train(T)

  ## Reload best weights
  net = cfig.getnet().cuda()
  net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_f1.pt"))

  ## show prediction on train & vali data
  if False:
    td,vd = _ltvd(0)
    res = detector.predict_raw(net,td.input,dims="NCZYX").astype(np.float16)
    save(res, cfig.savedir / 'pred_train.npy')
    res = detector.predict_raw(net,vd.input,dims="NCZYX").astype(np.float16)
    save(res, cfig.savedir / 'pred_vali.npy')

  ## Predict and Evaluate model result on all available data
  gtpts = load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{testset}_traj.pkl")
  if testset=='01':
    gtpts[6] = pts[6]
    gtpts[7] = pts[7]
  scores = []
  stack  = []
  for i in np.r_[:190]:
    outfile = cfig.savedir / f'dscores{testset}/t{i:03d}.pkl'
    # if outfile.exists(): continue
    x = load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{testset}/t{i:03d}.tif")
    x = normalize3(x,2,99.4,clip=False)
    res = detector.predict_raw(net,x,dims="ZYX").astype(np.float32)
    pts = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
    score3  = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=3,scale=[2,1,1])
    score10 = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=[2,1,1])
    s = {3:score3,10:score10}
    scores.append(s)
    print("time", i, "gt", score10.n_gt, "match", score10.n_matched, "prop", score10.n_proposed)
    stack.append(res.max(0))

  save(scores, cfig.savedir / f'scores{testset}.pkl')
  # save(np.array(stack).astype(np.float16), cfig.savedir / f"pred_{testset}.npy")

def _e14_celegans():
  
  # p0,p1,p2 = np.unravel_index(pid,[4,3,5]) ## train timepoint, kernel size, repeat n, 

  ## all config params
  cfig = SimpleNamespace()
  cfig.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
  # cfig.sigmas         = np.array(kernelshape)
  # cfig.kernel_shape   = (cfig.sigmas*7).astype(np.int) ## 7sigma in each direction?
  cfig.rescale_for_matching = [2,1,1]
  cfig.fg_bg_thresh = np.exp(-16/2)
  cfig.bg_weight_multiplier = 0.2 #1.0
  cfig.weight_decay = True
  cfig.time_weightdecay = 1600 # for pixelwise weights
  cfig.sampler      = detector.content_sampler
  cfig.patch_space  = np.array([16,128,128])
  cfig.batch_shape  = np.array([1,1,16,128,128])
  cfig.batch_axes   = "BCZYX"
  time_total = 10_000 # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr
  cfig.time_agg = 1 # aggregate gradients before backprop
  cfig.time_loss = 10
  cfig.time_print = 100
  cfig.time_savecrop = max(100,time_total//50)
  cfig.time_validate = max(600,time_total//50)
  cfig.time_total = time_total
  cfig.lr = 4e-4

  return cfig

## example detection in c. elegans with StructN2V denoising

def e15_celegans(pid=0):
  """
  denoise c. elegans images
  """
  p0,p1,p2 = np.unravel_index(pid,[3,3,3]) ## train timepoint, kernel size, n repeats
  ## NOTE: pid = np.ravel_multi_index([p0,p1,p2],[3,3,3])

  ## Train a detection model on a single timepoint
  train_time = {0:6,1:100,2:180}[p0]

  trainset = "01"
  testset  = "01"
  
  img  = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{trainset}/t{n:03d}.tif") for n in [train_time, train_time+1]])
  img  = img[:,None]
  img  = normalize3(img,2,99.4,clip=False)
  pts  = np.array(load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{trainset}_traj.pkl"))[[train_time, train_time+1]]
  cfig = _e15_celegans(img,pid=pid)

  if True:
    T = denoiser.train_init(cfig)
    # return SimpleNamespace(**locals())
    denoiser.train(T)
    net, ta = T.m.net, T.ta


  res = denoiser.predict_raw(net,img,dims="NCZYX",ta=ta)
  save(res,cfig.savedir / 'pred.npy')

def _e15_celegans(img,pid=0):

  cfig = SimpleNamespace()

  t=5_000
  cfig.times = [10,100,t//50,t//50,t]
  cfig.lr = 1e-4
  cfig.batch_space  = np.array([16,128,128])
  cfig.batch_shape  = np.array([1,1,16,128,128])
  cfig.sampler      = denoiser.flat_sampler
  cfig.getnet       = lambda : torch_models.Unet2(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)

  kern = np.ones([1,17,1]) #np.array([[[1]]])
  cfig.mask = kern
  cfig.masker = denoise_utils.structN2V_masker

  cfig.savedir = savedir / f'e15_celegans/pid{pid:02d}/'

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = img[[0]]
    td.target = img[[0]]
    denoiser.add_meta_to_td(td)
    vd = SimpleNamespace()
    vd.input  = img[[1]]
    vd.target = img[[1]]
    denoiser.add_meta_to_td(vd)
    return td,vd
  cfig.load_train_and_vali_data = _ltvd

  return cfig

## multi-size kernels

def e16_ce_adapt(pid=0):
  """
  train on multiple timepoints with adaptive kernel sizes.
  v01 same three sizes we've used all along on 6,100,180, plus slightly smaller versions. (they perorm better!).
  """

  if type(pid) is list:
    p0,p1 = pid
    pid = np.ravel_multi_index(pid,[2,5])
  else:
    p0,p1 = np.unravel_index(pid,[2,5]) ## kernel size, repeat n, 
  print("params:", p0,p1)

  ## convert p's to meaningful params
  train_times   = [6,100,180] #[p0]
  vali_times    = [7,101,181] #[p0]
  kernel_shapes = [[(2,11,11),(1.5,7,7),(1,3,3)],
                   [(2,7,7),(1.5,5,5),(1,3,3)]][p0]
  trainset = "01" #["01","02"][p0]
  testset  = "01" #,"02"][p1]
  
  ## load pts and apply corrections
  pts = np.array(load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{trainset}_traj.pkl"))
  correction_6 = load("/projects/project-broaddus/devseg_2/raw/t006.npy")
  correction_6 = correction_6[:,[1,2,3]].astype(np.int)
  pts[6] = correction_6
  correction_7 = load("/projects/project-broaddus/devseg_2/raw/t007.npy")
  correction_7 = correction_7[:,[1,2,3]].astype(np.int)
  pts[7] = correction_7

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{trainset}/t{n:03d}.tif") for n in train_times])
    td.input  = td.input[:,None]  ## add channels
    td.input  = normalize3(td.input,2,99.4,clip=False)
    td.target = detector.pts2target_many(pts[train_times],td.input[0,0].shape,kernel_shapes)
    td.target = td.target[:,None] ## add channels
    td.gt = pts[train_times]

    vd = SimpleNamespace()
    vd.input  = np.array([load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{trainset}/t{n:03d}.tif") for n in vali_times])
    vd.input  = vd.input[:,None] ## add channels
    vd.input  = normalize3(vd.input,2,99.4,clip=False)
    vd.target = detector.pts2target_many(pts[vali_times],vd.input[0,0].shape,kernel_shapes)
    vd.target = vd.target[:,None] ## add channels
    vd.gt = pts[vali_times]
    
    return td,vd

  cfig = _e14_celegans() ## MUST USE SAME NET CONFIGURATION!!
  cfig.load_train_and_vali_data = _ltvd
  cfig.savedir = savedir / f'e16_ce_adapt/v01/pid{pid:02d}/'

  # Train the net
  T = detector.train_init(cfig)
  # T = detector.train_continue(cfig)
  # shutil.copy("/projects/project-broaddus/devseg_2/src/experiments2.py", cfig.savedir)
  detector.train(T)

  ## Reload best weights
  net = cfig.getnet().cuda()
  net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_f1.pt"))

  ## show prediction on train & vali data
  if False:
    td,vd = _ltvd(0)
    res = detector.predict_raw(net,td.input,dims="NCZYX").astype(np.float16)
    save(res, cfig.savedir / 'pred_train.npy')
    res = detector.predict_raw(net,vd.input,dims="NCZYX").astype(np.float16)
    save(res, cfig.savedir / 'pred_vali.npy')

  ## Predict and Evaluate model result on all available data
  gtpts = load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{testset}_traj.pkl")
  if testset=='01':
    gtpts[6] = pts[6]
    gtpts[7] = pts[7]
  scores = []
  stack  = []
  for i in np.r_[:190]:
    outfile = cfig.savedir / f'dscores{testset}/t{i:03d}.pkl'
    # if outfile.exists(): continue
    x = load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{testset}/t{i:03d}.tif")
    x = normalize3(x,2,99.4,clip=False)
    res = detector.predict_raw(net,x,dims="ZYX").astype(np.float32)
    pts = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
    score3  = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=3,scale=[2,1,1])
    score10 = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=[2,1,1])
    s = {3:score3,10:score10}
    scores.append(s)
    print("time", i, "gt", score10.n_gt, "match", score10.n_matched, "prop", score10.n_proposed)
    stack.append(res.max(0))

  save(scores, cfig.savedir / f'scores{testset}.pkl')
  save(np.array(stack).astype(np.float16), cfig.savedir / f"pred_{testset}.npy")

def e17_ce_nuclei(pid=0):
  """
  Train and pred on individual nuclei.
  Estimate size info from different nuclei.
  Try to predict time?
  """
  pass



## proof of principle for every dataset
isbi_datasets = [
  ("HSC",             "BF-C2DL-HSC"),
  ("MuSC",            "BF-C2DL-MuSC"),
  ("HeLa",            "DIC-C2DH-HeLa"),
  ("MSC",             "Fluo-C2DL-MSC"),
  ("A549",            "Fluo-C3DH-A549"),
  ("A549-SIM",        "Fluo-C3DH-A549-SIM"),
  ("H157",            "Fluo-C3DH-H157"),
  ("MDA231",          "Fluo-C3DL-MDA231"),
  ("GOWT1",           "Fluo-N2DH-GOWT1"),
  ("SIM+",            "Fluo-N2DH-SIM+"),
  ("HeLa",            "Fluo-N2DL-HeLa"),
  ("celegans_isbi",   "Fluo-N3DH-CE"),
  ("hampster",        "Fluo-N3DH-CHO"),
  ("SIM+",            "Fluo-N3DH-SIM+"),
  ("fly_isbi",        "Fluo-N3DL-DRO"),
  ("trib_isbi_proj",  "Fluo-N3DL-TRIC"),
  ("U373",            "PhC-C2DH-U373"),
  ("PSC",             "PhC-C2DL-PSC"),
  ("trib_isbi/crops", "Fluo-N3DL-TRIF"),
  ]

## WARNING: IN XYZ ORDER!!!
scales = {
  "Fluo-C3DH-A549":      (0.126, 0.126, 1.0),
  "Fluo-C3DH-H157":      (0.126, 0.126, 0.5),
  "Fluo-C3DL-MDA231":    (1.242, 1.242, 6.0),
  "Fluo-N3DH-CE":        (0.09 , 0.09, 1.0),
  "Fluo-N3DH-CHO":       (0.202, 0.202, 1.0),
  "Fluo-N3DL-DRO":       (0.406, 0.406, 2.03),
  "Fluo-N3DL-TRIC":      (1.,1.,1.), # NA dueto cartographic projections
  "Fluo-N3DL-TRIF":      (0.38 , 0.38, 0.38),
  "Fluo-C3DH-A549-SIM":  (0.126, 0.126, 1.0),
  "Fluo-N3DH-SIM+":      (0.125, 0.125, 0.200),
  "BF-C2DL-HSC" :        (0.645 ,0.645),
  "BF-C2DL-MuSC" :       (0.645 ,0.645),
  "DIC-C2DH-HeLa" :      (0.19 ,0.19),
  "Fluo-C2DL-MSC" :      (0.3 ,0.3), # (0.3977 x 0.3977) for dataset 2?,
  "Fluo-N2DH-GOWT1" :    (0.240 ,0.240),
  "Fluo-N2DL-HeLa" :     (0.645 ,0.645),
  "PhC-C2DH-U373" :      (0.65 ,0.65),
  "PhC-C2DL-PSC" :       (1.6 ,1.6),
  "Fluo-N2DH-SIM+" :     (0.125 ,0.125),
  }

def e18_trib(pid=0):
  """
  v01 : For each 3D ISBI dataset: Train, Vali, Predict on times 000,001,002 respectively. pid selects dataset.
  """

  myname, isbiname  = isbi_datasets[pid]

  trainset, testset = "01", "01"
  bg_weight_multiplier = 1.0
  train_times, vali_times, pred_times = np.r_[0:6], np.r_[7:10], np.r_[0:13]
  tif_name = "t{n:04d}.tif" if myname in ["HSC", "MuSC",] else "t{n:03d}.tif"

  if "2D" in isbiname:
    ndim = 2
    batch_shape = [1,1,512,512]
    _getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=torch_models.nn.Sequential)
    kernel_shape = [5,5]
    print("2D",myname, isbiname)
  elif "3D" in isbiname:
    ndim = 3
    batch_shape  = [1,1,16,128,128]
    _getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
    kernel_shape = [2,5,5]
    print("3D",myname, isbiname)


  if myname=="celegans_isbi":
    kernel_shape = [1,7,7]
    bg_weight_multiplier = 0.2
  if myname=="trib_isbi":
    kernel_shape = [3,3,3]
    train_times, vali_times, pred_times = [0],[1],[0,1,2]
  if myname=="fly_isbi":
    bg_weight_multiplier=0.0
    train_times, vali_times, pred_times = [0],[1],[0,1,2]
  if myname=="MDA231":   kernel_shape = [1,3,3]
  if myname=="A549":     kernel_shape = [1,5,5]
  if myname=="hampster": kernel_shape = [1,5,5]
  if myname=="psc":
    train_times, vali_times, pred_times = np.r_[150:156], np.r_[157:160], np.r_[150:163]

  kernel_shape = np.array(kernel_shape)

  print("train_times,vali_times", train_times,vali_times)
  print("kernel_shape", kernel_shape)

  pts  = load(f"/projects/project-broaddus/rawdata/{myname}/traj/{isbiname}/{trainset}_traj.pkl")
  if pid==9: pts = {k:(v*(1,0.5,0.5)).astype(np.int) for k,v in pts.items()}
  
  def get_many(dict,list_of_keys):
    return np.array([dict[k] for k in list_of_keys])

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = np.array([load(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + tif_name.format(n=n)) for n in train_times])
    if pid==9: td.input  = td.input[:,:,::2,::2]
    td.input  = td.input[:,None]  ## add channels
    td.input  = normalize3(td.input,2,99.4,clip=False)
    td.target = detector.pts2target(get_many(pts,train_times),td.input[0,0].shape,kernel_shape)
    td.target = td.target[:,None] ## add channels
    td.gt = get_many(pts,train_times)

    vd = SimpleNamespace()
    vd.input  = np.array([load(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + tif_name.format(n=n)) for n in vali_times])
    if pid==9: vd.input  = vd.input[:,:,::2,::2]
    vd.input  = vd.input[:,None] ## add channels
    vd.input  = normalize3(vd.input,2,99.4,clip=False)
    vd.target = detector.pts2target(get_many(pts,vali_times),vd.input[0,0].shape,kernel_shape)
    vd.target = vd.target[:,None] ## add channels
    vd.gt = get_many(pts,vali_times)
    
    return td,vd

  def _config():

    ## all config params
    cfig = SimpleNamespace()
    cfig.getnet = _getnet
    cfig.rescale_for_matching = [1,1,1] if len(kernel_shape)==3 else [1,1]
    cfig.fg_bg_thresh = np.exp(-16/2)
    cfig.bg_weight_multiplier = bg_weight_multiplier #0.2 #1.0
    cfig.time_weightdecay = 1600 # for pixelwise weights
    cfig.weight_decay = False
    cfig.sampler      = detector.content_sampler
    cfig.patch_space  = np.array(batch_shape[2:])
    cfig.batch_shape  = np.array(batch_shape)
    cfig.batch_axes   = "BCZYX"
    time_total = 10_000 # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr
    cfig.time_agg = 1 # aggregate gradients before backprop
    cfig.time_loss = 10
    cfig.time_print = 100
    cfig.time_savecrop = max(100,time_total//50)
    cfig.time_validate = max(600,time_total//50)
    cfig.time_total = time_total
    cfig.lr = 4e-4

    return cfig

  cfig = _config()
  cfig.load_train_and_vali_data = _ltvd
  cfig.savedir = savedir / f'e18_trib/v01/pid{pid:03d}/'

  ## Train the net
  if True:
    T = detector.train_init(cfig)
    # T = detector.train_continue(cfig,cfig.savedir / 'm/best_weights_f1.pt')
    T.ta.train_times = train_times
    T.ta.vali_times  = vali_times
    shutil.copy("/projects/project-broaddus/devseg_2/src/experiments2.py", cfig.savedir)
    detector.train(T)

  ## Reload best weights
  net = cfig.getnet().cuda()
  net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_f1.pt"))

  ## Show prediction on train & vali data
  if False:
    td,vd = _ltvd(0)
    res = detector.predict_raw(net,td.input,dims="NCZYX").astype(np.float16)
    save(res, cfig.savedir / 'pred_train.npy')
    res = detector.predict_raw(net,vd.input,dims="NCZYX").astype(np.float16)
    save(res, cfig.savedir / 'pred_vali.npy')

  ## Predict and Evaluate model result on all available data
  gtpts = load(f"/projects/project-broaddus/rawdata/{myname}/traj/{isbiname}/{testset}_traj.pkl")

  # if myname == "celegans_isbi" and testset=='01':
  #   gtpts[6] = pts[6]
  #   gtpts[7] = pts[7]


  ndims = list(gtpts.values())[0].shape[1]
  if ndims==3:
    footy = np.ones((3,8,8))
    dims = "ZYX"
    scale = [1,1,1]
  elif ndims==2:
    footy = np.ones((8,8))
    dims = "YX"
    scale = [1,1]

  def zmax(x):
    if x.ndim==3: return x.max(0)
    return x

  scores = []
  pred   = []
  raw    = []
  for i in predtimes:
    rawname = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{testset}/t{i:03d}.tif"
    print(rawname)
    x = load(rawname)
    x2 = x.copy()
    if pid==9:
      x2=x[:,::2,::2]
    x2  = normalize3(x2,2,99.4,clip=False)
    res = detector.predict_raw(net,x2,dims=dims).astype(np.float32)
    pts = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=footy)
    if pid==9: 
      pts = pts*(1,2,2)
      res = zoom(res,(1,2,2))
    score3  = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=3,scale=scale)
    score10 = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=scale)
    print("time", i, "gt", score10.n_gt, "match", score10.n_matched, "prop", score10.n_proposed)
    s = {3:score3,10:score10}
    scores.append(s)
    pred.append(zmax(res))
    raw.append(zmax(x))

  save(scores, cfig.savedir / f'scores_{testset}.pkl')
  save(np.array(pred).astype(np.float16), cfig.savedir / f"pred_{testset}.npy")
  save(np.array(raw).astype(np.float16),  cfig.savedir / f"raw_{testset}.npy")

def e19_tracking(pid=0):

  (p0,p1,p2),pid = _parse_pid(pid,[3,19,2])

  myname, isbiname = isbi_datasets[p1]
  isbi_dir = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/"
  for dataset in ['01','02']:
    trackfiles  = sorted(glob(os.path.join(isbi_dir,dataset+"_GT","TRA","man_track*.tif")))
    _timebounds = [re.search(r'man_track(\d+)\.tif',x).group(1) for x in trackfiles]
    _ndigits    = len(_timebounds[0])
    _shape = load(trackfiles[0]).shape
    start,stop  = int(_timebounds[0]), int(_timebounds[-1])+1    
    scale = np.array(scales[isbiname])[::-1]
    scale = scale / scale[-1]
    ignore_FP = '' if isbiname not in ["Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF"] else '1'
    print(myname,isbiname,dataset,_ndigits,start,stop,scale,sep='\t')

    outdir = savedir/f"e19_tracking/v01/pid{pid:03d}/"

    _tracking = [lambda lpts: tracking.nn_tracking_on_ltps(lpts,scale=scale),
                 lambda lpts: tracking.random_tracking_on_ltps(lpts)
                ][p2]
    if p2>0 and p0==1: return


    if p0==0:
      nap  = tracking.load_isbi2nap(isbi_dir,dataset,[start,stop])
      ltps = tracking.nap2ltps(nap)
      tb = _tracking(lpts)
      tracking._tb_add_orig_labels(tb,nap)
      lbep = tracking.save_permute_existing(tb,isbi_dir,dataset,[start,stop],savedir=outdir)
    if p0==1:
      nap  = tracking.load_isbi2nap(isbi_dir,dataset,[start,stop])
      kern = nap.avg_kern
      tracking.save_isbi(nap,shape=_shape,_kern=kern,savedir=outdir)
    if p0==2:
      lpts = load(f"/projects/project-broaddus/rawdata/{myname}/traj/{isbiname}/{dataset}_traj.pkl")
      if type(lpts) is dict:
        lpts = [lpts[k] for k in sorted(lpts.keys())]
      kern = np.ones([3,5,5]) if '3D' in isbiname else np.ones([5,5])
      tb = _tracking(lpts)
      nap = tracking.tb2nap(tb,lpts)
      nap.tracklets[:,1] += start
      tracking.save_isbi(nap,shape=_shape,_kern=kern,savedir=outdir)

    # lsd1 = tracking.lbep2lsd(tracking.load_lbep("naptest_pid18/res_track.txt"))
    # lsd2 = tracking.TRAdir2lsd("naptest_pid18")
    # tracking.compare_lsds(lsd1,lsd2)

    bashcmd = f"""
    isbidir={isbi_dir}
    localtra=/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/TRAMeasure
    mkdir -p "$isbidir"{dataset}_RES/
    rm "$isbidir"{dataset}_RES/*
    cp -r {outdir}/* "$isbidir"{dataset}_RES/
    time $localtra $isbidir {dataset} {_ndigits} > {outdir}/{dataset}_TRA.txt
    cat {outdir}/{dataset}_TRA.txt
    rm "$isbidir"{dataset}_RES/*.tif
    rm {outdir}/*.tif
    """
    run(bashcmd,shell=True)




if __name__ == '__main__':
  "RUN ALL THE EXPERIMENTS!"
  run_slurm()


history = """

# Tue Apr 14 15:35:21 2020

I'm not getting good results with every_30_min_timelapse.tiff in job10_predict_alex_retina_t33
But that is the clearest image we have?

param-free experiments are different from "one time tasks".
They need to be re-executed as we change the internal params, but they don't need to follow the standard experiment interface with wildcard params.
Ideally we could easily rerun all of them (in parallel). This is possible if we connect them properly to snakemake.

# Sat May 23 18:01:49 2020

We have two Horst datasets. Should we train on one, pred on the other? Or do it 2-way cross validation?

# Wed Aug  5 13:28:58 2020

Adding Mangal's nuclei datasets to see if my code produces stripes like the juglab StructN2V implementation does.

# 2020-09-17 15:36:53

# Thu Sep 24 01:39:44 2020

OK, I realized that for this type of experiment structure I don't need snakemake at all, nor any other fancy shit.
Each experiment function is totally self contained, with all paths, etc, except for a single, global savedir path.
Also, I can run all parameter versions of all jobs via SLURM with a single run_slurm() function!

# Sat Sep 26 23:19:00 2020

Adding e16.




"""