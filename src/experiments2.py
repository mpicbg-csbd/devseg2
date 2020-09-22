"""
## blocking cuda enables straightforward time profiling
export CUDA_LAUNCH_BLOCKING=1
ipython

import denoiser
from segtools.ns2dir import load,save,toarray
import experiments2
import numpy as np
%load_ext line_profiler
"""

"""
Each function not prefixed by an underscore corresponds to a snakemake rule.
They accept only one params from snakemake, which is an integer pid, to vary the task (and the output) in some way.
The underscore functions are only for building method configs (denoiser/detector).
All jobs are totally self contained in terms of filenames/input/output (all hardcoded) and parameter values.
It's easy to run these jobs interactively from ipython.
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

savedir = Path('/projects/project-broaddus/devseg_2/expr/')

# def qsave(img):
#   save(img.astype(np.float16), savedir / "qsave.tif")

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
    cfig.savedir = Path(savedir / 'e01_alexretina/v2_timelapse1/')
    # cfig.best_model = savedir / 'e01_alexretina/v2_timelapse1/m/'
    cfig.masker  = denoise_utils.nearest_neib_masker
  if pid==2:
    kern = np.zeros((40,1,1)) ## must be odd
    kern[:,0,0]  = 1
    kern[20,0,0] = 2
    cfig.mask = kern
    cfig.savedir = Path(savedir / 'e01_alexretina/v2_timelapse2/')
    # cfig.best_model = savedir / 'e01_alexretina/v2_timelapse2/m/'
    cfig.masker  = denoise_utils.structN2V_masker
  if pid==3:
    kern = np.zeros((40,1,1)) ## must be odd
    kern[:,0,0]  = 1
    kern[20,0,0] = 2
    cfig.mask = kern
    cfig.masker  = denoise_utils.footprint_masker
    cfig.savedir = Path(savedir / 'e01_alexretina/v2_timelapse3/')
    # cfig.best_model = savedir / 'e01_alexretina/v2_timelapse3/m/'

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
  # save(res.astype(np.float16),T.config.savedir / 'pred_t33.tif')

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
    cfig.savedir = savedir / 'e01_alexretina/v2_timelapse2d_0/'

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

def job12_horst_predict():
  """
  run predictions for Horst on his 2nd round of long timeseries.
  2k timepoints 1 zslice
  # TODO: how to upload these?
  """
  imglist = [
    '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00001.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00002.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00003.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00004.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00005.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_00006.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/88592-openfield_00001_00003.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/88592-openfield_00001_00001.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/88592-openfield_00001_00002.tif',
    '/projects/project-broaddus/rawdata/HorstObenhaus/88592-openfield_00001_00004.tif',
  ]

  from segtools.numpy_utils import norm_to_percentile_and_dtype

  model = torch_models.Unet2(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=torch_models.nn.Sequential).cuda()
  model.load_state_dict(torch.load(savedir / "e08_horst/v2_t02/m/net049.pt"))
  print(torch_models.summary(model, (1,512,512)))

  for name in imglist:
    resname = name.replace("HorstObenhaus/","HorstObenhaus/pred_")
    if Path(resname).exists(): continue
    img = load(name)[::2]
    img = img[:,None] # to conform to "NCYX"
    print(name, Path(name).exists(), '\n')
    res = denoiser.predict_raw(model,img,"NCYX")
    res = norm_to_percentile_and_dtype(res,img,2,99.5)
    save(res,resname)
    
def job12_horst(pid=1):

  if pid <= 4: img = load('/projects/project-broaddus/rawdata/HorstObenhaus/img60480.npy')
  elif 4 < pid <= 8: img = load('/projects/project-broaddus/rawdata/HorstObenhaus/img88592.npy')
  elif 8 < pid <= 12:
    img = load(f'/projects/project-broaddus/rawdata/HorstObenhaus/60480-openfield_00001_0000{pid-8}.tif')
    img = img[:2000:20] ## odd frames are blank (seperate channel). subsample for easy training.

  # img = load('/projects/project-broaddus/rawdata/HorstObenhaus/img60480.npy')
  # if pid==9:
  #   img = load('/projects/project-broaddus/rawdata/HorstObenhaus/img88592.npy')
  #   _job12_nlm_horst(img)
  #   return

  img = img[:,None]
  cfig = _job12_horst(img,pid=pid)

  T   = denoiser.train_init(cfig)
  denoiser.train(T)
  net, ta = T.m.net, T.ta
  # ta = None
  # net = cfig.getnet().cuda()
  # net.load_state_dict(torch.load(savedir / f'e08_horst/v2_t{pid-4:02d}' / 'm/net049.pt'))

  res = denoiser.predict_raw(net,img.reshape([10,10,1,512,512]),dims="NBCYX",ta=ta)
  save(res,cfig.savedir / 'pred.npy')

def _job12_horst(data,pid=1):
  """
  data.data is synthetic membranes with values in {0,1}
  data.noisy_data is synthetic membranes with noisy values in [0,2]
  """
  
  cfig = denoiser.config_example()

  t=10_000
  cfig.times = [10,100,t//50,t//50,t]
  cfig.lr = 1e-4
  cfig.batch_space  = np.array([512,512])
  cfig.batch_shape  = np.array([1,1,512,512])
  cfig.sampler      = denoiser.flat_sampler
  cfig.getnet       = lambda : torch_models.Unet2(16, [[1],[1]], pool=(2,2), kernsize=(5,5), finallayer=torch_models.nn.Sequential)

  if pid in [1,5]:
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
  elif pid in [4,8]:
    cfig.masker = denoise_utils.nearest_neib_masker

  cfig.savedir = savedir / f'e08_horst/v2_t{pid:02d}/'
  # cfig.best_model = savedir / f'e08_horst/v2_t{pid-4:02d}' / 'm/net049.pt'

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

## Do NLM _locally/interactively in napari_
  # def _job12_nlm_horst(img):
  #   import gputools
  #   # img = img[0]
  #   # res = gputools.denoise.nlm2(img)
  #   # res = np.array([gputools.denoise.nlm2(img[i],np.log10(-2 + i/25)) for i in range(100)]) # sig
  #   # res = np.array([gputools.denoise.nlm2(img[i],-0.5 + i/33) for i in range(100)]) # sig2
  #   # img = img[0]
  #   img = normalize3(img,2,99)
  #   res = np.array([gputools.denoise.nlm2(img[i],-0.5 + i/33,size_filter=i%10,size_search=i//10) for i in range(100)]) # sig2
  #   res = res.reshape([10,10,512,512])
  #   save(res,'../e08_horst/nlm2d/pred3.npy')
  #   res = res.reshape([10,10,512,512])
  #   save(res,'../e08_horst/nlm2d/pred3.npy')

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

## train 7x1 c. elegans gaussian detector

def job14_celegans(pid=0):
  """
  train on a single timepoint with an appropriate kernel size.
  see how predictions decay at other times.
  """

  ## Train a detection model on a single timepoint
  train_time = {0:0,1:100,2:180}[pid]
  trainset = "01"
  testset = "01"
  
  img  = load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{trainset}/t{train_time:03d}.tif")
  img  = img[None,None]
  img  = normalize3(img,2,99.4,clip=False)
  pts  = np.array(load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{trainset}_traj.pkl"))[[train_time]]
  cfig = _job14_celegans(img,pts,pid=pid)
  
  ## Don't retrain if not necessary
  if Path(cfig.savedir / "m/net099.pt").exists():
    net = cfig.getnet().cuda()
    net.load_state_dict(torch.load(cfig.savedir / "m/net099.pt"))
  else:
    T = detector.train_init(cfig)
    detector.train(T)
    net, ta = T.m.net, T.ta
    res = detector.predict_raw(net,img,dims="NCZYX").astype(np.float16)
    save(res, cfig.savedir / 'pred.npy')

  ## Predict and Evaluate model result
  gtpts = load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{testset}_traj.pkl")
  scores = []

  # for i in [100,180]:
  for i in range(190):
    print(f"pred on {i}")
    outfile = cfig.savedir / f'scores{testset}/t{i:03d}.pkl'
    # if outfile.exists(): continue
    x = load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{testset}/t{i:03d}.tif")
    x = normalize3(x,2,99.4,clip=False)
    res = detector.predict_raw(net,x,dims="ZYX").astype(np.float32)
    pts = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=np.ones((3,8,8)))
    score3  = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=3,scale=[2,1,1])
    score10 = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=[2,1,1])
    s = {3:score3,10:score10}
    scores.append(s)
    save(s,outfile)
    if i in [0,100,180]: save(res, cfig.savedir/f'dpred/t{i:03d}.tif')
    print(score10.n_gt, score10.n_matched, score10.n_proposed)
  save(scores, cfig.savedir / f'scores{testset}.pkl')

def _job14_celegans(img,pts,pid):
  
  cfig = SimpleNamespace()
  cfig.savedir = savedir / f'e14_celegans/pid{pid:02d}/'
  ## prediction / evaluation / training
  cfig.getnet = lambda : torch_models.Unet3(16, [[1],[1]], finallayer=torch_models.nn.Sequential)
  ## detector target kernel shape
  cfig.sigmas       = np.array([1,7,7])
  cfig.kernel_shape = np.array([43,43,43]) ## 7sigma in each direction
  cfig.rescale_for_matching = [2,1,1]
  ## fg/bg weights stuff for fluorescence images & class-imbalanced data
  cfig.fg_bg_thresh = np.exp(-16/2)
  cfig.bg_weight_multiplier = 0.0
  cfig.weight_decay = True
  ## img sampling
  cfig.sampler      = detector.content_sampler # if pid%2==0 else detector.content_sampler
  cfig.patch_space  = np.array([16,128,128])
  cfig.batch_shape  = np.array([1,1,16,128,128])
  cfig.batch_axes   = "BCZYX"
  ## accumulate gradients, print loss, save x,y,yt,model, save vali_pred, total time in Fwd/Bwd passes
  # cfig.times = [10,100,500,4000,100_000] ##1k = 5mins 100k = 7 hours ? 100/25 = 4/s 
  time_total = 10_000
  cfig.time_weightdecay = 400 # for pixelwise weights
  cfig.time_agg = 10 # aggregate gradients before backprop
  cfig.time_print = 100
  cfig.time_savecrop = max(400,time_total//50)
  cfig.time_validate = max(500,time_total//50)
  cfig.time_total = time_total
  cfig.lr = 4e-4

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = img[[0]]
    td.target = detector._pts2target(pts[[0]],img[0,0].shape,cfig)[None]
    td.gt = pts[[0]]
    # detector.add_meta_to_td(td)
    vd = SimpleNamespace()
    vd.input  = img[[0]]
    vd.target = detector._pts2target(pts[[0]],img[0,0].shape,cfig)[None]
    vd.gt = pts[[0]]
    # detector.add_meta_to_td(vd)
    return td,vd
  cfig.load_train_and_vali_data = _ltvd

  detector.check_config(cfig)

  return cfig



## run single job from command line

if __name__ == '__main__':
  job12_horst(pid=4)


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



"""