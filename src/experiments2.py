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
from types import SimpleNamespace
import denoiser, denoise_utils, denoiser2d


## Alex's retina

def job10_alex_retina_3D(_id=1):
  img = load("../raw/every_30_min_timelapse.tiff")
  T   = _job10_a(img,_id=_id)
  denoiser.train(T)
  res = denoiser.predict_raw(T, T.td.input[0,0])
  save(res,T.config.savedir / 'pred_t33.tif')
  # res = denoiser.predict_raw(T, T.td.input[0,0])
  # save(res,T.config.savedir / 'td_pred3d.tif')

def _job10_a(img,_id=1):
  """
  load image, train net, predict, save
  img data already normalized (and possibly even clipped poorly)
  """
  
  cfig = denoiser.config(denoiser.eg_img_meta())

  if _id==1:
    # kern = np.zeros((1,1,1)) ## must be odd
    # kern[0,0,0]  = 2
    # cfig.mask = kern
    cfig.savedir = '../e01_alexretina/timelapse/'
    cfig.masker = denoise_utils.nearest_neib_sampler
  if _id==2:
    kern = np.zeros((40,1,1)) ## must be odd
    kern[:,0,0]  = 1
    kern[20,0,0] = 2
    cfig.mask = kern
    cfig.savedir = '../e01_alexretina/timelapse2/'
    cfig.masker  = denoise_utils.footprint_sampler
  if _id==3:
    kern = np.zeros((40,1,1)) ## must be odd
    kern[:,0,0]  = 1
    kern[20,0,0] = 2
    cfig.mask = kern
    cfig.savedir = '../e01_alexretina/timelapse_test/'
    cfig.i_final = 2001

  cfig.i_final      = 10**5
  cfig.bp_per_epoch = 4*10**3
  cfig.lr = 1e-4

  def _ltvd(config):
    td = SimpleNamespace()
    vd = SimpleNamespace()
    
    td.input  = img[None,None,30]
    td.target = img[None,None,30]
    denoiser.add_meta_to_td(td)
    
    vd.input  = img[None,None,33]
    vd.target = img[None,None,33]
    denoiser.add_meta_to_td(vd)

    return td,vd
  cfig.load_train_and_vali_data = _ltvd

  T = denoiser.train_init(cfig)
  return T

def job10_alex_retina_2D(_id=0):
  img = load("../raw/every_30_min_timelapse.tiff")
  T = _job10_b(img)
  denoiser2d.train(T)
  # res = denoiser2d.predict_raw(T, T.td.input[0,0])
  _predict_and_save_2D(T,img)

def _predict_and_save_2D(T,img):
  res = []
  for i in range(img[33].shape[0]):
    res.append(denoiser2d.predict_raw(T,img[33,[i]]))
  res = np.array(res)
  save(res.astype(np.float16),T.config.savedir / 'pred_t33.tif')
  save(res[20,0],T.config.savedir / 'pred_t33_z20.tif')
  return res

def _job10_b(img):
  """
  load image, train net, predict, save
  img data already normalized (and possibly even clipped poorly)
  """
  
  cfig = denoiser2d.config(denoiser2d.eg_img_meta())

  # # 1
  # kern = np.zeros((40,1,1)) ## must be odd
  # kern[:,0,0]  = 1
  # kern[20,0,0] = 2
  # cfig.mask = kern
  # cfig.savedir = '../e01_alexretina/timelapse/'

  ## 2
  kern = np.zeros((1,1)) ## must be odd
  kern[0,0]  = 2
  cfig.mask = kern
  cfig.savedir = '../e01_alexretina/timelapse2_2d/'

  cfig.i_final      = 10**5
  cfig.bp_per_epoch = 4*10**3
  cfig.lr = 1e-4

  def _ltvd(config):
    td = SimpleNamespace()
    vd = SimpleNamespace()
    
    # td.input  = img[None,None,30,20]
    # td.target = img[None,None,30,20]
    td.input  = img[30,:,None]
    td.target = img[30,:,None]
    denoiser2d.add_meta_to_td(td)
    
    # vd.input  = img[None,None,33,20]
    # vd.target = img[None,None,33,20]
    vd.input  = img[33,:,None]
    vd.target = img[33,:,None]
    denoiser2d.add_meta_to_td(vd)

    return td,vd
  cfig.load_train_and_vali_data = _ltvd

  T = denoiser2d.train_init(cfig)
  return T

## Alex's synthetic membranes

def job11_synthetic_membranes(_id=1):
  signal = load('/projects/project-broaddus/rawdata/synth_membranes/gt.npy')

  from scipy.ndimage import convolve
  def f():
    noise = []
    kern  = np.array([[1,1,1]])/3
    a,b,c = signal.shape
    for i,_x in enumerate(signal):
      noise = np.random.rand(b,c)
      noise = convolve(noise,kern)
      noise = noise-noise.mean()
      noise.append(noise)
    return np.array(noise)
  noise = f()
  noisy_dataset = signal + noise

  mu,sig = np.mean(noisy_dataset), np.std(noisy_dataset)
  X = (noisy_dataset-mu)/sig
  X = X[:,None]
  T = _job11_synthetic_membranes(X,_id=_id)

  save(noise, T.config.savedir / 'noise.npy')

  denoiser2d.train(T)
  X = X[:6000].reshape([12,500,1,128,128])
  res = denoiser2d.predict_raw(T,X,dims="NBCYX")
  res = res.reshape([12*500,1,128,128])
  res = res*sig + mu
  save(res[::10].astype(np.float16),T.config.savedir / 'pred.npy')

def _job11_synthetic_membranes(noisy_data,_id=1):
  """
  noisy_data is synthetic membranes with noisy values in [0,2]
  """
  
  cfig = denoiser2d.config(denoiser2d.eg_img_meta())

  if _id==1:
    kern = np.array([[0,0,0,1,1,1,1,1,0,0,0]])
    cfig.mask = kern
    cfig.masker = denoise_utils.apply_structN2Vmask
    cfig.savedir = '../e07_synmem/t01/'

  cfig.i_final      = 10**5
  cfig.bp_per_epoch = 4*10**3
  cfig.lr = 1e-4
  cfig.patch_space  = np.array([128,128])
  cfig.patch_full   = np.array([1,1,128,128])

  def _ltvd(config):
    td = SimpleNamespace()
    td.input  = noisy_data[800:]
    td.target = noisy_data[800:]
    denoiser2d.add_meta_to_td(td)
    vd = SimpleNamespace()
    vd.input  = noisy_data[:800:40]
    vd.target = noisy_data[:800:40]
    denoiser2d.add_meta_to_td(vd)
    return td,vd

  cfig.load_train_and_vali_data = _ltvd

  T = denoiser2d.train_init(cfig)
  return T

def j11_sm_analysis():
  data = load('/projects/project-broaddus/rawdata/synth_membranes/gt.npy')


## horst's calcium images

def job12_horst(id=1):
  img = load('/projects/project-broaddus/rawdata/HorstObenhaus/img60480.npy')
  img = img[:,None]
  mu,sig = np.mean(img),np.std(img)
  img = (img-mu)/sig
  T   = _job12_horst(img,id=id)

  denoiser2d.train(T)
  res = denoiser2d.predict_raw(T,img.reshape([10,10,1,512,512]),dims="NBCYX")
  res = img*sig + mu
  res = res.astype(np.int16)
  save(res,T.config.savedir / 'pred.npy')

def _job12_horst(data,id=1):
  """
  data.data is synthetic membranes with values in {0,1}
  data.noisy_data is synthetic membranes with noisy values in [0,2]
  """
  
  cfig = denoiser2d.config(denoiser2d.eg_img_meta())

  if id==1:
    kern = np.array([[1,1,1]])
    cfig.mask = kern
    cfig.masker = denoise_utils.apply_structN2Vmask
    cfig.savedir = '../e08_horst/tt01/'
  elif id==2:
    kern = np.array([[1]])
    cfig.mask = kern
    cfig.masker = denoise_utils.apply_structN2Vmask
    cfig.savedir = '../e08_horst/tt02/'
  elif id==3:
    # kern = np.array([[1]])
    # cfig.mask = kern
    cfig.masker = denoise_utils.nearest_neib_sampler
    cfig.savedir = '../e08_horst/tt03/'

  cfig.i_final      = 10**3
  cfig.bp_per_epoch = 4*10**1
  cfig.lr = 1e-4
  cfig.patch_space  = np.array([512,512])
  cfig.patch_full   = np.array([1,1,512,512])
  # cfig.norm  = lambda x: normalize3(x,2,99.6,clip=False)

  def _ltvd(config):
    td = SimpleNamespace()
    vd = SimpleNamespace()    
    td.input  = data[:95]
    td.target = data[:95]
    denoiser2d.add_meta_to_td(td)
    vd.input  = data[95:]
    vd.target = data[95:]
    denoiser2d.add_meta_to_td(vd)

    return td,vd
  cfig.load_train_and_vali_data = _ltvd

  T = denoiser2d.train_init(cfig)
  return T


history = """

# Tue Apr 14 15:35:21 2020

I'm not getting good results with every_30_min_timelapse.tiff in job10_predict_alex_retina_t33
But that is the clearest image we have?

param-free experiments are different from "one time tasks".
They need to be re-executed as we change the internal params, but they don't need to follow the standard experiment interface with wildcard params.
Ideally we could easily rerun all of them (in parallel). This is possible if we connect them properly to snakemake.

"""