import numpy as np
from math import floor,ceil


def _traintimes_cpnet(info):
  t0,t1 = info.start, info.stop
  N = t1 - t0
  # pix_needed   = time_total * np.prod(batch_shape)
  # pix_i_have   = N*info.shape
  # cells_i_have = len(np.concatenate(ltps))
  # Nsamples = int(N**0.5)
  # if info.ndim==2: Nsamples = 2*Nsamples
  # Nsamples = 5 if info.ndim==3 else min(N//5,100)
  Nsamples = min(N//5, 100)
  gap = ceil(N/Nsamples)
  # dt = gap//2
  # train_times = np.r_[t0:t1:gap]
  # vali_times  = np.r_[t0+dt:t1:3*gap]
  train_times = np.r_[t0:t1:gap]

  # pred_times  = np.r_[t0:t1]
  # assert np.in1d(train_times,vali_times).sum()==0
  # return train_times, vali_times, pred_times
  return train_times

def cpnet_experiment_modifications(CPNet):
  CPNet.zoom = (1,1,1) if p0 in [0,1,2,3] else (1,0.5,0.5) # v06
  # v03 ONLY
  kernel_size = 256**(p1/49) / 4
  CPNet.kern = np.array(kernel_size)*[1,1]
  CPNet.nms_footprint = [5,5] #np.ones(().astype(np.int))
  # v04 ONLY
  kernel_size = 4.23 ##
  CPNet.kern = np.array(kernel_size)*[1,1]
  CPNet.nms_footprint = [5,5] #np.ones(().astype(np.int))
  CPNet.extern.noise_level = (p1/49)*20 #if p0==0 else (p1/49)*20

def add_centerpoint_noise(data):
  """
  data is List of SimpleNamespace with `pts` attribute
  """
  for d in data:
    # x = ((np.random.rand(*d.pts.shape) - 0.5)*noise_level*2).astype(np.int) 
    # ipdb.set_trace()
    # _sh = 10000,2
    ## floors towards zero
    _sh = d.pts.shape
    x = np.random.randn(*_sh)
    x = x / np.linalg.norm(x,axis=1)[:,None]
    r = np.random.rand(_sh[0])**(1/_sh[1]) * noise_level
    x = (x*r[:,None]).astype(np.int)
    d.pts += x

