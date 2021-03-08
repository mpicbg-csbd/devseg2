"""
## blocking cuda enables straightforward time profiling
export CUDA_LAUNCH_BLOCKING=1
ipython

import denoiser, detector, tracking
import networkx as nx
#import numpy_indexed as ndi
#import experiments2 as ex
import analysis2, ipy
import e22_flywing as e22

import numpy as np
from segtools.ns2dir import load,save,toarray
import e21_isbidet as e21
import e23_mauricio as e23
import e24_isbi_datagen as e24

%load_ext line_profiler
"""

# from types import SimpleNamespace


import ipdb
import itertools
from math import floor,ceil
import numpy as np
from pathlib import Path
from types import SimpleNamespace


def merge(sn1,sn2):
  sn1 = SimpleNamespace(**{**sn1.__dict__,**sn2.__dict__})
  return sn1

def savedir_global():
  return Path('/projects/project-broaddus/devseg_2/expr/')

def rchoose(x,dim=0,**kwargs):
  """improved np.random.choice()"""
  if type(x) is np.ndarray:
    n = x.shape[dim]
    i = np.random.choice(n,**kwargs)
    tup = [None,]*x.ndim; tup[dim]=i
    return x[tuple(tup)]
  else:
    n = len(x)
    i = np.random.choice(n,**kwargs)
    return x[i]

def partition(predicate, iterable):
  """from https://stackoverflow.com/questions/4578590/python-equivalent-of-filter-getting-two-output-lists-i-e-partition-of-a-list"""
  trues = []
  falses = []
  for item in iterable:
      if predicate(item):
          trues.append(item)
      else:
          falses.append(item)
  return trues, falses

def shuffle_and_split(l, valifrac=1/8):
  # slicelist = [(i,ss) for i,d in enumerate(self.data) for ss in d.slices]
  np.random.seed(0)
  np.random.shuffle(l)
  N = len(l)
  Nvali  = ceil(N*valifrac)
  Ntrain = N-Nvali
  return l[:Ntrain], l[Ntrain:]

def parse_pid(pid_or_params,dims):
  if hasattr(pid_or_params,'__len__') and len(pid_or_params)==len(dims):
    params = pid_or_params
    pid = np.ravel_multi_index(params,dims)
  elif 'int' in str(type(pid_or_params)):
    pid = pid_or_params
    params = np.unravel_index(pid,dims)
  else:
    a = hasattr(pid_or_params,'__len__')
    b = len(pid_or_params)==len(dims)
    print("ERROR", a, b)
    assert False
  return params, pid

def iterdims(shape):
  return itertools.product(*[range(x) for x in shape])





