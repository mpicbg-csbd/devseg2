from types import SimpleNamespace
from pathlib import Path,PosixPath
from skimage import io
import tifffile
import pickle
import json
import numpy as np
import os
import re
import torch

import ipdb

def clean(s):
  "replaces arbitrary string with valid python identifier (SimpleNamespace attributes follow same rules as python identifiers)"

  s = str(s)
  ## Remove invalid characters
  s = re.sub('[^0-9a-zA-Z_]', '', s)
  ## Remove leading characters until we find a letter or underscore
  s2 = re.sub('^[^a-zA-Z_]+', '', s)
  ## fix pure-numbers dirnames "01" → "d01"
  s = s2 if s2 else "d"+s
  return s

known_filetypes = ['.npy', '.png', '.tif', '.pkl',] # '.json',]
known_scalars = [int,float,str,Path,PosixPath]
known_collections = [dict, set, list]
known_array_collection = [np.ndarray, torch.Tensor]

def _is_scalar(x):
  if type(x) in known_scalars: return True
  if type(x) in known_array_collection and x.ndim==0: return True
  return False

def _is_collection(x):
  if type(x) in known_collections: return True
  elif type(x) in known_array_collection and x.ndim > 0: return True
  return False

def save(d, base):
  
  base = Path(base).resolve()

  if base.suffix in known_filetypes and _is_collection(d):
    _save_file(base.parent, base.stem, d)
    return

  assert type(d) is SimpleNamespace

  scalars = SimpleNamespace()
  for k,v in d.__dict__.items():

    if _is_collection(v): _save_file(base,k,v)
    
    elif type(v) is SimpleNamespace:
      save(v,base/str(k))
    
    else:
      assert _is_scalar(v)
      scalars.__dict__[k] = v
      # print("Scalar key,val: ", k, v)

  pickle.dump(scalars,open(base/"scalars.pkl",'wb'))

def _save_file(dir,k,v):
  dir = Path(dir); dir.mkdir(parents=True,exist_ok=True)

  if type(v) is torch.Tensor: v = v.numpy()

  if type(v) is np.ndarray and v.dtype == np.uint8 and (v.ndim==2 or (v.ndim==3 and v.shape[2] in [3,4])):
    io.imsave(dir/(str(k)+'.png'),v)
  elif type(v) is np.ndarray:
    file = str(dir/(str(k)+".tif"))
    tifffile.imsave(file,v,compress=0)
    # np.save(dir/(str(k)+'.npy'),v)
  elif type(v) in known_collections:
    pickle.dump(v,open(dir/(str(k)+'.pkl'),'wb'))
    # try:
    #   json.dump(v,open(dir/(str(k)+'.json'),'w'))
    # except:
    #   os.remove(dir/(str(k)+'.json'))
    #   pickle.dump(v,open(dir/(str(k)+'.pkl'),'wb'))

def load(base,filtr='.'):
  res  = dict()
  base = Path(base).resolve()

  if base.is_file(): return _load_file(base) ## ignore filter

  for d in base.iterdir():
    d2 = clean(d.stem)
    if d2 in res.keys(): print("double assignment: ", str(d), d2)

    if d.is_dir():
      obj = load(d,filtr=filtr)
      if len(obj.__dict__)>0:
        res[d2] = obj

    if d.is_file() and d.suffix in known_filetypes and re.search(filtr,str(d)):

      obj = _load_file(d)

      if d.name=="scalars.pkl":
        for k,v in obj.__dict__.items():
          res[k] = v
      else:
        res[d2] = obj

  return SimpleNamespace(**res)

def _load_file(f):
  f = Path(f)
  if f.suffix=='.npy':
    return np.load(f)
  if f.suffix=='.png':
    return np.array(io.imread(f))
  if f.suffix in ['.tif','.tiff']:
    return tifffile.imread(str(f))
  if f.suffix=='.pkl':
    return pickle.load(open(f,'rb'))
  if f.suffix=='.json':
    return json.load(open(f,'r'))
