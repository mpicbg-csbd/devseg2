from segtools.ns2dir import load,save,toarray
import numpy as np
import itertools

from glob import glob
import re
from subprocess import run
from pathlib import Path
from collections import defaultdict


def _parse_pid(pid_or_params,dims):  
  if hasattr(pid_or_params,'__len__') and len(pid_or_params)==len(dims):
    params = pid_or_params
    pid = np.ravel_multi_index(params,dims)
  elif 'int' in str(type(pid_or_params)):
    pid = pid_or_params
    params = np.unravel_index(pid,dims)
  return params, pid

def iterdims(shape):
  return itertools.product(*[range(x) for x in shape])



def job11_sm_analysis():
  signal = load('/projects/project-broaddus/rawdata/synth_membranes/gt.npy')[:6000:10]
  noise  = load(savedir / 'e07_synmem/v2_t01/noise.npy')[:6000:10]
  pred   = load(savedir / 'e07_synmem/v2_t01/pred.npy')
  diff   = signal + noise - pred
  print(signal.mean(), noise.mean(), pred.mean(), diff.mean())
  """
  most of the errors that i see are on membranes along the _horizontal_, i.e. the axis of the noise.
  This is totally 
  """

def e14v05():
  dims = [4,3,5,190]
  res = np.zeros(dims)
  for i in np.r_[:60]:
    params = np.unravel_index(i,dims[:-1])
    try:
      _x = load(f"/projects/project-broaddus/devseg_2/expr/e14_celegans/v05/pid{i:02d}/scores01.pkl")
      res[params] = [x[10].f1 for x in _x]
    except:
      print(f"Missing: {i}")
  # r = np.stack([x.mean(-1) for x in np.array_split(res,[20,60],-1)],axis=-1)
  save(res,"/projects/project-broaddus/devseg_2/expr/analysis/e14v05.npy")

def e14v06():
  dims = [30,1,5,190]
  res = np.zeros(dims)
  for params in iterdims(dims[:3]):
    # params = np.unravel_index(i,dims[:-1])
    i = np.ravel_multi_index(params,dims[:-1])
    try:
      _x = load(f"/projects/project-broaddus/devseg_2/expr/e14_celegans/v06/pid{i:03d}/scores01.pkl")
      res[params] = [x[10].f1 for x in _x]
    except:
      print(f"Missing: {params} = {i}")
  # r = np.stack([x.mean(-1) for x in np.array_split(res,[20,60],-1)],axis=-1)
  save(res,"/projects/project-broaddus/devseg_2/expr/analysis/e14v06.npy")

def e14v07():
  dims = [30,1,5,190]
  res = np.zeros(dims)
  for i in np.r_[:150]:
    params = np.unravel_index(i,dims[:-1])
    try:
      _x = load(f"/projects/project-broaddus/devseg_2/expr/e14_celegans/v07/pid{i:03d}/scores01.pkl")
      res[params] = [x[10].f1 for x in _x]
    except:
      print(f"Missing: {i}")
  r = np.stack([x.mean(-1) for x in np.array_split(res,[20,60],-1)],axis=-1)
  save(r,"/projects/project-broaddus/devseg_2/expr/analysis/e14v07.npy")

def e14v08():
  dims = [10,190]
  res = np.zeros(dims)
  totals = np.zeros(10)
  for i, in iterdims([10]):
    # try:
    _x = load(f"/projects/project-broaddus/devseg_2/expr/e14_celegans/v08/pid{i:03d}/scores01.pkl")
    res[i] = [x[10].f1 for x in _x]
    _tot = np.array([[2*x[10].n_matched,x[10].n_proposed+x[10].n_gt] for x in _x])
    _tot = _tot.sum(0)
    totals[i] = _tot[0] / _tot[1]
    print(i, totals[i])
    # except:
    #   print(f"Missing: {i}")
  # r = np.stack([x.mean(-1) for x in np.array_split(res,[20,60],-1)],axis=-1)
  save(res,"/projects/project-broaddus/devseg_2/expr/analysis/e14v08.npy")

def e08_analyze():
  """
  make stacks that let us toggle through the variables we want to compare
  compare across: datasize within single img for a small number of 512 frames for a subset of epochs...
  """
  res  = []
  raw  = []
  loss = []
  for params in iterdims([1,10,5,1,1]):
    (p0_unet,p1_data,p2_repeat,p3_chan,p4_datasize), pid = _parse_pid(params,[1,10,5,1,1])
    if p2_repeat != 0: continue
    if p1_data not in [1,5]: continue
    x = toarray(load(f"../expr/e08_horst/v02/pid{pid:03d}/ta/vali_pred/")).reshape([-1,3,512,512])
    res.append(x)
    x = load(f"../expr/e08_horst/v02/pid{pid:03d}/ta/losses.json")
    loss.append(x)
    if p2_repeat==0:
      x = load(f"../expr/e08_horst/v02/pid{pid:03d}/vali_raw.npy")
      raw.append(x)
  res = np.array(res).reshape([2,5,10,3,512,512]) # p1,p4,epoch,vali-patch,Y,X
  raw = np.array(raw).reshape([2,1,1,3,512,512])
  loss = np.array(loss).reshape([2,5,-1])
  save(res,"../expr/e08_horst/v02/res.npy")
  save(raw,"../expr/e08_horst/v02/raw.npy")
  save(loss,"../expr/e08_horst/v02/loss.npy")
  return res,raw,loss

def e08_res():
  res = []
  import experiments2
  for i in range(10):
    name = experiments2.horst_data[i]
    x = load(name)
    print(x.shape)
    res.append(x[0])
    
    resname = name.replace("HorstObenhaus/","HorstObenhaus/pred/v02/pred_")
    x = load(resname)
    print(x.shape)
    res.append(x[0,0])
  res = np.array(res).reshape([10,2,512,512])
  save(res,"../expr/e08_horst/v02/final.npy")

def e18_isbidet():

  """
  added v02.
  """

  res = dict()
  # res[1,:,1]=-2
  ## -1 = missing, -2 = not supposed to exist

  for name in glob("../expr/e18_isbidet/v02/pid*/*.pkl"):
    print(name)
    m = re.search(r'pid(\d+)/ltps_(0[12])\.pkl',name)
    if not m: continue
    pid,_ds = m.groups()
    (p0,p1),pid = _parse_pid(int(pid),[2,19])
    res[(p0,p1)] = load(name)
    # m = re.search(r'(DET|TRA) measure: (\d\.\d+)', open(name,'r').read())
    # if m: res[p0,p1,p2,p3,p4] = float(m.group(2))

  # redo = np.array(np.where((res==[-1,-1]).sum(-1)!=0))
  # res  = res.transpose([0,2,1,3]).reshape([4*2,19,2])[[0,1,2,4,5]]
  # save(res,"../expr/e19_tracking/v02/res.npy")

  return res

def e19_tracking():
  """
  added v02. 
  now v03.
  v04: tracks e19_v04 which tracks e18 v03...
  v05: -> e19_v05 -> e21_v01
  v06: e21_v02
  """

  # isbiID, 01/02, flat/content, random, DET/TRA
  res = np.full([19,2,2,5,2],np.nan)

  for name in sorted(glob("../expr/e21_isbidet/v02/pid*/*.txt")):
    print(name)
    m = re.search(r'pid(\d+)/(0[12])_(TRA|DET)\.txt',name)
    if not m: continue
    pid,dataset,tradet = m.groups()
    params,pid = _parse_pid(int(pid),[19,2,2,5])
    m2 = re.search(r'(DET|TRA) measure: (\d\.\d+)', open(name,'r').read())
    if not m2: continue
    p4 = ['DET','TRA'].index(tradet)
    idx = tuple(params) + (p4,)
    res[idx] = float(m2.group(2))
    
  # redo = np.array(np.where((res==[-1,-1]).sum(-1)!=0))
  # res  = res.transpose([0,2,1,3]).reshape([4*2,19,2])[[0,1,2,4,5]]
  save(res,"../expr/e19_tracking/v06/res.npy")

  return res #,redo

def e19_showerrors():
  """
  v02.
  """
  for name in sorted(glob("../expr/e19_tracking/v02/*/*TRA.txt")):
    ps,pid = _parse_pid(int(Path(name).parts[-2][3:]),[4,19,2,2,])
    print(Path(name).parts[-2:], ps, end='\t',flush=True)
    run(f"cat {name}",shell=1)
    continue
    m = re.search(r'pid_(\d)_(\d\d)_(\d)_(\d)/(0[12])_(TRA|DET)\.txt',name)
    # m = re.search(r'pid(\d{3})/(0[12])_(TRA|DET)\.txt',name))
    if not m: continue
    # p0,p1,p2,p3,ds,tra_or_det = m.groups()
    # (p0,p1,p2,p3),pid = _parse_pid(int(pid),[4,19,2,2])
    m2 = re.search(r'(DET|TRA) measure: (\d\.\d+)', open(name,'r').read())
    if m2 and float(m.group(2))<0.8:
      # print(pid,(p0,p1,p2,p3),_ds,p4)
      print(m.groups())
      run(f"cat {name}",shell=1)


