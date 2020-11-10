from segtools.ns2dir import load,save,toarray
import numpy as np
import itertools

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

from glob import glob
import re
from subprocess import run
from pathlib import Path
from collections import defaultdict

def e19_tracking():

  res = np.full([3,19,2,2],-1.0)

  for name in glob("../expr/e19_tracking/v01/pid*/*TRA.txt"):
    print(name)
    pid,ds = re.search(r'pid(\d+)/(0[12])_TRA\.txt',name).groups()
    pid = int(pid)
    (p0,p1,p2),pid = _parse_pid(pid,[3,19,2])
    p3 = {'01':0,'02':1}[ds]
    m = re.search(r'TRA measure: (\d\.\d+)', open(name,'r').read())
    if m:
      res[p0,p1,p2,p3] = float(m.group(1))

  redo = np.array(np.where((res==[-1,-1]).sum(-1)!=0))
  res  = res.transpose([0,2,1,3]).reshape([3*2,19,2])[[0,1,2,4,5]]
  save(res,"../expr/e19_tracking/v01/res.npy")

  return res,redo

def e19(pids):
  for pid in pids:
    (p0,p1,p2), pid = _parse_pid(pid,[3,19,2])
    # name = f"slurm/e19_pid{pid:03d}.out"
    name = f"../expr/e19_tracking/v01/pid{pid:03d}/02_TRA.txt"
    print('\n\n',pid, name,'\n')
    run(f"cat {name}",shell=1)



# { 5, 6, 7, 12, 13, 16,}