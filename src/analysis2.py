from segtools.ns2dir import load,save,toarray
import numpy as np
import itertools

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

def e08v01():
  """
  make stacks that let us toggle through the variables we want to compare
  compare across: datasize within single img for a small number of 512 frames for a subset of epochs...
  """
  res  = []
  raw  = []
  loss = []
  for params in iterdims([1,10,1,1,5]):
    (p0_unet,p1_data,p2_repeat,p3_chan,p4_datasize), pid = _parse_pid(params,[1,10,1,1,5])
    if p1_data not in [1,5]: continue
    x = toarray(load(f"../expr/e08_horst/pid{pid:03d}/ta/vali_pred/")).reshape([-1,3,512,512])
    res.append(x)
    x = load(f"../expr/e08_horst/pid{pid:03d}/ta/losses.json")
    loss.append(x)
    if p4_datasize==0:
      x = load(f"../expr/e08_horst/pid{pid:03d}/vali_raw.npy")
      raw.append(x)
  res = np.array(res).reshape([2,5,10,3,512,512]) # p1,p4,epoch,vali-patch,Y,X
  raw = np.array(raw).reshape([2,1,1,3,512,512])
  loss = np.array(loss).reshape([2,5,-1])
  save(res,"../expr/e08_horst/res.npy")
  save(raw,"../expr/e08_horst/raw.npy")
  save(loss,"../expr/e08_horst/loss.npy")
  return res,raw,loss