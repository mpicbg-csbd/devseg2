from segtools.ns2dir import load,save,toarray
import numpy as np


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
  for i in np.r_[:150]:
    params = np.unravel_index(i,dims[:-1])
    try:
      _x = load(f"/projects/project-broaddus/devseg_2/expr/e14_celegans/v06/pid{i:03d}/scores01.pkl")
      res[params] = [x[10].f1 for x in _x]
    except:
      print(f"Missing: {i}")
  r = np.stack([x.mean(-1) for x in np.array_split(res,[20,60],-1)],axis=-1)
  save(r,"/projects/project-broaddus/devseg_2/expr/analysis/e14v06.npy")

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
