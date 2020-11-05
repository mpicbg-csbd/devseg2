# %load ipy.py
# import torch
# from torch import nn
# import torch_models
import itertools
# from math import floor,ceil
import numpy as np
# from scipy.ndimage import label,zoom
from skimage.feature  import peak_local_max
from skimage.measure  import regionprops
# import tifffile
from pathlib import Path

# from subprocess import Popen,run
# from segtools.math_utils import conv_at_pts_multikern
# import files
# from segtools.numpy_utils import normalize3
# from segtools.render import get_fnz_idx2d
from segtools.ns2dir import load,save,flatten_sn,toarray
from types import SimpleNamespace


## one-time-tasks

def job01():
  "fly max projections"
  for p in sorted(Path("/projects/project-broaddus/rawdata/fly_isbi/fly2/Fluo-N3DL-DRO/01/").glob("*.tif")):
    print(p)
    img = load(p)
    save(img.max(0),str(p).replace("fly_isbi/","fly_isbi/mx_z/"))
    save(img.max(1),str(p).replace("fly_isbi/","fly_isbi/mx_y/"))
    save(img.max(2),str(p).replace("fly_isbi/","fly_isbi/mx_x/"))
    a,b,c = img.shape
    save(img[:a//2].max(0),str(p).replace("fly_isbi/","fly_isbi/mx_z_half/"))
    save(img[:,:b//2].max(1),str(p).replace("fly_isbi/","fly_isbi/mx_y_half/"))
    save(img[:,:,:c//2].max(2),str(p).replace("fly_isbi/","fly_isbi/mx_x_half/"))

def job02():
  "guess number of nuclei and explore sensitivity to threshold_abs."
  ## estimated at 400 s. 
  xs=np.linspace(0.06,0.6,100)
  pts=[peak_local_max(res.astype(np.float32),threshold_abs=x,exclude_border=False,footprint=np.ones((3,8,8))) for x in xs]
  return pts

def job03():
  "max proj over z on (sphere projected) tribolium data"
  for p in sorted(Path("/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/02/").glob("*.tif")):
    print(p)
    img = load(p)
    save(img.max(0),str(p).replace("trib_isbi_proj","trib_isbi_proj/mx_z/"))
    sy,sx = slice(1280,1700), slice(500,1000)
    save(img[:,sy,sx].max(0),str(p).replace("trib_isbi_proj","trib_isbi_proj/mx_z_crop/"))

def job04_downsample():
  "tribolium (full) downsample"
  for p in sorted(Path("/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/01/").glob("*.tif")):
    print(p)
    p2 = Path(str(p).replace("trib_isbi/","trib_isbi/down/"))
    if p2.exists(): continue
    img = load(p)
    print(img.shape) ## (991, 1871, 965)
    save(img[::3,::3,::3],p2)

def job05():
  "combine points into single traj file."
  for n,m in itertools.product([1,2,3,4,5,6,7],[1,2]):
    home = Path(f"../e02/t{n}/pts/Fluo-N3DH-CE/0{m}/").resolve()
    res = []
    for d in home.glob('p*'):
      print(d)
      if d.is_dir():
        res.append(load(d/'p1.npy'))
    save(res, home/'traj.pkl')

def job06():
  maxdir = Path(str(files.raw_ce_train_02[0].parent).replace('isbi/','isbi/mx/'))
  for i,fi in enumerate(files.raw_ce_train_02):
    img = load(fi)
    mx = img.max(0) #; mx *= 255/mx.max(); mx = mx.clip(0,255).astype(np.uint8)
    save(mx, maxdir / f't{i:03d}.png')
    print(i)

def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

def job08():
  p = Path("../../rawdata/A549/Fluo-C3DH-A549/02_GT/TRA/")
  allpts = []
  for name in sorted(p.glob("*.tif")):
    print(name)
    lab = load(name)
    pts = mantrack2pts(lab)
    allpts.append(pts)
  save(allpts, "/projects/project-broaddus/rawdata/A549/traj/Fluo-C3DH-A549/02_traj.pkl")
  return allpts
  
def job09(deps):
  p = deps.params
  scores = []
  # for total in deps.target:
  for r,prd,t,tid,kxy,kz in itertools.product(p.rawdirs,p.preds,p.trains,p.tid_list,p.kernxy_list,p.kernz_list):
    try:
      total = deps.name_total_scores.format(rawdir=r,isbiname=p.map1[r],pred=prd,train_set=t,tid=tid,kernxy=float(kxy),kernz=float(kz))
      tot = load(total)
      tot.kxy = kxy
      tot.kz = kz
      tot.pred=prd
      scores.append(tot)
      print(kxy,kz,tot.f1)
    except FileNotFoundError as e:
      print(e)
  save(scores,'/lustre/projects/project-broaddus/devseg_2/ex7/celegans_isbi/train1/allscores.pkl')
  return scores

def job10():
  "tribolium max projections"
  for i in range(60):
    name = f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/01/t{i:03d}.tif"
    raw = load(name)
    print(i, raw.shape) ## (991, 1871, 965)
    raw = raw.max(0)
    save(raw,name.replace("trib_isbi/","trib_isbi/mx_z/"))

def job11():
  "make 3D crops at full res of trib data"
  from segtools.point_tools import trim_images_from_pts2
  pts = load("/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/01_traj.pkl")
  newpts = []
  for i in range(60):
    name = f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/01/t{i:03d}.tif"
    raw = load(name)
    print(i, raw.shape) ## (991, 1871, 965)
    pts2,ss = trim_images_from_pts2(pts[i],border=(5,10,10))
    newpts.append(pts2)
    save(raw[ss],name.replace("trib_isbi/","trib_isbi/crops/"))
  save(newpts,"/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/01_traj_crops.pkl")

def job12():
  "max projections"
  myname = "MDA231"
  isbiname = "Fluo-C3DL-MDA231"
  for i in range(60):
    name = f"/projects/project-broaddus/rawdata/A549/Fluo-C3DH-A549/02/t{i:03d}.tif"
    raw = load(name)
    print(i, raw.shape) ## (991, 1871, 965)
    raw = raw.max(0)
    save(raw,name.replace("/A549/","/A549/mx_z/"))

def job13():
  p = Path("../../rawdata/hampster/Fluo-N3DH-CHO/02_GT/TRA/")
  allpts = []
  for name in sorted(p.glob("*.tif")):
    print(name)
    lab = load(name)
    pts = mantrack2pts(lab)
    allpts.append(pts)
  save(allpts, "/projects/project-broaddus/rawdata/hampster/traj/Fluo-N3DH-CHO/02_traj.pkl")
  return allpts

def job14():
  myname = ["psc","u373","simplus","hela","gowt1",]
  isbiname = ["PhC-C2DL-PSC","PhC-C2DH-U373","Fluo-N2DH-SIM+","Fluo-N2DL-HeLa","Fluo-N2DH-GOWT1",]
  dataset = ["01","02"]
  for i in range(10):
    m,n = np.unravel_index(i,[5,2])
    _myname = myname[m]
    _isbiname = isbiname[m]
    _dataset = dataset[n]
    p = Path(f"/projects/project-broaddus/rawdata/{_myname}/{_isbiname}/{_dataset}_GT/TRA/")
    allpts = dict()
    for name in sorted(p.glob("*.tif")):
      print(name)
      lab = load(name)
      pts = mantrack2pts(lab)
      time = int(str(name)[-7:-4])
      print(time)
      allpts[time] = pts
    save(allpts, f"/projects/project-broaddus/rawdata/{_myname}/traj/{_isbiname}/{_dataset}_traj.pkl")

isbi_datasets = [
  ("HSC",            "BF-C2DL-HSC"),
  ("MuSC",           "BF-C2DL-MuSC"),
  ("HeLa",           "DIC-C2DH-HeLa"),
  ("MSC",            "Fluo-C2DL-MSC"),
  ("A549",           "Fluo-C3DH-A549"),
  ("A549-SIM",       "Fluo-C3DH-A549-SIM"),
  ("H157",           "Fluo-C3DH-H157"),
  ("MDA231",         "Fluo-C3DL-MDA231"),
  ("GOWT1",          "Fluo-N2DH-GOWT1"),
  ("SIM+",           "Fluo-N2DH-SIM+"),
  ("HeLa",           "Fluo-N2DL-HeLa"),
  ("celegans_isbi",  "Fluo-N3DH-CE"),
  ("hampster",       "Fluo-N3DH-CHO"),
  ("SIM+",           "Fluo-N3DH-SIM+"),
  ("fly_isbi",       "Fluo-N3DL-DRO"),
  ("trib_isbi_proj", "Fluo-N3DL-TRIC"),
  ("trib_isbi",      "Fluo-N3DL-TRIF"),
  ("U373",           "PhC-C2DH-U373"),
  ("PSC",            "PhC-C2DL-PSC"),
 ]

def job15():
  import re
  dataset = ["01","02"]
  for i in range(19*2):
    m,n = np.unravel_index(i,[19,2])
    _myname, _isbiname = isbi_datasets[m]
    _dataset = dataset[n]
    p = Path(f"/projects/project-broaddus/rawdata/{_myname}/{_isbiname}/{_dataset}_GT/TRA/")
    allpts = dict()
    print(sorted(p.glob("*.tif"))[0])
    targetname = f"/projects/project-broaddus/rawdata/{_myname}/traj/{_isbiname}/{_dataset}_traj.pkl"
    if Path(targetname).exists(): continue
    for name in sorted(p.glob("*.tif")):
      print(name)
      lab = load(name)
      pts = mantrack2pts(lab)
      time = int(re.search(r"([0-9]{3,4})\.tif",str(name)).group(1))
      print(time)
      allpts[time] = pts
    save(allpts, targetname)

def job16_cropTrib(ds="01"):
  from segtools.point_tools import trim_images_from_pts2
  pts = load(f"/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/{ds}_traj.pkl")
  _pts = np.concatenate(pts,axis=0)
  pts2, ss = trim_images_from_pts2(_pts,border=(6,6,6))
  _idxs = np.cumsum([0] + [len(x) for x in pts])
  list_pts2 = [pts2[_idxs[i]:_idxs[i+1]] for i in range(len(_idxs)-1)]
  print(pts2.shape, _idxs)

  save(list_pts2, f"/projects/project-broaddus/rawdata/trib_isbi/crops/traj/Fluo-N3DL-TRIF/{ds}_traj.pkl")
  save(ss, f"/projects/project-broaddus/rawdata/trib_isbi/crops/traj/Fluo-N3DL-TRIF/{ds}_slice.pkl")

  return
  print(ss)
  a = np.prod([_s.stop-_s.start for _s in ss])
  b = np.prod(load(f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{ds}/t000.tif").shape)
  print(a,b,a/b)

  def _f(name):
    img = load(name)
    print(img.shape) ## (991, 1871, 965)
    save(img[ss],name.replace("trib_isbi/","trib_isbi/crops/"))

  for i in range(60):
    _f(f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{ds}/t{i:03d}.tif")
    _f(f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{ds}_GT/TRA/man_track{i:03d}.tif")
  import shutil
  name = f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{ds}_GT/TRA/man_track.txt"
  shutil.copy(name,name.replace("trib_isbi/","trib_isbi/crops/"))

# def job16_downsampleTrib(ds="01"):
#   from segtools.point_tools import trim_images_from_pts2
  
#   pts = load(f"/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/{ds}_traj.pkl")
#   _pts = np.concatenate(pts,axis=0)
#   pts2, ss = trim_images_from_pts2(_pts,border=(6,6,6))
#   _idxs = np.cumsum([0] + [len(x) for x in pts])
#   list_pts2 = [pts2[_idxs[i]:_idxs[i+1]] for i in range(len(_idxs)-1)]
#   print(pts2.shape, _idxs)

#   save(list_pts2, f"/projects/project-broaddus/rawdata/trib_isbi/crops/Fluo-N3DL-TRIF/{ds}_traj.pkl")
#   save(ss, f"/projects/project-broaddus/rawdata/trib_isbi/crops/Fluo-N3DL-TRIF/{ds}_slice.pkl")

#   print(ss)
#   a = np.prod([_s.stop-_s.start for _s in ss])
#   b = np.prod(load(f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{ds}/t000.tif").shape)
#   print(a,b,a/b)

#   def _f(name):
#     img = load(name)
#     print(img.shape) ## (991, 1871, 965)
#     save(img[ss],name.replace("trib_isbi/","trib_isbi/crops/"))

#   for i in range(60):
#     _f(f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{ds}/t{i:03d}.tif")
#     _f(f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{ds}_GT/TRA/man_track{i:03d}.tif")
#   import shutil
#   name = f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{ds}_GT/TRA/man_track.txt"
#   shutil.copy(name,name.replace("trib_isbi/","trib_isbi/crops/"))

