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

# from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
from segtools.point_tools import trim_images_from_pts2
import matplotlib

savedir_global = Path("/projects/project-broaddus/devseg_2/expr/")


# from subprocess import Popen,run
# from segtools.math_utils import conv_at_pts_multikern
# import files
# from segtools.numpy_utils import normalize3
# from segtools.render import get_fnz_idx2d
from segtools.ns2dir import load,save,flatten_sn,toarray
from types import SimpleNamespace
from segtools.point_tools import trim_images_from_pts2
import shutil
import re
from experiments_common import iterdims

from segtools.numpy_utils import normalize3
from matplotlib import pyplot as plt

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
  ("U373",           "PhC-C2DH-U373"),
  ("PSC",            "PhC-C2DL-PSC"),
  ("trib_isbi",      "Fluo-N3DL-TRIF"),
 ]

def job15():
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

  old_dir = Path(f"/projects/project-broaddus/rawdata/trib_isbi/")
  new_dir = Path(f"/projects/project-broaddus/rawdata/trib_isbi/crops")
  
  ## find slice and resave ltps / ss
  pts  = load(old_dir / f"traj/Fluo-N3DL-TRIF/{ds}_traj.pkl")
  _pts = np.concatenate(pts,axis=0)
  pts2, ss = trim_images_from_pts2(_pts,border=(6,6,6))
  _idxs = np.cumsum([0] + [len(x) for x in pts])
  list_pts2 = [pts2[_idxs[i]:_idxs[i+1]] for i in range(len(_idxs)-1)]
  # save(list_pts2, new_dir / f"traj/Fluo-N3DL-TRIF/{ds}_traj.pkl")
  # save(ss, new_dir / f"traj/Fluo-N3DL-TRIF/{ds}_slice.pkl")

  ## how much space do we save by cropping?
  print(ss)
  a = np.prod([_s.stop-_s.start for _s in ss])
  b = np.prod(load(old_dir / f"Fluo-N3DL-TRIF/{ds}/t000.tif").shape)
  print(a,b,a/b)

  ## crop and resave images,
  def _f(name):
    img = load(old_dir / name)
    print(img.shape) ## (991, 1871, 965)
    save(img[ss],new_dir / name)
  for i in range(60,len(pts)):
    _f(f"Fluo-N3DL-TRIF/{ds}/t{i:03d}.tif")
    _f(f"Fluo-N3DL-TRIF/{ds}_GT/TRA/man_track{i:03d}.tif")

  ## direct copy man_track.txt
  # name = f"Fluo-N3DL-TRIF/{ds}_GT/TRA/man_track.txt"
  # shutil.copy(old_dir / name, new_dir / name)

def job16_downscaleTrib(ds="01"):

  old_dir = Path(f"/projects/project-broaddus/rawdata/trib_isbi/crops") #traj/Fluo-N3DL-TRIF/{ds}_traj.pkl")
  new_dir = Path(f"/projects/project-broaddus/rawdata/trib_isbi/crops_2xDown")
  
  ## find slice and resave ltps / ss
  pts  = load(old_dir / f"traj/Fluo-N3DL-TRIF/{ds}_traj.pkl")
  pts = [x//2 for x in pts]
  save(pts, new_dir / f"traj/Fluo-N3DL-TRIF/{ds}_traj.pkl")

  ## downscale resave images,
  def _f(name):
    img = load(old_dir / name)
    print(name, img.shape) ## (991, 1871, 965)
    save(img[::2,::2,::2],new_dir / name)
  for i in range(60,len(pts)):
    _f(f"Fluo-N3DL-TRIF/{ds}/t{i:03d}.tif")
    _f(f"Fluo-N3DL-TRIF/{ds}_GT/TRA/man_track{i:03d}.tif")

  ## direct copy man_track.txt
  name = f"Fluo-N3DL-TRIF/{ds}_GT/TRA/man_track.txt"
  shutil.copy(old_dir / name, new_dir / name)


def job17_ISBI_projections():
  savedir = savedir_global / "dataviews"
  dataset = ["01","02"]

  cm_r = np.random.rand(256,3)
  cm_r = (cm_r+0.2).clip(max=1.0)
  cm_r[0] = (0,0,0)
  cm_r = matplotlib.colors.ListedColormap(cm_r)

  def cmrand(lab):
    m = lab==0
    x = lab%255
    x += 1
    x[m] = 0
    x = cm_r(x)
    return x
  
  for i in range(19*2):
    p0,p1 = np.unravel_index(i,[19,2])
    _myname, _isbiname = isbi_datasets[p0]
    _dataset = dataset[p1]
    info = get_isbi_info(_myname,_isbiname,_dataset)
    print(_myname,_isbiname,_dataset)
    # p = Path(f"/projects/project-broaddus/rawdata/{_myname}/{_isbiname}/{_dataset}_GT/TRA/")
    # if "TRIC" not in _isbiname: continue

    if info.ndim==2:
      for p2,p3 in iterdims([2,2]):
        p2,_t = [("start",info.start), ("stop",info.stop-1)][p2]
        p3,_name = [("raw",info.raw_full.format(time=_t))
                  ,("lab",info.lab_full.format(time=_t))][p3]
        img = load(_name)
        if p3=="raw":
          img = normalize3(img,0,100)
          img = (plt.cm.gray(img)*255).astype(np.uint8)
        else:
          img = (cmrand(img)*255).astype(np.uint8)
        save(img, savedir / (_isbiname + f"_t-{_t}_d{_dataset}_{p3}.png"))
        # print(savedir / (_isbiname + f"_t-{_t}_{p3}.tif"))
    elif info.ndim==3:
      for p2,p3,p4 in iterdims([2,2,3]):
        p2,_t = [("start",info.start), ("stop",info.stop-1)][p2]
        p3,_name = [("raw",info.raw_full.format(time=_t))
                  ,("lab",info.lab_full.format(time=_t))][p3]
        _X = ["Z","Y","X"][p4]
        img  = load(_name)
        if "TRI" in _isbiname or "DRO" in _isbiname:
          traj = load(f"/projects/project-broaddus/rawdata/{_myname}/traj/{_isbiname}/{_dataset}_traj.pkl")
          pts2,ss = trim_images_from_pts2(traj[_t])
          img = img[ss]
        img = img.max(p4)
        if p3=="raw":
          img = normalize3(img,0,100)
          img = (plt.cm.gray(img)*255).astype(np.uint8)
        else:
          img = (cmrand(img)*255).astype(np.uint8)
        save(img, savedir / (_isbiname + f"_t-{_t}_d{_dataset}_{p3}_max{_X}.png"))
        # print(savedir / (_isbiname + f"_t-{_t}_{p3}_max{_X}.tif"))


import zarr
from glob import glob
from time import time
from tifffile import imread
import os

def conver_dirtree(root):
  for _dir,_subdirs,_files in os.walk(root):
    _dir = Path(_dir)
    fs = sorted(_files)
    # if len(fs)>2: continue
    for _f in fs: #[fs[0],fs[-1]]:
      if _f.endswith(".tif"):
        name = _dir / _f
        newname = name.with_suffix(".zarr")
        newname = str(newname).replace("rawdata","rawdata/zarr")
        print(name)
        print(newname)
        x = load(name)
        zarr.save_array(str(newname),x)

def job18_convrt_to_zarr():
  # [6,11,12,13,14,15,18]
  for i in [0,1,2,3,4,5,7,8,9,10,16,17]:
    # for j in [0,1]:
    myname,isbiname = isbi_datasets[i]
    # dataset = ["01","02"][j]
    root = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/"
    conver_dirtree(root)

def job19_compare_tif_zarr_timings():
  tiff_filenames = sorted(glob(f"/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01/*.tif"))
  zarr_filenames = [x.replace("fly_isbi","fly_isbi/zarr").replace(".tif",".zarr") for x in tiff_filenames]

  def open_file_and_mean(filename):
    if filename.endswith(".tif"):
      t1 = time()
      x = imread(filename)
      # a,b,c = x.shape
      # x = x[a//4:a//2,b//4:b//2,c//4:c//2]
      # res = np.mean(x)    #.mean()
      t2 = time()
    elif filename.endswith(".zarr"):
      t1 = time()
      x = zarr.open_array(filename)
      # a,b,c = x.shape
      # x = x[a//4:a//2,b//4:b//2,c//4:c//2]
      # res = np.mean(x)
      t2 = time()

    return (t2-t1)

  tiff_times = np.array([open_file_and_mean(x) for x in tiff_filenames])
  zarr_times = np.array([open_file_and_mean(x) for x in zarr_filenames])

  print()

  print("----------")
  print("TIFF stats:")
  mean = np.mean(tiff_times)
  std = np.std(tiff_times)
  print(f"mean: {mean:6f}")
  print(f"stddev: {std:6f}")

  print("----------")
  print("ZARR stats:")
  mean = np.mean(zarr_times)
  std = np.std(zarr_times)
  print(f"mean: {mean:6f}")
  print(f"stddev: {std:6f}")





