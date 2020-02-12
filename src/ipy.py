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
import files
# from segtools.numpy_utils import normalize3
# from segtools.render import get_fnz_idx2d
from segtools.ns2dir import load,save,flatten_sn,toarray
# from types import SimpleNamespace


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
  for p in sorted(Path("/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/02/").glob("*.tif")):
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

def job07():
  "explore c. elegans nlm denoising"

def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

def job08():
  p = Path("../../rawdata/fly_isbi/Fluo-N3DL-DRO/02_GT/TRA/")
  allpts = []
  for name in sorted(p.glob("*.tif")):
    print(name)
    lab = load(name)
    pts = mantrack2pts(lab)
    allpts.append(pts)
  return allpts

def isbi_make_dataset_gt():
  home = Path("/lustre/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/01_GT/TRA/")
  allpts = []
  for name in sorted(home.glob("man_track*.tif")):
    print(name)
    lab = load(name)
    pts = mantrack2pts(lab)
    allpts.append(pts)
  save(allpts,"/lustre/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/01_traj.pkl")
  