import torch
from torch import nn
import torch_models
import itertools
from math import floor,ceil
import numpy as np
from scipy.ndimage import label,zoom
from skimage.feature  import peak_local_max
from skimage.measure  import regionprops
import tifffile
from pathlib import Path

from subprocess import Popen,run
from segtools.math_utils import conv_at_pts_multikern
import files
from segtools.numpy_utils import normalize3
from segtools.render import get_fnz_idx2d
from ns2dir import load,save
from types import SimpleNamespace

## utils. generic. all datasets.

def apply_net_tiled_3d(net,img):
  """
  Applies net to image with dims Channels,Z,Y,X.
  Assume 3x or less max pooling layers => (U-net) translational symmetry with period 8px.
  """

  # borders           = [8,24,24] ## border width within each patch that is thrown away after prediction
  # patchshape_padded = [32,240,240] ## the size of the patch that we feed into the net. must be divisible by 8 or net fails.
  # patchshape        = [16,200,200] ## must be divisible by 8 to avoid artifacts.
  # stride            = [16,200,200] ## same as patchshape in this case
  # def g(n,m): return floor(n/m)*m-n ## f(n,m) gives un-padding needed for n to be divisible by m
  def f(n,m): return ceil(n/m)*m-n ## gives padding needed for n to be divisible by m

  a,b,c = img.shape[1:]
  ## padding per image
  ## calculate extra border needed for stride % 8 = 0.
  q,r,s = f(a,8),f(b,8),f(c,8) 

  ## padding per patch. must be divisible by 8.
  ZPAD,YPAD,XPAD = 8,32,32

  img_padded = np.pad(img,[(0,0),(ZPAD,ZPAD+q),(YPAD,YPAD+r),(XPAD,XPAD+s)],mode='constant')
  output = np.zeros(img.shape)

  ## coordinate for each patch. each stride must be divisible by 8.
  zs = np.r_[:a:16]
  ys = np.r_[:b:200]
  xs = np.r_[:c:200]

  for x,y,z in itertools.product(xs,ys,zs):
    qe,re,se = min(z+16,a+q),min(y+200,b+r),min(x+200,c+s)
    ae,be,ce = min(z+16,a),min(y+200,b),min(x+200,c)
    patch = img_padded[:,z:qe+2*ZPAD,y:re+2*YPAD,x:se+2*XPAD]
    patch = torch.from_numpy(patch).cuda().float()
    patch = net(patch[None])[0,:,ZPAD:-ZPAD,YPAD:-YPAD,XPAD:-XPAD].detach().cpu().numpy()
    output[:,z:ae,y:be,x:ce] = patch[:,:ae-z,:be-y,:ce-x]

  return output



## C. elegans

def all_cele_centers():
  net = net_cele_centers()
  for i in range(65,190):
    name1 = f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t{i:03d}.tif"
    name2 = str(name1).replace("rawdata/celegans_isbi/", "devseg_2/e03_celedet/test_02/pred/")
    print(name1)
    print(name2)
    assert name1 != name2
    cele_centers(net,name1,name2)

def net_cele_centers():
  net = torch_models.Unet3(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  net.load_state_dict(torch.load("/projects/project-broaddus/devseg_2/e03_celedet/test_02/m/net12.pt"))
  return net

def cele_centers(net,name1,name2):
  img = load(name1)
  img = normalize3(img,2,99.6)
  res = apply_net_tiled_3d(net,img[None])[0]
  save(res.astype(np.float16), name2)


def all_cele_points():
  filenames_in = [f"/projects/project-broaddus/devseg_2/e03_celedet/test_02/pred/Fluo-N3DH-CE/01/t{i:03d}.tif" for i in range(190)]
  filename_out = "/projects/project-broaddus/devseg_2/e03_celedet/test_02/pts/Fluo-N3DH-CE/01/traj.pkl"
  cele_points(filenames_in,filename_out)

def cele_points(filenames_in,filename_out):

  traj = []
  zcolordir = Path(filename_out.replace("pts/","zcolor/")).parent
  maxdir = Path(filename_out.replace("pts/","maxs/")).parent

  for i,file in enumerate(filenames_in):
    res = load(file).astype(np.float32)

    ## save views of this result
    zcolor = (1+get_fnz_idx2d(res>0.3)).astype(np.uint8)
    save(zcolor, zcolordir / f'p{i:03d}.png')
    mx = res.max(0); mx *= 255/mx.max(); mx = mx.clip(0,255).astype(np.uint8)
    save(mx, maxdir / f'p{i:03d}.png')

    di  = dict()
    for th,fp in itertools.product([0.1], [10,20,30]):
      pts = peak_local_max(res,threshold_abs=th,exclude_border=False,footprint=np.ones((3,fp,fp)))
      di[(th,fp)] = pts
    traj.append(di)
  save(traj,filename_out)

def cele_denoise(filename_raw,filename_net,filename_out):
  outdir = Path(filename_out).parent; outdir.mkdir(exist_ok=True,parents=True)
  net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  net.load_state_dict(torch.load(filename_net))
  img = load(filename_raw)
  # img = normalize3(img,2,99.6)
  res = apply_net_tiled_3d(net,img[None])[0]
  save(res.astype(np.float16), filename_out)
  save(res.astype(np.float16).max(0), filename_out.replace('pred/','mxpred/'))


## Drosophila

def fly_centers(f_raw,f_pred):
  img = load(f_raw)
  img = img/1300
  net = torch_models.Unet3(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  net.load_state_dict(torch.load("/projects/project-broaddus/devseg_2/e04_flydet/test3/m/net30.pt"))
  res = apply_net_tiled_3d(net,img[None])[0]
  save(res.astype(np.float16),f_pred)
  save(res.astype(np.float16).max(0), f_pred.replace('pred/','mxpred/'))

def fly_pts(f_pred,f_pts=None):
  print(f_pred)
  img = load(f_pred)
  pts = peak_local_max(img.astype(np.float32),threshold_abs=0.2,exclude_border=False,footprint=np.ones((3,8,8)))
  save(pts,f_pts)

def fly_max_preds():
  for p in files.flies_pred:
    print(p)
    img = load(p).astype(np.float32)
    save(img.max(0),str(p).replace("pred/","pred_mx_z/"))
    save(img.max(1),str(p).replace("pred/","pred_mx_y/"))
    save(img.max(2),str(p).replace("pred/","pred_mx_x/"))
    a,b,c = img.shape
    save(img[:a//2].max(0),str(p).replace("pred/","pred_mx_z_half/"))
    save(zoom(img[:,:b//2].max(1), (5,1)),str(p).replace("pred/","pred_mx_y_half/"))
    save(zoom(img[:,:,:c//2].max(2), (5,1)),str(p).replace("pred/","pred_mx_x_half/"))

#### build fly NHLs for tracking

def img2nhl_fly(img,raw):
  nhl = SimpleNamespace()
  nhl.pts = peak_local_max(img.astype(np.float32),threshold_abs=.2,min_distance=6)
  nhl.imgvals = img[tuple(nhl.pts.T)]
  nhl.rawvals = raw[tuple(nhl.pts.T)]
  mask = nhl.rawvals>500
  nhl.pts = nhl.pts[mask]
  nhl.imgvals = nhl.imgvals[mask]
  nhl.label = np.arange(len(nhl.pts))+1
  return nhl

def process_all_flyimgs():
  nhls = []
  for i in range(20):
    print(files.flies_all[i])
    raw  = load(files.flies_all[i])
    pred = load(files.flies_pred[i])
    nhl  = img2nhl_fly(pred,raw)
    nhls.append(nhl)
  return nhls



## Tribolium 2D

def trib2d_centers(name1):
  net = torch_models.Unet3(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  net.load_state_dict(torch.load("/projects/project-broaddus/devseg_2/e05_trib2d/test2/m/net07.pt"))
  # for i in range(65):
  # name1 = f"/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01/t{i:03d}.tif"
  print(name1)
  img = load(name1)
  img = (img / 1800).clip(max=2400/1800)
  res = apply_net_tiled_3d(net,img[None])[0]
  save(res.astype(np.float16),name1.replace("rawdata/trib_isbi_proj/", "devseg_2/e05_trib2d/pred/"))

def trib2d_points():
  traj = []
  for i in range(65):
  # for i in [64]:
    name1 = f"/projects/project-broaddus/devseg_2/e05_trib2d/pred/Fluo-N3DL-TRIC/01/t{i:03d}.tif"
    print(name1)
    img = load(name1)
    pts = peak_local_max(img.astype(np.float32),threshold_abs=0.2,exclude_border=False,footprint=np.ones((3,5,5)))
    save(pts,name1.replace("e05_trib2d/pred/", "e05_trib2d/pts/"))
    traj.append(pts)
  save(traj,"/projects/project-broaddus/devseg_2/e05_trib2d/traj/Fluo-N3DL-TRIC/01/traj.pkl")

## trib 3D

def trib3d_centers(name1):
  net = torch_models.Unet3(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  net.load_state_dict(torch.load("/projects/project-broaddus/devseg_2/e06_trib/t03_downsize/m/net60.pt"))
  # for i in range(80):
  # name1 = f"/projects/project-broaddus/rawdata/trib_isbi/down/Fluo-N3DL-TRIF/02/t{i:03d}.tif"
  print(name1)
  img = load(name1)
  img = (img / 1800).clip(max=2400/1800)
  res = apply_net_tiled_3d(net,img[None])[0]
  save(res.astype(np.float16),name1.replace("rawdata/trib_isbi/", "devseg_2/e06_trib/pred/"))

def trib3d_points(name1):
  print(name1)
  img = load(name1)
  pts = peak_local_max(img.astype(np.float32),threshold_abs=0.2,exclude_border=False,footprint=np.ones((4,4,4)))
  save(pts,name1.replace("e06_trib/pred/", "e06_trib/pts/"))

def trib3d_joinpoints():
  traj = []
  for i in range(80):
    name1 = f"/projects/project-broaddus/devseg_2/e06_trib/pts/down/Fluo-N3DL-TRIF/02/t{i:03d}.tif"
    pts = load(name1)*3
    traj.append(pts)
  save(traj,"/projects/project-broaddus/devseg_2/e06_trib/traj/Fluo-N3DL-TRIF/02/traj.pkl")



