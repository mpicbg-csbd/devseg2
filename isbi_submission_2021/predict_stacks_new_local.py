# from skimage.io import imread
# from expand_labels_scikit import expand_labels
# from utils import expand_labels
import ipdb

from glob import glob
from math import floor,ceil
from pathlib import Path
from subprocess import run
from time import time
from types import SimpleNamespace
import argparse
# import ipdb
import numpy as np
import pickle
import re

import sys
from scipy.ndimage import zoom,label
from skimage.feature  import peak_local_max
from tifffile import imread, imsave

import torch
from os import path
# from tqdm import tqdm
import tracking
import torch_models


def run_slurm(isbiname='Fluo-N3DL-TRIC',dataset='01'):

  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 "    ## TODO: more cores?
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = """
  cp *.py temp
  cd temp
  sbatch -J p-{name} {_resources} -o ../slurm/p-{name}.out -e ../slurm/p-{name}.err --wrap \'python -c \"import predict_stacks_new_local.py as A; A.myrun_slurm_entry(isbiname={isbiname},dataset={dataset})\"\' 
  """
  slurm = slurm.replace("{_resources}",_gpu) ## you can't partially format(), but you can replace().


  for p1 in [0]:
    for p0 in range(19):
      # if p0 in [3,6]: continue
      (p1,p0,),pid = parse_pid([p1,p0],[3,19])
      Popen(slurm.format(pid=pid),shell=True)  

## Utils

def norm_affine01(x,lo,hi):
  if hi==lo: 
    return x-hi
  else: 
    return (x-lo)/(hi-lo)

def norm_percentile01(x,p0,p1):
  lo,hi = np.percentile(x,[p0,p1])
  return norm_affine01(x,lo,hi)

def zoom_pts(pts,scale):
  """
  rescale pts to be consistent with scipy.ndimage.zoom(img,scale)
  """
  # assert type(pts) is np.ndarray
  pts = pts+0.5                         ## move origin from middle of first bin to left boundary (array index convention)
  pts = pts * scale                     ## rescale
  pts = pts-0.5                         ## move origin back to middle of first bin
  pts = np.round(pts).astype(np.uint32) ## binning
  return pts


## To perform "segmentation"

def conv_at_pts_multikern(pts,kerns,sh,func=lambda a,b:np.maximum(a,b),beyond_borders=False):
  
  if len(kerns)==0: return np.zeros(sh)
  kern_shapes = np.array([k.shape for k in kerns])
  local_coord_center = kern_shapes//2
  min_extent  = (pts - local_coord_center).min(0).clip(max=[0]*len(sh))
  max_extent  = (pts - local_coord_center + kern_shapes).max(0).clip(min=sh)
  full_extent = max_extent - min_extent
  pts2 = pts - min_extent
  
  target = np.zeros(full_extent)

  for i,p in enumerate(pts2):
    ss = se2slice(p - local_coord_center[i], p - local_coord_center[i] + kern_shapes[i])
    target[ss] = func(target[ss], kerns[i])

  # print("min extent: ", min_extent)
  # print("max extent: ", max_extent)

  A = np.abs(min_extent)
  _tmp = sh-max_extent
  B = np.where(_tmp==0,None,_tmp)
  ss = se2slice(A,B)

  if beyond_borders is True:
    target2 = target.copy()
    target2[...] = -5
    target2[ss] = target[ss]
  else:
    target2 = target[ss]

  return target2

def conv_at_pts4(pts,kern,sh,func=lambda a,b:a+b):
  "kernel is centered on pts. kern must have odd shape. sh is shape of output array."
  assert pts.ndim == 2;
  assert kern.ndim == pts.shape[1] == len(sh)
  assert 'int' in str(pts.dtype)

  kerns = [kern for _ in pts]
  return conv_at_pts_multikern(pts,kerns,sh,func)

def se2slice(s,e):
  # def f(x): return x if x is not in [0,-0] else None
  return tuple(slice(a,b) for a,b in zip(s,e))


## Tile Predictions over large image

def apply_net_tiled_nobounds(f_net,img,outchan=1,patch_shape=(512,512),border_shape=(20,20)):
  """
  apply an arbitrary function to a big image over padded, overlapping tiles without padding image boundaries.
  """

  ## Large enough patches for UNet3
  if img.ndim==4:
    # print("WARNING: DEFAULT PATCH SIZES")
    patch_shape  = (32,400,400)
    border_shape = (6,40,40)
  if img.ndim==3:
    # print("WARNING: DEFAULT PATCH SIZES")
    patch_shape  = (600,600)
    border_shape = (32,32)

  patch_shape  = np.array(patch_shape).clip(max=img.shape[1:])
  border_shape = np.array(border_shape).clip(max=patch_shape//2-1)

  container = np.zeros((outchan,) + img.shape[1:])
  g = tile_multidim(img.shape[1:],patch_shape,border_shape) ## tile generator

  for s in g:
    a = (slice(None),) + s.outer
    b = (slice(None),) + s.inner
    c = (slice(None),) + s.inner_rel
    container[b] = f_net(img[a])[c] ## apply function to patch, take middle piece and insert into container

  return container

def tile1d_predict(end,length,border,divisible=8):
  """
  The input array and target container are the same size.
  Returns a,b,c (input, container, local-patch-coords).
  For use in lines like:
  `container[inner] = f(img[outer])[inner_rel]`
  Ensures that img[outer] has shape divisible by 8 in each dim with length > 8.
  """
  inner = length-2*border
  
  ## enforce roughly equally sized "main" region with max of "length"
  # n_patches = ceil(end/(length+2*border))
  # n_patches = max(n_patches,2)

  DD = divisible

  if length >= end and end%DD==0:
    n_patches=1
    length=end ## but only used in calculating n_patches
  else:
    n_patches = max(ceil(end/(length+2*border)),2)

  # ## should be restricted only to Z-dimension... because no pooling along this dimension.
  # if end <= DD:
  #   n_patches=1
  #   length=end ## but only used in calculating n_patches

  borderpoints  = np.linspace(0,end,n_patches+1).astype(np.int)
  target_starts = borderpoints[:-1]
  target_ends   = borderpoints[1:]

  # ipdb.set_trace()

  input_starts = target_starts - border; input_starts[0]=0  ## no extra context on image border
  input_ends = target_ends + border; input_ends[-1]=end     ## no extra context on image border

  if n_patches > 1:
    ## variably sized "context" regions to ensure total input size % DD == 0.  
    _dw = input_ends-input_starts
    deltas  = np.ceil(_dw/DD)*DD - _dw
    input_starts[1:-1] -= np.floor(deltas/2).astype(int)[1:-1]
    input_ends[1:-1]   += np.ceil(deltas/2).astype(int)[1:-1]
    input_ends[0] += deltas[0]
    input_starts[-1] -= deltas[-1]  
    assert np.all((input_ends - input_starts)%DD==0)

  # ipdb.set_trace()

  relative_inner_start = target_starts - input_starts
  relative_inner_end   = target_ends  -  input_starts
  
  _input     = tuple([slice(a,b) for a,b in zip(input_starts,input_ends)])
  _container = tuple([slice(a,b) for a,b in zip(target_starts,target_ends)])
  _patch     = tuple([slice(a,b) for a,b in zip(relative_inner_start,relative_inner_end)])
  res = np.array([_input,_container,_patch,]).T
  res = [SimpleNamespace(outer=x[0],inner=x[1],inner_rel=x[2]) for x in res]
  return res

def tile_multidim(img_shape,patch_shape,border_shape=None,f_singledim=tile1d_predict):
  "Turns a 1-D tile iterator into an N-D tile iterator, potentially with different divisibility reqs for each dimension."
  "generates all the patch coords for iterating over large dims.  ## list of coords. each coords is Dx4. "

  if border_shape is None: border_shape = (0,)*len(img_shape)
  divisible = (1,8,8)[-len(border_shape):]

  r = [f_singledim(a,b,c,d) for a,b,c,d in zip(img_shape,patch_shape,border_shape,divisible)] ## D x many x 3
  D = len(r) ## dimension
  # r = np.array(product())
  r = np.array(np.meshgrid(*r)).reshape([D,-1]).T ## should be an array D x len(a) x len(b)
  def f(s): # tuple of simple namespaces
    a = tuple([x.outer for x in s])
    b = tuple([x.inner for x in s])
    c = tuple([x.inner_rel for x in s])
    return SimpleNamespace(outer=a,inner=b,inner_rel=c) ## input, container, patch
  r = [f(s) for s in r]
  return r


## Method params

def init_params(ndim):
  P = SimpleNamespace()
  if ndim==2:
    P.zoom  = (1,1) #(0.5,0.5)
    P.kern  = [5,5]
    P.patch = (256,256)
    P.border = [2,2]
    P.match_dub = 10
    P.match_scale = [1,1]
    # P.evalBorder = (5,5)

  elif ndim==3:
    P.zoom   = (1,1,1)
    P.kern   = [3,5,5]
    P.patch  = (16,128,128)
    P.border = [1,2,2]
    P.match_dub = 10
    P.match_scale = [4,1,1]
    # P.evalBorder = (1,5,5)

  P.nms_footprint = P.kern
  P.patch = np.array(P.patch)

  return P

def _init_unet_params(ndim):
  T = SimpleNamespace()
  if ndim==2:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
  elif ndim==3:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
  torch_models.init_weights(T.net)
  return T

# from time import time as pytime


"""
Run the Prediciton. Main entry point.
"""
def predict_and_save_tracking(indir,outdir,cpnet_weights,seg_weights,params,mantrack_t0=None):

  # t0 = pytime(); print(f"t0:{t0}")
  # outdir = outdir.replace("isbi_challenge_out","isbi_challenge_out_2")
  outdir = Path(outdir)

  outdir.mkdir(parents=True,exist_ok=True)
  cpnet = _init_unet_params(params.ndim).net
  # cpnet  = torch.load(cpnet_weights)
  cpnet.load_state_dict(torch.load(cpnet_weights))
  segnet = None
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cpnet  = cpnet.to(device)
  # t1 = pytime(); print(f"t1:{t1}")

  fileglob = sorted(Path(indir).glob("t*.tif"))
  assert len(fileglob)>0 , "Empty Directory"
  print(f"Running tracking over {len(fileglob)} files...\n\n",flush=True)

  if "project-broaddus" in str(outdir):
    print("Running main_loop_local()")
    tb = main_loop_local(fileglob,cpnet,segnet,params,outdir)
  else:
    print("Running main_loop_isbi()")
    tb = main_loop_isbi(fileglob,cpnet,segnet,params,outdir)

  t_start = int(re.search(r"(\d{3,4})\.tif", str(fileglob[0])).group(1))
  # t3 = pytime(); print(f"t3:{t3}")

  sampling = params.scale * np.array([0.5,1,1])[-len(params.scale):] ## extra width in Z

  if   'isbi_challenge/Fluo-N3DL-TRIF/01' in indir: o_shape = (975, 1820, 1000)
  elif 'isbi_challenge/Fluo-N3DL-TRIF/02' in indir: o_shape = (991, 1871, 965)
  else: o_shape = imread(str(fileglob[0])).astype(np.float).shape # orig shape

  # ipdb.set_trace()

  if mantrack_t0:
    lbep, labelset, stackset = tracking.save_isbi_tb_2(tb,params.radius,sampling,o_shape,t_start,params.ndim,outdir,penalizeFP='0',mantrack_t0=mantrack_t0)
  else:
    lbep, labelset, stackset = tracking.save_isbi_tb_2(tb,params.radius,sampling,o_shape,t_start,params.ndim,outdir,penalizeFP='1',mantrack_t0=None)

def main_loop_isbi(fileglob,cpnet,segnet,params,outdir):
  ltps = []
  for i,rawname in enumerate(fileglob):
    # print(f"i={i+1}/{len(fileglob)} , file={rawname} \033[F", flush=True)
    print(f"i={i+1}/{len(fileglob)} , file={rawname}", flush=True)
    pts = eval_sample(rawname,cpnet,segnet,params)
    ltps.append(pts)
  tb = tracking.nn_tracking_on_ltps(ltps, scale=params.scale, dub=params.radius*2)
  return tb

def main_loop_local(fileglob,cpnet,segnet,params,outdir):
    ## predict & extract pts for each image independently
  extrasdir = Path(str(outdir).replace("isbi_challenge_out", "isbi_challenge_out_extra"))
  
  if (extrasdir / 'ltps/ltps.npy').exists():
    ltps = np.load(str(extrasdir / 'ltps/ltps.npy'), allow_pickle=1)
  elif (outdir / 'ltps.npy').exists():
    ltps = np.load(str(outdir / 'ltps.npy'), allow_pickle=1)
  elif path.exists(path.join(str(outdir).replace('_2','') , 'ltps.npy')):
    ltps = np.load(path.join(str(outdir).replace('_2','') , 'ltps.npy'), allow_pickle=1)
  else:
    extrasdir.mkdir(parents=True,exist_ok=True)
    (extrasdir / "ltps").mkdir(exist_ok=1)
    ltps = []
    for i,rawname in enumerate(fileglob):
      print(f"i={i+1}/{len(fileglob)} , file={rawname}", flush=True)
      pts = eval_sample(rawname,cpnet,segnet,params)
      ltps.append(pts)
      np.save(str(extrasdir / 'ltps/pts.npy'), pts)
    np.save(str(extrasdir / 'ltps/ltps.npy'), np.array(ltps,dtype=object))

  tb = tracking.nn_tracking_on_ltps(ltps, scale=params.scale, dub=params.radius*2)
  return tb


"""
TODO: speed up this function. 2mins 2sec to run on TRIF shape=(975, 1820, 1000) with zoom=(0.5 , 0.5 , 0.5)
"""
def eval_sample(rawname,cpnet,segnet,params):

  raw = imread(str(rawname)).astype(np.float)
  o_shape = raw.shape ## original shape

  ## downscale raw
  _raw = zoom(raw,params.zoom,order=1)
  zoom_effective = _raw.shape / np.array(raw.shape)
  raw = _raw

  ## normalize intensity
  raw = norm_percentile01(raw,2,99.4)
  # percentiles = np.percentile(raw,np.linspace(0,100,101))

  ## run through net
  def f_net(patch):
    with torch.no_grad():
      patch = torch.from_numpy(patch).float().cuda()
      return cpnet(patch[None])[0].cpu().numpy() ## add then remove batch dimension
  pred = apply_net_tiled_nobounds(f_net,raw[None],outchan=1,)
  pred = pred[0]

  # newdir = str(rawname).replace("isbi_challenge","isbi_challenge_pred")
  # Path(newdir).parent.mkdir(parents=True,exist_ok=True)
  # imsave(newdir, pred)

  ## renormalize intensity
  _peaks = pred/pred.max()

  ## extract points from predicted image
  pts = peak_local_max(_peaks,threshold_abs=.2,exclude_border=False,footprint=np.ones(params.nms_footprint))
  yPts_small = pts.copy()

  ## undo scaling
  pts = zoom_pts(pts,1/zoom_effective)

  ## filter out points outside of Field of Interest (FoI)
  o_shape = np.array(o_shape)
  filterzone = params.evalBorder - params.radius/params.scale ## scale
  filterzone = filterzone.clip(min=0) ## in case radius > border width
  pts2    = [p for p in pts   if np.all(p%(o_shape - filterzone) >= filterzone)]

  print(f"{len(pts)} obj detected.",flush=True)
  if len(pts2) < len(pts):
    print(f"{len(pts) - len(pts2)} obj removed by Field of Interest filter.")

  return pts2

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track cells. Take a folder of TIFFs as input...')

  parser.add_argument('-i', "--indir", default="/projects/CSBDeep/ISBI/upload_test/01")
  parser.add_argument('-o', "--outdir", default="/projects/CSBDeep/ISBI/upload_test/01_RES")
  parser.add_argument("--cpnet_weights", default="models/Fluo-N2DH-GOWT1-01_weights.pt")
  parser.add_argument("--segnet_weights", default=None)
  parser.add_argument('--zoom', type=float, nargs='*', default=[1.,1.])
  parser.add_argument('--nms_footprint', type=int, nargs='*', default=[3,3])
  parser.add_argument('--radius', type=float)
  parser.add_argument('--scale', type=float, nargs='*', default=[1.,1.])
  parser.add_argument('--mantrack_t0', type=str)
  parser.add_argument('--evalBorder', type=int, nargs='*',)

  # mantrack_t0 = "None"

  # parser.add_argument('--threshold_abs',type = float, default=.3)
  # parser.add_argument('--min_distance',type = int, default=4)
  # parser.add_argument('--tile_size', type = int, nargs = 3, default=[-1,512,256])
  

  args = parser.parse_args()

  if args.mantrack_t0 == "None":
    mantrack_t0 = None
  else:
    mantrack_t0 = args.mantrack_t0


  params = SimpleNamespace()
  params.zoom = args.zoom
  params.nms_footprint = args.nms_footprint
  params.ndim = len(args.zoom)
  params.scale = np.array(args.scale)
  params.radius = args.radius
  params.evalBorder = np.array(args.evalBorder)

  predict_and_save_tracking(
    args.indir,
    args.outdir,
    args.cpnet_weights,
    args.segnet_weights,
    params,
    mantrack_t0,
    )



