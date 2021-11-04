# from skimage.io import imread
from expand_labels_scikit import expand_labels
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
import tifffile
import torch

# from tqdm import tqdm
import tracking
import torch_models

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
    P.evalBorder = (5,5)

  elif ndim==3:
    P.zoom   = (1,1,1)
    P.kern   = [3,5,5]
    P.patch  = (16,128,128)
    P.border = [1,2,2]
    P.match_dub = 10
    P.match_scale = [4,1,1]
    P.evalBorder = (1,5,5)

  P.nms_footprint = P.kern
  P.patch = np.array(P.patch)

  return P


## Run the Prediciton

def _init_unet_params(ndim):
  T = SimpleNamespace()
  if ndim==2:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
  elif ndim==3:
    T.net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
  torch_models.init_weights(T.net)
  return T

def predict_and_save_tracking(indir,outdir,cpnet_weights,seg_weights,params,mantrack_t0=None):

  extrasdir = Path(outdir.replace("isbi_challenge_out", "isbi_challenge_out_extra"))
  extrasdir.mkdir(parents=True,exist_ok=True)

  t0 = time()
  
  Path(outdir).mkdir(parents=True,exist_ok=True)
  cpnet = _init_unet_params(params.ndim).net
  # cpnet  = torch.load(cpnet_weights)
  cpnet.load_state_dict(torch.load(cpnet_weights))
  segnet = None
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cpnet  = cpnet.to(device)

  fileglob = sorted(Path(indir).glob("t*.tif"))
  # fileglob = fileglob[:20]  ## FIXME
  print(f"Running tracking over {len(fileglob)} files...\n\n",flush=True)
  # ipdb.set_trace()

  ## predict & extract pts for each image independently
  (extrasdir / "ltps").mkdir(exist_ok=1)
  ltps = []
  for i,rawname in enumerate(fileglob):
    print(f"i={i+1}/{len(fileglob)} , file={rawname} \033[F", flush=True)
    pts = eval_sample(rawname,cpnet,segnet,params,ptsOnly=True)
    np.save(str(extrasdir / 'ltps/pts.npy'), pts)
    ltps.append(pts)

  np.save(str(extrasdir / 'ltps/ltps.npy'), np.array(ltps,dtype=object))

  ## do tracking from pts
  radius = np.max(np.array(params.nms_footprint) / params.zoom) * 2
  print(f"Radius = {radius}")

  tb = tracking.nn_tracking_on_ltps(ltps, scale=params.scale, dub=radius*2)

  raw = imread(str(fileglob[0])).astype(np.float)
  o_shape = raw.shape ## original shape
  t_start = int(re.search(r"(\d{3,4})\.tif", str(fileglob[0])).group(1))

  print(indir)
  if "Fluo-N3DL-DRO" in indir:
    radius = 5

  savedir = outdir
  if mantrack_t0:
    lbep, labelset, stackset = tracking.save_isbi_tb_2(tb,radius,o_shape,t_start,params.ndim,savedir,penalizeFP='0',mantrack_t0=mantrack_t0)
  else:
    lbep, labelset, stackset = tracking.save_isbi_tb_2(tb,radius,o_shape,t_start,params.ndim,savedir,penalizeFP='1',mantrack_t0=None)


  # ## save masks given tracking
  # for i,rawname in enumerate(fileglob):
  #   print(f"i={i+1}/{len(fileglob)} , file={rawname} \033[F", flush=True)
  #   inname = Path(rawname).name
  #   outname = "mask" + inname[1:] ## remove the 't'
  #   seg = buildSegFromLPtsAndTB(ltps,tb,i,o_shape)
  #   seg = seg.astype(np.uint16)
  #   imsave(Path(outdir)/outname, seg)


"""
Takes list-of-timepts (ltps) , TrueBranching (tb) and current time (i)
Returns points rasterized to an image of proper size.
"""
def buildSegFromLPtsAndTB(ltps,tb,i,o_shape):
  pts = ltps[i]
  stack  = conv_at_pts4(pts,np.ones([1,]*len(o_shape)),o_shape).astype(np.uint16)
  stack  = label(stack)[0]
  radius = np.max(np.array(params.nms_footprint) / params.zoom) * 2
  print(f"Old size: 25 , new size : {radius} \n")
  stack = expand_labels(stack,radius)


"""
TODO: speed up this function. 2mins 2sec to run on TRIF shape=(975, 1820, 1000) with zoom=(0.5 , 0.5 , 0.5)
"""
def eval_sample(rawname,cpnet,segnet,params,ptsOnly=False):

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

  ## renormalize intensity
  _peaks = pred/pred.max()

  ## extract points from predicted image
  pts = peak_local_max(_peaks,threshold_abs=.2,exclude_border=False,footprint=np.ones(params.nms_footprint))
  yPts_small = pts.copy()

  ## undo scaling
  pts = zoom_pts(pts,1/zoom_effective)

  if ptsOnly: return pts

  o_shape = np.array(o_shape)
  ## Filter out points near border
  # pts2   = [p for p in pts   if np.all(p%(o_shape - params.evalBorder) >= params.evalBorder)]
  # gtpts2 = [p for p in s.pts if np.all(p%(o_shape - params.evalBorder) >= params.evalBorder)]
  stack  = conv_at_pts4(pts,np.ones([1,]*len(o_shape)),o_shape).astype(np.uint16)
  stack  = label(stack)[0]
  radius = np.max(np.array(params.nms_footprint) / params.zoom) * 2
  print(f"Old size: 25 , new size : {radius} \n")
  stack = expand_labels(stack,radius)
  return stack




if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Track cells. Take a folder of TIFFs as input...')

  parser.add_argument('-i', "--indir", default="/projects/CSBDeep/ISBI/upload_test/01")
  parser.add_argument('-o', "--outdir", default="/projects/CSBDeep/ISBI/upload_test/01_RES")
  parser.add_argument("--cpnet_weights", default="models/Fluo-N2DH-GOWT1-01_weights.pt")
  parser.add_argument("--segnet_weights", default=None)
  parser.add_argument('--zoom', type=float, nargs='*', default=[1.,1.])
  parser.add_argument('--nms_footprint', type=int, nargs='*', default=[3,3])
  parser.add_argument('--scale', type=float, nargs='*', default=[1.,1.])
  parser.add_argument('--mantrack_t0', type=str)

  # mantrack_t0 = "None"

  # parser.add_argument('--threshold_abs',type = float, default=.3)
  # parser.add_argument('--min_distance',type = int, default=4)
  # parser.add_argument('--tile_size', type = int, nargs = 3, default=[-1,512,256])
  

  args = parser.parse_args()

  if args.mantrack_t0 == "None":
    mantrack_t0 = None
  else:
    mantrack_t0 = args.mantrack_t0

  # ipdb.set_trace()


  params = SimpleNamespace()
  params.zoom = args.zoom
  params.nms_footprint = args.nms_footprint
  params.ndim = len(args.zoom)
  params.scale = args.scale

  predict_and_save_tracking(
    args.indir,
    args.outdir,
    args.cpnet_weights,
    args.segnet_weights,
    params,
    mantrack_t0,
    )



