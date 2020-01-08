import torch
from torch import nn
import torch_models
import itertools
from math import floor,ceil
import numpy as np
from scipy.ndimage import label
from skimage.feature  import peak_local_max
from skimage.measure  import regionprops
import tifffile

from segtools.numpy_utils import normalize3
from segtools.render import get_fnz_idx2d
from ns2dir import load,save


## utils. generic.

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

## 


def centers(filename_raw,filename_net,filename_out):
  outdir = Path(filename_out).parent; outdir.mkdir(exist_ok=True,parents=True)
  net = torch_models.Unet3(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  net.load_state_dict(torch.load(filename_net))
  img = load(filename_raw)
  img = normalize3(img,2,99.6)
  res = predict.apply_net_tiled_3d(net,img[None])[0]
  save(res.astype(np.float16), filename_out,)


def points(filenames_in,filename_out):

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

def denoise(filename_raw,filename_net,filename_out):
  outdir = Path(filename_out).parent; outdir.mkdir(exist_ok=True,parents=True)
  net = torch_models.Unet2(16,[[1],[1]],finallayer=nn.Sequential).cuda()
  net.load_state_dict(torch.load(filename_net))
  img = load(filename_raw)
  # img = normalize3(img,2,99.6)
  res = apply_net_tiled_3d(net,img[None])[0]
  save(res.astype(np.float16), filename_out)
  save(res.astype(np.float16).max(0), filename_out.replace('pred/','mxpred/'))