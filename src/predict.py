import torch
import numpy as np
from math import floor,ceil
import itertools


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