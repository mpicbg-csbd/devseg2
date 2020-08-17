import numpy as np
import ipdb
from segtools.math_utils import conv_at_pts4
import itertools

def nearest_neib_masker(x,yt,T):
  sh = x.shape[2:]
  ndim = len(sh)
  frac = 0.01
  pts  = (np.random.rand(int(np.prod(sh)*frac),ndim)*sh).astype(int)
  deltas = np.random.randint(0,3,pts.shape)-1
  neibs  = pts + deltas
  for i in range(ndim):
    neibs[:,i] = neibs[:,i].clip(min=0,max=sh[i]-1)

  yt = x.copy()
  ss_pts = (0,0,) + tuple(pts.T[i] for i in range(ndim))
  ss_neibs = (0,0,) + tuple(neibs.T[i] for i in range(ndim))
  x[ss_pts] = x[ss_neibs]
  _w = np.zeros(x.shape)
  _w[ss_pts] = 1

  return x,yt,_w

def structN2V_masker(x,yt,T):
  """
  each point in coords corresponds to the center of the mask.
  then for point in the mask with value=1 we assign a random value
  """
  mask = T.config.mask
  ndim = mask.ndim
  center = np.array(mask.shape)//2
  ## leave the center value alone
  mask[tuple(center.T)] = 1
  w = np.zeros(x.shape)
  
  for i,j in itertools.product(range(x.shape[0]), range(x.shape[1])):
    xij = x[i,j]
    wij = w[i,j]
    coords = (np.random.rand(int(xij.size * 0.01), ndim) * xij.shape).astype(np.int).T

    ## displacements from center
    dx = np.indices(mask.shape)[:,mask==1] - center[:,None]
    ## combine all coords (ndim, npts,) with all displacements (ncoords,ndim,)
    mix = (dx.T[...,None] + coords[None])
    mix = mix.transpose([1,0,2]).reshape([ndim,-1]).T
    ## stay within patch boundary
    mix = mix.clip(min=np.zeros(ndim),max=np.array(xij.shape)-1).astype(np.uint)
    ## replace neighbouring pixels with random values from flat dist
    xij[tuple(mix.T)]  = np.random.rand(mix.shape[0])*4 - 2
    wij[tuple(coords)] = 1
    x[i,j] = xij
    w[i,j] = wij

    # patch[tuple(coords)] = 2

  return x,yt,w

def footprint_masker(x,yt,T):
  # patch_space = x.shape
  kern = T.config.mask
  ma = mask_from_footprint(x.shape[2:],kern,)
  # print(ma.shape)
  ma = ma[None,None] ## samples , channels
  yt = x.copy()
  x[ma>0] = np.random.rand(*x.shape)[ma>0]
  w = (ma==2).astype(np.float32)
  return x,yt,w

def mask_from_footprint(sh,kern,frac=0.01):
  # takes a shape sh. returns random mask with that shape
  pts = (np.random.rand(int(np.prod(sh)*frac),3)*sh).astype(int)
  target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
  target = target.astype(np.uint8)
  return target

def mask_2(patch_size,frac):
  "build random mask for small number of central pixels"
  n = int(np.prod(patch_size) * frac)
  kern = np.zeros((19,3,3)) ## must be odd
  kern[:,1,1] = 1
  kern[9] = 1
  kern[9,1,1] = 1
  mask = np.random.rand(*patch_size)<frac
  indices = np.indices(patch_size)[:,mask]
  deltas  = np.indices(kern.shape)[:,kern==1]
  newmask = np.zeros(patch_size)
  for dx in deltas.T:
    inds = (indices+dx[:,None]).T.clip(min=[0,0,0],max=np.array(patch_size)-1).T
    newmask[tuple(inds)] = 1

def sparse_3set_mask(d):
  "build random mask for small number of central pixels"
  n = int(np.prod(d.patch_size) * d.frac)
  z_inds = np.random.randint(0,d.patch_size[0],n)
  y_inds = np.random.randint(0,d.patch_size[1],n)
  x_inds = np.random.randint(0,d.patch_size[2],n)
  ma = np.zeros(d.patch_size,dtype=np.int)
  
  for i in d.xmask:
    m = x_inds+i == (x_inds+i).clip(0,d.patch_size[2]-1)
    ma[z_inds[m], y_inds[m],x_inds[m]+i] = 1

  for i in d.ymask:
    m = y_inds+i == (y_inds+i).clip(0,d.patch_size[1]-1)
    ma[z_inds[m], y_inds[m]+i,x_inds[m]] = 1

  for i in d.zmask:
    m = z_inds+i == (z_inds+i).clip(0,d.patch_size[0]-1)
    ma[z_inds[m]+i, y_inds[m],x_inds[m]] = 1

  ma = ma.astype(np.uint8)
  ma[z_inds,y_inds,x_inds] = 2
  return ma
