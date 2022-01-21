
import ipdb
from  types    import SimpleNamespace
from  math     import floor,ceil
import numpy         as np
from skimage.measure      import regionprops
from segtools.math_utils import conv_at_pts4, conv_at_pts_multikern
from segtools.point_tools import trim_images_from_pts2, patches_from_centerpoints

from pykdtree.kdtree import KDTree as pyKDTree

from numba import jit

## manipulate labels and targets
from expand_labels_scikit import expand_labels
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.morphology import thin, binary_opening


## targets

# def pts2target_gaussian(list_of_pts,sh,sigmas):
#   target = np.array([place_gaussian_at_pts(pts,sh,sigmas) for pts in list_of_pts])
#   return target

# def pts2target_gaussian_sigmalist(list_of_pts,sh,list_of_sigmas):
#   return np.array([pts2target_gaussian([x],sh,sig)[0] for x,sig in zip(list_of_pts,list_of_sigmas)])


def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

def place_gaussian_at_pts(pts,sh,sigmas):
  s  = np.array(sigmas)
  ks = np.ceil(s*7).astype(np.int) #*6 + 1 ## gaurantees odd size and thus unique, brightest center pixel
  ks = ks - ks%2 + 1## enfore ODD shape so kernel is centered! (grow even dims by 1 pix)
  # ks = (s*7).astype(np.int) ## must be ODD
  def f(x):
    x = x - (ks-1)/2
    return np.exp(-(x*x/s/s).sum()/2)
  kern = np.array([f(x) for x in np.indices(ks).reshape((len(ks),-1)).T]).reshape(ks)
  kern = kern / kern.max()
  target = conv_at_pts4(pts,kern,sh,lambda a,b:np.maximum(a,b))
  return target

def lab_to_distance_transform(lab):
  y = lab.copy()
  y = y>0
  for _ in range(5): y = binary_opening(y)
  # y = y.astype(np.float32)
  # y = convolve(y,np.ones([5,5]))
  # y = gaussian_filter(y.astype(np.float32),sigma=3)
  # y = gputools.convolve()
  # for i in range(3): y=binary_dilation(y)
  # y = expand_labels(y,3)>0
  y = distance_transform_edt(y)
  y = expand_labels(y,10)>0
  y = distance_transform_edt(y)
  return y

## weights

"weight pixels in the slice based on ytrue content"
def weights__decaying_bg_multiplier(yt,time,thresh=1.0,decayTime=None,bg_weight_multiplier=1.0):
  
  w = np.ones(yt.shape)
  m0 = yt<thresh # background
  m1 = yt>thresh # foreground
  if 0 < m0.sum() < m0.size:
    # ws = 1/np.array([m0.mean(), m1.mean()]).astype(np.float) ## REMOVE. WE DON'T WANT TO BALANCE FG/BG. WE WANT TO EMPHASIZE FG.
    ws = [1,1] * np.array(1.)
    ws[0] *= bg_weight_multiplier
    ws /= ws.mean()
    if np.isnan(ws).any(): ipdb.set_trace()

    if decayTime:
      ## decayto1 :: linearly decay scalar x to value 1 after 3 epochs, then const
      decayto1 = lambda x: x*(1-time/decayTime) + time/decayTime if time<=decayTime else 1
      ws[0] = decayto1(ws[0])
      ws[1] = decayto1(ws[1])
      
    w[yt<thresh]  = ws[0]
    w[yt>=thresh] = ws[1]

  return w


## Utils

## Point, Patch and Sampling tools

def find_points_within_patches(centerpoints, allpts, _patchsize):
  kdt = pyKDTree(allpts)
  # ipdb.set_trace()
  N,D = centerpoints.shape
  dists, inds = kdt.query(centerpoints, k=10, distance_upper_bound=np.linalg.norm(_patchsize)/2)
  def _test(x,y):
    return (np.abs(x-y) <= np.array(_patchsize)/2).all()
  pts = [[allpts[n] for n in ns if n<len(allpts) and _test(centerpoints[i],allpts[n])] for i,ns in enumerate(inds)]
  return pts

def find_points_within_patches2(points, patches):
  def ptinpatch(p,patch):
    return np.all([p[i] in range(patch[i].start,patch[i].stop) for i in range(len(patch))])

  res = np.zeros((len(points),len(patches)))
  for i in range(len(points)):
    for j in range(len(patches)):
      res[i,j] = ptinpatch(points[i],patches[j])

  return res

def find_points_within_patches3(points,patches):
  _patches = np.array([[(x.start,x.stop) for x in s] for s in patches])
  return jit_find_points_within_patches3(points,_patches)

@jit(nopython=True)
def jit_find_points_within_patches3(points, patches):
  "points in NxD; patches in MxDx2"

  N = points.shape[0]
  M = patches.shape[0]
  D = points.shape[1]

  assert D==patches.shape[1]

  res = np.ones((N,M))
  for i in range(N):
    p = points[i]
    for j in range(M):
      ss = patches[j]
      for k in range(D):
        x = ss[k,0] <= p[k] < ss[k,1]
        res[i,j] *= x

  return res

def test_findinpoints():
  points = np.random.randint(0, 100,  size=(100,1))
  points = np.r_[5:100:10].reshape([10,1])
  patches = [[slice(a,b)] for a,b in np.random.randint(0, 100, size=(20,2))]
  # patches = np.random.randint(0,100,size=(20,1,2))
  # patches = np.r_[0:100:10,10:101:10].reshape([2,10]).T.reshape([10,1,2])
  # ipdb.set_trace()
  res = find_points_within_patches3(points, patches)


## Tiling for training and prediction


def tile1d(end,length,border):
  """
  TODO: enfore divisibility constraints
  The input array and target container are the same size.
  """
  inner = length-2*border
  
  input_start  = np.r_[0:end-length:inner, end-length]
  input_start[0] = 0
  input_end = input_start+length
  input_end[-1] = end
  
  target_start = input_start+border
  target_start[0] = 0
  target_end   = input_start+length-border
  target_end[-1] = end #input_start[-1]+length

  relative_inner_start = [border]*len(input_start)
  relative_inner_start[0] = 0
  relative_inner_end   = [length-border]*len(input_start)
  relative_inner_end[-1] = length ## == None?

  _input     = tuple([slice(a,b) for a,b in zip(input_start,input_end)])
  _container = tuple([slice(a,b) for a,b in zip(target_start,target_end)])
  _patch     = tuple([slice(a,b) for a,b in zip(relative_inner_start,relative_inner_end)])
  res = np.array([_input,_container,_patch,]).T
  res = [SimpleNamespace(a=x[0],b=x[1],c=x[2]) for x in res]
  return res

def tile1d_random(sz_container,sz_outer,sz_inner):
  """
  The input array and target container are the same size.
  Returns a,b,c (input, container, local-patch-coords) for use in `container[b] = f(img[a])[c]`.
  Does NOT enforce divisibility constraints.
  Useful for training data generation where we want non-overlapping patches, but doesn't work when we need a prediction on every pixel.
  - non-overlapping patches (pixel coverage always <= 1)
  - no need to store pixel mask to ensure coverage <= 1
  - probabilistic, so border regions on all sides are probably somewhat represented in the training data

  """
  # inner = sz_outer-2*sz_inner
  
  ## enforce roughly equally sized "main" region with max of "sz_outer"
  n_patches = floor(sz_container/sz_outer)
  assert n_patches >= 1

  borderpoints  = np.linspace(0,sz_container,n_patches+1).astype(np.int)
  outer_starts  = borderpoints[:-1]
  outer_ends    = borderpoints[1:]

  sz_outer = outer_ends - outer_starts
  inner_starts = np.random.randint(outer_starts, high=outer_ends-sz_inner+1)
  inner_ends = inner_starts+sz_inner

  relative_inner_start = inner_starts - outer_starts
  relative_inner_end   = inner_ends - outer_starts

  _outer     = tuple([slice(a,b) for a,b in zip(outer_starts,outer_ends)])
  _inner     = tuple([slice(a,b) for a,b in zip(inner_starts,inner_ends)])
  _inner_rel = tuple([slice(a,b) for a,b in zip(relative_inner_start,relative_inner_end)])
  res = np.array([_outer,_inner,_inner_rel,]).T
  res = [SimpleNamespace(outer=x[0],inner=x[1],inner_rel=x[2]) for x in res]

  # ipdb.set_trace()
  return res

"""
generate patch coords from image,box,patch shapes. 
returns a list of coords. each coords is Dx4 = (top-left-box , top-left)
"""
def tileND_random(img_shape,outer_shape,inner_shape,):
  # if inner_shape is None: inner_shape = (0,)*len(img_shape)
  r = [tile1d_random(a,b,c) for a,b,c in zip(img_shape,outer_shape,inner_shape)] ## D x many x 3
  D = len(r) ## dimension
  # r = np.array(product())
  r = np.array(np.meshgrid(*r)).reshape([D,-1]).T ## should be an array D x len(a) x len(b)
  def f(s): # tuple of simple namespaces
    res = SimpleNamespace()
    keys = s[0].__dict__.keys()
    for k in keys:
      res.__dict__[k] = tuple([x.__dict__[k] for x in s])
    return res
  r = [f(s) for s in r]
  return r

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



def apply_net_tiled_nobounds(f_net,img,outchan=1,patch_shape=(512,512),border_shape=(20,20)):
  """
  TODO: generalize this to work with arbitrary function applied to big image over padded tiles.
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
  count = 0
  g = tile_multidim(img.shape[1:],patch_shape,border_shape)

  # print(img.shape, patch_shape,border_shape)
  # print(g)

  for s in g:
    a = (slice(None),) + s.outer ## add channels
    b = (slice(None),) + s.inner ## add channels
    c = (slice(None),) + s.inner_rel ## add channels
    # print(count); count += 1
    container[b] = f_net(img[a])[c]
    # del patch
  return container



## Applications of tiling

def find_errors(img, matching, _patchsize = (33,33)):
  # ipdb.set_trace()

  pts = matching.pts_yp[~matching.yp_matched_mask]
  patches = patches_from_centerpoints(img, pts, _patchsize)
  yp_in = _find_points_within_patches(pts, matching.pts_yp, _patchsize)
  gt_in = _find_points_within_patches(pts, matching.pts_gt, _patchsize)
  fp = [SimpleNamespace(pt=pts[i],patch=patches[i],yp=[pts[i]] + yp_in[i],gt=gt_in[i]) for i in range(pts.shape[0])]

  pts = matching.pts_gt[~matching.gt_matched_mask]
  patches = patches_from_centerpoints(img, pts, _patchsize)
  yp_in = _find_points_within_patches(pts, matching.pts_yp, _patchsize)
  gt_in = _find_points_within_patches(pts, matching.pts_gt, _patchsize)
  fn = [SimpleNamespace(pt=pts[i],patch=patches[i],yp=yp_in[i],gt=[pts[i]] + gt_in[i]) for i in range(pts.shape[0])]

  return fp, fn

def shape2slicelist(imgshape,patchsize,divisible=(1,4,4)):
  D  = len(patchsize)
  ns = np.ceil(np.array(imgshape) / patchsize).astype(int)
  ss = (np.indices(ns).T * patchsize).reshape([-1,D])
  # pad = [(0,0),(0,0),(0,0)]
  def _f(_s,i):
    low  = _s[i]
    high = min(_s[i]+patchsize[i],imgshape[i])
    high = low + floor((high-low)/divisible[i])*divisible[i]
    return slice(low, high)
  ss = [tuple(_f(_s,i) for i in np.r_[:D]) for _s in ss]
  return ss

def jitter_center_inbounds(pt,patch,bounds,jitter=0.1):
  D = len(patch)
  patch = np.array(patch)
  _pt  = pt + (2*np.random.rand(D))*patch*jitter ## jitter by 10% of patch width
  _pt  = _pt - patch//2 ## center
  _max = np.clip([bounds - patch],a_min=[0]*D,a_max=None)
  _pt  = _pt.clip(min=[0]*D,max=_max)[0] ## clip to bounds
  _pt  = _pt.astype(int)
  ss  = tuple(slice(_pt[i],_pt[i] + patch[i]) for i in range(len(_pt)))
  return ss

def sample_slice_from_volume(patch,bounds):
  _patch = np.minimum(patch,bounds)
  pt = (np.random.rand(len(patch))*(bounds - _patch)).astype(int)
  ss = tuple([slice(pt[i],pt[i]+_patch[i]) for i in range(len(_patch))])
  return ss








