
import ipdb
from  types    import SimpleNamespace
from  math     import floor,ceil
import numpy         as np
from skimage.measure      import regionprops
from segtools.math_utils import conv_at_pts4, conv_at_pts_multikern
from segtools.point_tools import trim_images_from_pts2, patches_from_centerpoints

## manipulate labels and targets
from expand_labels_scikit import expand_labels
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.morphology import thin, binary_opening


## targets

def pts2target_gaussian(list_of_pts,sh,sigmas):
  target = np.array([place_gaussian_at_pts(pts,sh,sigmas) for pts in list_of_pts])
  return target

def pts2target_gaussian_sigmalist(list_of_pts,sh,list_of_sigmas):
  return np.array([pts2target_gaussian([x],sh,sig)[0] for x,sig in zip(list_of_pts,list_of_sigmas)])


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

def weights__decaying_bg_multiplier(yt,time,thresh=1.0,decayTime=None,bg_weight_multiplier=1.0):
  "weight pixels in the slice based on pred patch content"
  
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








