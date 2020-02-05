from pykdtree.kdtree import KDTree as pyKDTree
import numpy as np
from types import SimpleNamespace
from scipy.optimize import linear_sum_assignment

def match_points_single(pts_gt,pts_yp,dub=10):
  "pts_gt is ground truth. pts_yp as predictions. this function is not symmetric!"
  pts_gt = np.array(pts_gt)
  pts_yp = np.array(pts_yp)
  if 0 in pts_gt.shape: return 0,len(pts_yp),len(pts_gt)
  if 0 in pts_yp.shape: return 0,len(pts_yp),len(pts_gt)
  # print(pts_gt.shape, pts_yp.shape)

  kdt = pyKDTree(pts_yp)
  dists, inds = kdt.query(pts_gt, k=1, distance_upper_bound=dub)
  matched,counts = np.unique(inds[inds<len(pts_yp)], return_counts=True)
  return len(matched), len(pts_yp), len(pts_gt)

def hungarian_matching(x,y,dub=10):
  """
  matching that minimizes sum of distances.
  returns number of matched, proposed and gt cells.
  """
  cost = np.zeros((len(x), len(y)))
  for i,c in enumerate(x):
    for j,d in enumerate(y):
      cost[i,j] = np.linalg.norm(c-d)
  res = linear_sum_assignment(cost)
  return np.sum(res), len(y), len(x)

def matches2scores(matches):
  """
  matches is an Nx3 array with (n_matched, n_proposed, n_target) semantics.
  here we perform mean-then-divide to compute scores. As opposed to divide-then-mean.
  """
  d = SimpleNamespace()
  d.f1          = 2*matches[:,0].sum() / np.maximum(matches[:,[1,2]].sum(),1)
  d.precision   =   matches[:,0].sum() / np.maximum(matches[:,1].sum(),1)
  d.recall      =   matches[:,0].sum() / np.maximum(matches[:,2].sum(),1)
  d.f1_2        = (2*matches[:,0] / np.maximum(matches[:,[1,2]].sum(1),1)).mean()
  d.precision_2 = (  matches[:,0] / np.maximum(matches[:,1],1)).mean()
  d.recall_2    = (  matches[:,0] / np.maximum(matches[:,2],1)).mean()

  return d
