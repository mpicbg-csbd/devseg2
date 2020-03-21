"""
This module performs all evaluations of detection, segmentation and tracking results.
The Cell Tracking Challenge metrics `det` `seg` and `tra` are evaluated using the official binaries.

These metrics only have an interface for files written to disk.
Other metrics are implemented here and have an additional interface for in-memory python objects for more rapid evaluation.

"""


from types import SimpleNamespace
import numpy as np
from subprocess import Popen,run
from segtools.math_utils import conv_at_pts_multikern
from segtools import point_tools
from pathlib import Path
from segtools.ns2dir import load,save,toarray
import tifffile
from segtools import point_matcher
import ipdb
from itertools import product




## c elegans

def rasterize_celegans_predictions():
  traj = load("/projects/project-broaddus/devseg_2/e03_celedet/test/pts/Fluo-N3DH-CE/01/traj.pkl")
  raw  = load("/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01/t000.tif")

  home = Path("/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01_RES/")
  for i in range(len(traj)):
    pts = traj[i][(0.1,30)]
    kerns = [np.zeros((3,10,10))+j+1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,raw.shape)
    lab = lab.astype(np.uint16)
    save(lab, home / f'mask{i:03d}.tif')

def evaluate_celegans_DET():
  home = Path("/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/01_RES/")
  cmd = f"time '/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure' {home.parent} 01 3"
  run([cmd],shell=True)

## drosophila

def rasterize_drosophila_predictions():
  traj = toarray(load("/projects/project-broaddus/devseg_2/e04_flydet/pts/Fluo-N3DL-DRO/01/"))
  raw  = load("/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01/t000.tif") ## shape (125, 603, 1272)

  home = Path("/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01_RES/")
  for i in range(len(traj)):
    pts = traj[i]
    kerns = [np.zeros((10,10,10))+j+1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,raw.shape)
    lab = lab.astype(np.uint16)
    save(lab, home / f'mask{i:03d}.tif')

def evaluate_drosophila_DET():
  home = Path("/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01_RES/")
  cmd  = f"time '/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure' {home.parent} 01 3 0"
  run([cmd],shell=True)

## trib2d

def rasterize_trib2d_predictions():
  traj = load("/projects/project-broaddus/devseg_2/e05_trib2d/traj/Fluo-N3DL-TRIC/01/traj.pkl")
  raw  = load("/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01/t000.tif") ## shape (13, 2454, 1693)

  home = Path("/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01_RES/")
  for i in range(63,len(traj)):
    pts = traj[i]
    kerns = [np.zeros((10,7,7))+j+1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,raw.shape)
    lab = lab.astype(np.uint16)
    save(lab, home / f'mask{i:03d}.tif')

def evaluate_trib2d_DET():
  """
  0.938512
  """
  home = Path("/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01_RES/")
  cmd  = f"time '/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure' {home.parent} 01 3 0"

  run([cmd],shell=True)

## trib 3d

def rasterize_trib_predictions():
  traj = load("/projects/project-broaddus/devseg_2/e06_trib/traj/Fluo-N3DL-TRIF/02/traj.pkl")
  # raw  = load("/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/01/t000.tif") ## shape (975, 1820, 1000)

  ## (975, 1820, 1000) is shape of 02's (991, 1871, 965) of the 01's
  home = Path("/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/02_RES/")
  for i in range(len(traj)):
    pts = traj[i]
    kerns = [np.zeros((12,12,12))+j+1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,(975, 1820, 1000))
    lab = lab.astype(np.uint16)
    save(lab, home / f'mask{i:03d}.tif')

def evaluate_trib_DET():
  """
  DET measure: 0.968807
  """
  home = Path("/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/02_RES/")
  cmd  = f"time '/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure' {home.parent} 02 3 0"

  run([cmd],shell=True)


## NON-ISBI METRICS

## drosophila

def evaluate_drosophila_nn_matching():
  traj_gt = load("/lustre/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/traj_01.pkl")
  traj    = toarray(load("/projects/project-broaddus/devseg_2/e04_flydet/pts/Fluo-N3DL-DRO/01/"))
  matches = np.array([point_matcher.match_points_single(x,y,dub=8) for x,y in zip(traj_gt,traj)])
  scores  = point_matcher.matches2scores(matches)
  print(scores)

## c elegans

def celegans_printScores_saveMatchings():
  traj_gt_01 = load("/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts1_unscaled.pkl")
  traj_gt_02 = load("/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts2_unscaled.pkl")  

  for train,pred in product(['test','test_02'], ['01','02']):
    traj_gt = traj_gt_01 if pred=='01' else traj_gt_02
    name1 = f"/projects/project-broaddus/devseg_2/e03_celedet/{train}/pts/Fluo-N3DH-CE/{pred}/traj.pkl"
    traj  = load(name1)
    matches      = [point_matcher.match_unambiguous_nearestNeib(x,y[(0.1,30)],dub=20,scale=[2,1,1]) for x,y in zip(traj_gt,traj)]
    match_scores = point_matcher.listOfMatches_to_Scores(matches)
    print(f"train {train} pred {pred}: ", match_scores)
    save(matches, Path(name1).parent / "symmetric_matching.pkl")




def celegans_dubVSaccuracy_curve():
  traj_gt_01 = load("/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts1_unscaled.pkl")
  traj_gt_02 = load("/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts2_unscaled.pkl")  

  for train,pred in product(['test','test_02'], ['01','02']):
    traj_gt = traj_gt_01 if pred=='01' else traj_gt_02
    name1 = f"/projects/project-broaddus/devseg_2/e03_celedet/{train}/pts/Fluo-N3DH-CE/{pred}/traj.pkl"
    traj  = load(name1)
    roc_scores = []
    for dub in np.linspace(0,30,60):
      syms      = [point_matcher.match_unambiguous_nearestNeib(x,y[(0.1,30)],dub=dub,scale=[2,1,1]) for x,y in zip(traj_gt,traj)]
      symscores = point_matcher.listOfMatches_to_Scores(syms)
      print(f"train {train} pred {pred}: ", symscores)
      roc_scores.append((dub,symscores))
    save(roc_scores, Path(name1).parent / "roc_scores.pkl")

def load_movie_data(train='01',pred='01'):
  d3 = SimpleNamespace()
  d3.mx_alltimes_raw  = toarray(load(f"/projects/project-broaddus/rawdata/celegans_isbi/mx/Fluo-N3DH-CE/{pred}/"))
  d3.gt_pts = load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts{pred[-1]}_unscaled.pkl")
  d3.mx_alltimes_pred = toarray(load(f"/projects/project-broaddus/devseg_2/e03_celedet/{traindir}/maxs/Fluo-N3DH-CE/{pred}/"))
  traindir = 'test' if train=='01' else 'test_02'
  d3.matching_alltimes = load(f"/projects/project-broaddus/devseg_2/e03_celedet/{traindir}/pts/Fluo-N3DH-CE/{pred}/symmetric_matching.pkl")
  d3.savedir = Path(f"/projects/project-broaddus/devseg_2/e03_celedet/{traindir}/pts/Fluo-N3DH-CE/{pred}/")
  return d3

def load_movie_data3d(train='01',pred='01'):
  d3 = SimpleNamespace()
  d3.gt_pts = load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts{pred[-1]}_unscaled.pkl")
  d3.mx_alltimes_raw  = toarray(load(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{pred}/"))
  # d3.mx_alltimes_pred = toarray(load(f"/projects/project-broaddus/devseg_2/e03_celedet/{traindir}/maxs/Fluo-N3DH-CE/{pred}/"))
  traindir = 'test' if train=='01' else 'test_02'
  print(traindir)
  d3.matching_alltimes = load(f"/projects/project-broaddus/devseg_2/e03_celedet/{traindir}/pts/Fluo-N3DH-CE/{pred}/symmetric_matching.pkl")
  d3.savedir = Path(f"/projects/project-broaddus/devseg_2/e03_celedet/{traindir}/pts/Fluo-N3DH-CE/{pred}/")
  return d3

def celegans_panel_of_detection_errors(d3):
  res = SimpleNamespace()
  res.mistake_patches_mx = []
  res.mistake_patches_ms = []
  for i in range(len(d3.matching_alltimes)):
    img = d3.mx_alltimes_raw[i]
    matching = d3.matching_alltimes[i]
    false_negatives = matching.pts_gt[~matching.gt_matched_mask]
    if false_negatives.shape[0]>0:
      x = point_tools.patches_from_centerpoints(img,false_negatives,(10,100,100))
      res.mistake_patches_mx.append(x.max(1))
      res.mistake_patches_ms.append(x[:,5])
  res.panel_mx = np.concatenate(res.mistake_patches_mx,axis=0)[:100].reshape((10,10,100,100)).transpose((0,2,1,3)).reshape((10*100,10*100))
  res.panel_ms = np.concatenate(res.mistake_patches_ms,axis=0)[:100].reshape((10,10,100,100)).transpose((0,2,1,3)).reshape((10*100,10*100))
  save(res,d3.savedir/'panel')
  return res


## tribolium 2D

def trib2d_printScores_saveMatchings():
  traj_gt_01 = load("/projects/project-broaddus/rawdata/trib_isbi_proj/traj/Fluo-N3DL-TRIC/01_traj.pkl")
  traj_gt_02 = load("/projects/project-broaddus/rawdata/trib_isbi_proj/traj/Fluo-N3DL-TRIC/02_traj.pkl")

  for train,pred in product(['t01','t02'], ['01','02']):
    traj_gt = traj_gt_01 if pred=='01' else traj_gt_02
    name1 = f"/projects/project-broaddus/devseg_2/e05_trib2d/{train}/traj/Fluo-N3DL-TRIC/{pred}/traj.pkl"
    traj    = load(name1)
    matches = [point_matcher.match_unambiguous_nearestNeib(x,y,dub=5) for x,y in zip(traj_gt,traj)]
    match_scores  = point_matcher.listOfMatches_to_Scores(matches)
    print(match_scores)
    save(matches, Path(name1).parent / "symmetric_matching.pkl")
    ## note that only the recall really tells us about the accuracy of the method (sparse annotations)

def trib2d_dubVSaccuracy_curve():
  traj_gt = load("/projects/project-broaddus/rawdata/trib_isbi_proj/traj/Fluo-N3DL-TRIC/01_traj.pkl")
  name1   = "/projects/project-broaddus/devseg_2/e05_trib2d/traj/Fluo-N3DL-TRIC/01/traj.pkl"
  traj    = load(name1)

  roc_scores = []
  for dub in np.linspace(0,10,30):
    matches = [point_matcher.match_unambiguous_nearestNeib(x,y,dub=dub) for x,y in zip(traj_gt,traj)]
    match_scores  = point_matcher.listOfMatches_to_Scores(matches)
    print(match_scores)
  save(roc_scores, Path(name1).parent / "roc_scores.pkl")


## tribolium 3D

def evaluate_trib3d_nn_matching():
  traj_gt = load("/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/02_traj.pkl")
  traj    = load("/projects/project-broaddus/devseg_2/e06_trib/traj/Fluo-N3DL-TRIF/02/traj.pkl")
  matches = np.array([point_matcher.match_points_single(x,y,dub=10) for x,y in zip(traj_gt,traj)])
  scores  = point_matcher.matches2scores(matches)
  print(scores)
