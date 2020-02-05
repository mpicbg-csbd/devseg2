"""
This module performs all evaluations of detection, segmentation and tracking results.
The Cell Tracking Challenge metrics `det` `seg` and `tra` are evaluated using the official binaries.

These metrics only have an interface for files written to disk.
Other metrics are implemented here and have an additional interface for in-memory python objects for more rapid evaluation.

"""

import numpy as np
from subprocess import Popen,run
from segtools.math_utils import conv_at_pts_multikern
from pathlib import Path
from segtools.ns2dir import load,save,toarray
import tifffile
import point_matcher



def rasterize_celegans_predictions():
  traj = load("/projects/project-broaddus/devseg_2/e03/test/pts/Fluo-N3DH-CE/01/traj.pkl")
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


## non-isbi metrics

def evaluate_drosophila_nn_matching():
  traj_gt = load("/lustre/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/traj_01.pkl")
  traj    = toarray(load("/projects/project-broaddus/devseg_2/e04_flydet/pts/Fluo-N3DL-DRO/01/"))
  matches = np.array([point_matcher.match_points_single(x,y,dub=8) for x,y in zip(traj_gt,traj)])
  scores  = point_matcher.matches2scores(matches)
  print(scores)

def evaluate_celegans_nn_matching():
  traj_gt = load("/lustre/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/gtpts1_unscaled.pkl")
  traj    = load("/projects/project-broaddus/devseg_2/e03/test/pts/Fluo-N3DH-CE/01/traj.pkl")
  matches = np.array([point_matcher.match_points_single(x,y,dub=8) for x,y in zip(traj_gt,traj)])
  scores  = point_matcher.matches2scores(matches)
  print(scores)

# def evaluate_trib2d_nn_matching():
#   traj_gt = load("/projects/project-broaddus/rawdata/trib_isbi_proj/traj/Fluo-N3DL-TRIC/01_traj.pkl")
#   traj    = load("/projects/project-broaddus/devseg_2/e05_trib2d/traj/Fluo-N3DL-TRIC/01/traj.pkl")
#   matches = np.array([point_matcher.match_points_single(x,y,dub=5) for x,y in zip(traj_gt,traj)])
#   scores  = point_matcher.matches2scores(matches)
#   print(scores)

def evaluate_trib3d_nn_matching():
  traj_gt = load("/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/02_traj.pkl")
  traj    = load("/projects/project-broaddus/devseg_2/e06_trib/traj/Fluo-N3DL-TRIF/02/traj.pkl")
  matches = np.array([point_matcher.match_points_single(x,y,dub=10) for x,y in zip(traj_gt,traj)])
  scores  = point_matcher.matches2scores(matches)
  print(scores)
