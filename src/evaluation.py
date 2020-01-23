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

def rasterize_drosophila_predictions():
  traj = toarray(load("/projects/project-broaddus/devseg_2/e04_flydet/pts/Fluo-N3DL-DRO/01/"))
  raw  = load("/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01/t000.tif") ## shape (125, 603, 1272)

  home = Path("/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01_RES/")
  for i in range(len(traj)):
    pts = traj[i]
    kerns = [np.zeros((3,10,10))+j+1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,raw.shape)
    lab = lab.astype(np.uint16)
    save(lab, home / f'mask{i:03d}.tif')

def evaluate_drosophila_DET():
  home = Path("/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/01_RES/")
  cmd = f"time '/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure' {home.parent} 01 3"
  run([cmd],shell=True)

## non-isbi metrics

def evaluate_celegans_nn_matching():
  traj_gt = load("/lustre/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/traj_01.pkl")