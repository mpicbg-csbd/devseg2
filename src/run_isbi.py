from types import SimpleNamespace
from segtools import torch_models
from segtools.numpy_utils import normalize3
from segtools.math_utils import conv_at_pts_multikern
from segtools.ns2dir import save,load
from pathlib import Path
import numpy as np


def build_trib2d_dset():
  d_isbi = SimpleNamespace()
  d_isbi.name = "trib2d"
  
  d_isbi.trainer = SimpleNamespace()
  d_isbi.trainer.RAWdir = Path("/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01/")
  d_isbi.trainer.SEGdir = Path("/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01_GT/SEG/")
  d_isbi.trainer.TRAdir = Path("/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01_GT/TRA/")
  d_isbi.trainer.RESdir = Path("/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01_RES/")

  d_isbi.trainer.traindir = Path("/projects/project-broaddus/devseg_2/e05_trib2d/test_01/")
  d_isbi.trainer.pts_dir  = Path("/projects/project-broaddus/devseg_2/e05_trib2d/test_01/pts/Fluo-N3DL-TRIC/01/")
  d_isbi.trainer.pred_dir = Path("/projects/project-broaddus/devseg_2/e05_trib2d/test_01/pred/Fluo-N3DL-TRIC/01/")
  d_isbi.trainer.match_dir = Path("/projects/project-broaddus/devseg_2/e05_trib2d/test_01/matches/Fluo-N3DL-TRIC/01/")
  d_isbi.trainer.name_total_scores = d_isbi.trainer.match_dir / "total.pkl"
  d_isbi.trainer.name_total_traj   = d_isbi.trainer.pts_dir / "traj.pkl"
  
  ## Snakemake
  d_isbi.trainer.trainout = Path("/projects/project-broaddus/devseg_2/e05_trib2d/test_01/m/net10.pt") ## used as Snakemake output
  d_isbi.trainer.matches_inp_wc = "/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/01/t{time}.tif" ## Snakemake wildcard
  d_isbi.trainer.matches_out_wc = "/projects/project-broaddus/devseg_2/e05_trib2d/test_01/matches/Fluo-N3DL-TRIC/01/s{time}.pkl" ## Snakemake wildcard
  d_isbi.all_matches = [f"/projects/project-broaddus/devseg_2/e05_trib2d/test_01/matches/Fluo-N3DL-TRIC/01/s{time:03d}.pkl" for time in range(65)]

  d_isbi.trainer.valitimes  = [12,51]
  d_isbi.trainer.traintimes = [ 0, 25, 38, 64]
  d_isbi.trainer.f_net = lambda : torch_models.Unet3(16,[[1],[1]],finallayer=torch_models.nn.Sequential).cuda()
  d_isbi.trainer.norm = lambda img: normalize3(img,2,99.6)

  d_isbi.trainer.sigmas = np.array([1,3,3])   ## sigma for gaussian
  d_isbi.trainer.kernel_shape = np.array([7,21,21]) ## kernel size. must be all odd
  d_isbi.trainer.rawshape = load(d_isbi.trainer.RAWdir / "t000.tif").shape
  d_isbi.trainer.fg_bg_thresh = 0.005

  d_isbi.trainer.patch_space = np.array([13,256,256])
  d_isbi.trainer.patch_full  = np.array([1,1,13,256,256])
  d_isbi.trainer.best_model  = "/projects/project-broaddus/devseg_2/e05_trib2d/test_01/m/net04.pt"

  d_isbi.traj_gt = load("/projects/project-broaddus/rawdata/trib_isbi_proj/traj/Fluo-N3DL-TRIC/01_traj.pkl")

  d_isbi.DET_command = f"time '/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure' {d_isbi.trainer.RESdir.parent} 01 3"

  def pts2lab(pts):
    kerns = [np.zeros((3,10,10)) + j + 1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,d_isbi.trainer.rawshape)
    lab = lab.astype(np.uint16)
    return lab

  d_isbi.pts2lab = pts2lab

  return d_isbi


# def evaluate_method_on_isbi_dataset(d_isbi):
  # for dset in [01,02] (parallel on cluster w gpu)
    # train 
    # for dset in [01,02] (parallel in cluster w gpu)
      # for img in dset (parallel in cluster w gpu)
        # predict
        # find points
        # evaluate
      # evaluate_all (must wait for all evaluations to complete. no gpu reqd.)

