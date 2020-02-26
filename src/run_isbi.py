from types import SimpleNamespace
from segtools import torch_models
from segtools.numpy_utils import normalize3
from segtools.math_utils import conv_at_pts_multikern
from segtools.ns2dir import save,load
from pathlib import Path
import numpy as np
from skimage.measure  import regionprops

"""
EVALUATE_METHOD_ON_ALL_ISBI_DATASETS

for isbi_dataset in worm, fly, trib2d, trib3d
  for trainingset in [01,02]
    for iteration in n_repeats
      train on trainingset (gpu) [3 hrs]
      for predict_dataset in [01,02]
        for img in predict_dataset
          predict on img (gpu) [1 min]
          find points in img [1 min]
          score points [1 sec]
        compute total score (as function of dub)
      combine totals into distribution

a for loop is "parallel" if all the actions it contains (including child loops!) are independent conditioned on loop variable
but it is the actions that really do the work / require the resources / can have dependence relations, not the for's.
"""

import detect_isbi
import predict

def add_defaults_to_namespace(namespace,defaults):
  if type(namespace) is SimpleNamespace: namespace=namespace.__dict__
  if type(defaults) is SimpleNamespace: defaults=defaults.__dict__
  namespace.update(**{**defaults,**namespace})
  namespace = SimpleNamespace(**namespace)
  return namespace

def run_everything(rawdirs,train_sets=['01','02'],pred_sets=['01','02']):

  map1 = dict(celegans_isbi='Fluo-N3DH-CE',fly_isbi='Fluo-N3DL-DRO',trib_isbi_proj='Fluo-N3DL-TRIC',trib_isbi='Fluo-N3DL-TRIF')
  map2 = dict(celegans_isbi=celegans_isbi,fly_isbi=fly_isbi,trib_isbi_proj=trib_isbi_proj,trib_isbi=trib_isbi)

  for raw in rawdirs:
    config_builder = map2[raw]
    for tset in train_sets:
      # config = config_builder(train_set=tset)
      # if not config.predictor.best_model.exists():
      #   print(f"Training {raw} {tset}")
      #   T = detect_isbi.init(config.trainer)
      #   detect_isbi.train(T,config.trainer)
      for pset in pred_sets:
        config = config_builder(train_set=tset,pred=pset)
        
        print(f"Predicting from {raw} {tset} on {pset}")
        predict.isbi_predict(config.predictor)
        
        print(f"Evaluating from {raw} {tset} on {pset}")
        predict.total_matches(config.evaluator)
        predict.rasterize_isbi_detections(config.evaluator)
        predict.evaluate_isbi_DET(config.evaluator)



BEST_MODEL = "net14.pt"
TRAIN_TIME = 15*600 ## number of backprops (fixed 600/epoch)

## snakemake stuff

def _all_matches(wc):
  maxtime    = len(list(Path(f"/projects/project-broaddus/rawdata/{wc.rawdir}/{wc.isbiname}/{wc.pred}_GT/TRA/").glob("man_track*.tif")))
  allmatches = [f"/projects/project-broaddus/devseg_2/ex6/{wc.rawdir}/train/{wc.train_set}/matches/{wc.isbiname}/{wc.pred}/t{time:03d}.pkl" for time in range(maxtime)]
  # allmatches = expand(deps.matches_out_wc,**)
  return allmatches

def build_snakemake_wcs():
  deps = SimpleNamespace()

  deps.traindir = "/projects/project-broaddus/devseg_2/ex6/{rawdir}/train/{train_set}/"
  deps.trainout = deps.traindir + "m/" + BEST_MODEL
  deps.matches_inp_wc = "/projects/project-broaddus/rawdata/{rawdir}/{isbiname}/{pred}/t{time}.tif" ## Snakemake wildcard
  deps.matches_out_wc = deps.traindir + "matches/{isbiname}/{pred}/t{time}.pkl" ## Snakemake wildcard

  deps.name_total_scores = deps.traindir + "matches/{isbiname}/{pred}/total.pkl"
  deps.name_total_traj = deps.traindir + "pts/{isbiname}/{pred}/traj.pkl"
  deps.DET_output = "/projects/project-broaddus/rawdata/{rawdir}/{isbiname}/{pred}_RES/DET_log_{train_set}.txt"
  
  deps.all_matches = _all_matches

  return deps

def choose_config_from_snakemake_wildcards(wildcards):
  map2 = dict(celegans_isbi=celegans_isbi,fly_isbi=fly_isbi,trib_isbi_proj=trib_isbi_proj,trib_isbi=trib_isbi)
  print(wildcards)
  res = map2[wildcards.rawdir](**wildcards.__dict__)
  return res


## Here are the actual dataset specific setup functions

def trib_isbi_proj(train_set='01',pred='01',time='all',**kwargs):
  # defaults  = SimpleNamespace(isbiname='Fluo-N3DH-CE',rawdir='celegans_isbi',train_set='02',pred='02',name='normal')
  # train_set = wildcards['train_set']
  # pred = wildcards['pred']

  C = SimpleNamespace()
  C.trainer = SimpleNamespace()
  C.predictor = SimpleNamespace()
  C.evaluator = SimpleNamespace()

  C.trainer.input_dir      = Path(f"/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/{train_set}/")
  C.trainer.train_dir      = Path(f"/projects/project-broaddus/devseg_2/ex6/trib_isbi_proj/train/{train_set}/")  
  C.trainer.traj_gt_train  = load(f"/projects/project-broaddus/rawdata/trib_isbi_proj/traj/Fluo-N3DL-TRIC/{train_set}_traj.pkl")
  C.trainer.maxtime_train  = len(list(C.trainer.input_dir.glob("*.tif")))
  _times = np.linspace(0,C.trainer.maxtime_train - 1,8).astype(np.int)
  C.trainer.traintimes     = _times[[2,5]]
  C.trainer.valitimes      = _times[[0,1,3,4,6,7]]

  C.predictor.input_dir    = Path(f"/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/{pred}/")
  C.predictor.train_dir    = C.trainer.train_dir
  C.predictor.traj_gt_pred = load(f"/projects/project-broaddus/rawdata/trib_isbi_proj/traj/Fluo-N3DL-TRIC/{pred}_traj.pkl")
  C.predictor.out          = SimpleNamespace()
  C.predictor.out.base     = C.trainer.train_dir
  C.predictor.out.tail     = Path(f"Fluo-N3DL-TRIC/{pred}/")
  C.predictor.maxtime_pred = len(C.predictor.traj_gt_pred)
  C.predictor.best_model   = C.trainer.train_dir / ("m/" + BEST_MODEL)
  
  if time=='all':
    C.predictor.predict_times = list(range(C.predictor.maxtime_pred))
  else: 
    C.predictor.predict_times = [int(time)]

  C.evaluator.RAWdir          = Path(f"/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/{pred}/")
  C.evaluator.RESdir          = Path(f"/projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/{pred}_RES/")

  C.evaluator.name_total_scores = C.trainer.train_dir / f"matches/Fluo-N3DL-TRIC/{pred}/total.pkl"
  C.evaluator.name_total_traj   = C.trainer.train_dir / f"pts/Fluo-N3DL-TRIC/{pred}/traj.pkl"
  C.evaluator.out = C.predictor.out

  C.evaluator.DET_command = f"""
  time /projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure {C.evaluator.RESdir.parent} {pred} 3 0
  cd /projects/project-broaddus/rawdata/trib_isbi_proj/Fluo-N3DL-TRIC/{pred}_RES/
  mv DET_log.txt DET_log_{train_set}.txt
  """

  ## NO FILE DEPS

  ## data varies much over time, so we need special sampling
  C.trainer.f_net_args   = ((16,[[1],[1]]), dict(finallayer=torch_models.nn.Sequential)) ## *args and **kwargs
  C.trainer.norm         = lambda img: normalize3(img,2,99.4,clip=True)

  C.trainer.sigmas       = np.array([3,5,5])
  C.trainer.kernel_shape = np.array([43,43,43]) ## 7sigma in each direction
  C.trainer.sampler      = 'content'
  C.trainer.patch_space  = np.array([16,128,128])
  C.trainer.patch_full   = np.array([1,1,16,128,128])
  C.trainer.fg_bg_thresh = 0.01
  C.trainer.bg_weight_multiplier = 0 #0.2
  C.trainer.weight_decay = False
  C.trainer.rescale_for_matching = (1,1,1)
  C.trainer.i_final      = TRAIN_TIME
  C.trainer.bp_per_epoch = 600

  C.predictor.f_net_args = C.trainer.f_net_args
  C.predictor.norm       = C.trainer.norm
  C.predictor.plm_footprint = np.ones((3,10,10))

  C.evaluator.det_pts_transform = lambda x: x
  
  def pts2lab(pts,shape):
    kerns = [np.zeros((5,10,10)) + j + 1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,shape)
    lab = lab.astype(np.uint16)
    return lab
  C.evaluator.pts2lab = pts2lab

  return C

def celegans_isbi(train_set='01',pred='01',time='all',**kwargs):

  C = SimpleNamespace()
  C.trainer = SimpleNamespace()
  C.predictor = SimpleNamespace()
  C.evaluator = SimpleNamespace()

  C.trainer.input_dir      = Path(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{train_set}/")
  C.trainer.train_dir      = Path(f"/projects/project-broaddus/devseg_2/ex6/celegans_isbi/train/{train_set}/")  
  C.trainer.traj_gt_train  = load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{train_set}_traj.pkl")
  C.trainer.traintimes     = [0,5,33,100,189]
  C.trainer.valitimes      = [0,1,180]

  C.predictor.input_dir    = Path(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{pred}/")
  C.predictor.train_dir    = C.trainer.train_dir
  C.predictor.traj_gt_pred = load(f"/projects/project-broaddus/rawdata/celegans_isbi/traj/Fluo-N3DH-CE/{pred}_traj.pkl")
  C.predictor.out          = SimpleNamespace()
  C.predictor.out.base     = C.trainer.train_dir
  C.predictor.out.tail     = Path(f"Fluo-N3DH-CE/{pred}/")
  C.predictor.maxtime_pred = len(C.predictor.traj_gt_pred)
  C.predictor.best_model   = C.trainer.train_dir / ("m/net30.pt") # + BEST_MODEL)
  
  if time=='all':
    C.predictor.predict_times = list(range(C.predictor.maxtime_pred))
  else: 
    C.predictor.predict_times = [int(time)]

  C.evaluator.RAWdir          = Path(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{pred}/")
  C.evaluator.RESdir          = Path(f"/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{pred}_RES/")

  C.evaluator.name_total_scores = C.trainer.train_dir / f"matches/Fluo-N3DH-CE/{pred}/total.pkl"
  C.evaluator.name_total_traj   = C.trainer.train_dir / f"pts/Fluo-N3DH-CE/{pred}/traj.pkl"
  C.evaluator.out = C.predictor.out

  C.evaluator.DET_command = f"""
  time /projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure {C.evaluator.RESdir.parent} {pred} 3
  cd /projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/{pred}_RES/
  mv DET_log.txt DET_log_{train_set}.txt
  """

  ## NO FILE DEPS

  ## data varies much over time, so we need special sampling
  C.trainer.f_net_args   = ((16,[[1],[1]]), dict(finallayer=torch_models.nn.Sequential)) ## *args and **kwargs
  C.trainer.norm         = lambda img: normalize3(img,2,99.4,clip=True)
  # config.maxtime_train = len(list(C.trainer.input_dir.glob("*.tif")))
  # _times = np.linspace(0,config.maxtime_train - 1,8).astype(np.int)

  C.trainer.sigmas       = np.array([3,7,7])
  C.trainer.kernel_shape = np.array([43,43,43]) ## 7sigma in each direction
  C.trainer.sampler      = 'content'
  C.trainer.patch_space  = np.array([16,128,128])
  C.trainer.patch_full   = np.array([1,1,16,128,128])
  C.trainer.fg_bg_thresh = 0.01
  C.trainer.bg_weight_multiplier = 0.2
  C.trainer.weight_decay = True
  C.trainer.rescale_for_matching = (2,1,1)
  C.trainer.i_final      = 31*600 #TRAIN_TIME
  C.trainer.bp_per_epoch = 600

  C.predictor.f_net_args = C.trainer.f_net_args
  C.predictor.norm       = C.trainer.norm
  C.predictor.plm_footprint = np.ones((3,10,10))

  C.evaluator.det_pts_transform = lambda x: x
  
  def pts2lab(pts,shape):
    kerns = [np.zeros((3,10,10)) + j + 1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,shape)
    lab = lab.astype(np.uint16)
    return lab
  C.evaluator.pts2lab = pts2lab

  return C

def trib_isbi(train_set='01',pred='01',time='all',**kwargs):

  C = SimpleNamespace()
  C.trainer = SimpleNamespace()
  C.predictor = SimpleNamespace()
  C.evaluator = SimpleNamespace()

  C.trainer.input_dir      = Path(f"/projects/project-broaddus/rawdata/trib_isbi/down/Fluo-N3DL-TRIF/{train_set}/")
  C.trainer.train_dir      = Path(f"/projects/project-broaddus/devseg_2/ex6/trib_isbi/train/{train_set}/")  
  C.trainer.traj_gt_train  = load(f"/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/{train_set}_traj.pkl")
  C.trainer.maxtime_train  = len(list(C.trainer.input_dir.glob("*.tif")))
  _times = np.linspace(0,C.trainer.maxtime_train - 1,8).astype(np.int)
  C.trainer.traintimes     = _times[[2,5]]
  C.trainer.valitimes      = _times[[0,1,3,4,6,7]]

  C.predictor.input_dir    = Path(f"/projects/project-broaddus/rawdata/trib_isbi/down/Fluo-N3DL-TRIF/{pred}/")
  C.predictor.train_dir    = C.trainer.train_dir
  C.predictor.traj_gt_pred = load(f"/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/{pred}_traj.pkl")
  C.predictor.out          = SimpleNamespace()
  C.predictor.out.base     = C.trainer.train_dir
  C.predictor.out.tail     = Path(f"Fluo-N3DL-TRIF/{pred}/")
  C.predictor.maxtime_pred = len(C.predictor.traj_gt_pred)
  C.predictor.best_model   = C.trainer.train_dir / ("m/" + BEST_MODEL)
  
  if time=='all':
    C.predictor.predict_times = list(range(C.predictor.maxtime_pred))
  else: 
    C.predictor.predict_times = [int(time)]

  C.evaluator.RAWdir          = Path(f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{pred}/")
  C.evaluator.RESdir          = Path(f"/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{pred}_RES/")

  C.evaluator.name_total_scores = C.trainer.train_dir / f"matches/Fluo-N3DL-TRIF/{pred}/total.pkl"
  C.evaluator.name_total_traj   = C.trainer.train_dir / f"pts/Fluo-N3DL-TRIF/{pred}/traj.pkl"
  C.evaluator.out = C.predictor.out

  C.evaluator.DET_command = f"""
  time /projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure {C.evaluator.RESdir.parent} {pred} 3 0
  cd /projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/{pred}_RES/
  mv DET_log.txt DET_log_{train_set}.txt
  """

  ## NO FILE DEPS

  ## data varies much over time, so we need special sampling
  C.trainer.f_net_args   = ((16,[[1],[1]]), dict(finallayer=torch_models.nn.Sequential)) ## *args and **kwargs
  C.trainer.norm         = lambda img: normalize3(img,2,99.4,clip=True)

  C.trainer.sigmas       = np.array([3,3,3])
  C.trainer.kernel_shape = np.array([43,43,43]) ## 7sigma in each direction
  C.trainer.sampler      = 'content'
  C.trainer.patch_space  = np.array([64,64,64])
  C.trainer.patch_full   = np.array([1,1,64,64,64])
  C.trainer.fg_bg_thresh = np.exp(-16/2)
  C.trainer.bg_weight_multiplier = 0
  C.trainer.weight_decay = False
  C.trainer.rescale_for_matching = (1,1,1)
  C.trainer.i_final      = TRAIN_TIME
  C.trainer.bp_per_epoch = 600

  C.predictor.f_net_args = C.trainer.f_net_args
  C.predictor.norm       = C.trainer.norm
  C.predictor.plm_footprint = np.ones((3,10,10))

  C.trainer.traj_gt_train  = [(x/3).astype(int) for x in C.trainer.traj_gt_train] ## to match data
  C.predictor.traj_gt_pred = [(x/3).astype(int) for x in C.predictor.traj_gt_pred]
  C.evaluator.det_pts_transform = lambda x: [pts*3 for pts in x]
  
  def pts2lab(pts,shape):
    kerns = [np.zeros((20,20,20)) + j + 1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,shape)
    lab = lab.astype(np.uint16)
    return lab
  C.evaluator.pts2lab = pts2lab

  return C

def fly_isbi(train_set='01',pred='01',time='all',**kwargs):

  C = SimpleNamespace()
  C.trainer = SimpleNamespace()
  C.predictor = SimpleNamespace()
  C.evaluator = SimpleNamespace()

  C.trainer.input_dir      = Path(f"/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/{train_set}/")
  C.trainer.train_dir      = Path(f"/projects/project-broaddus/devseg_2/ex6/fly_isbi/train/{train_set}/")  
  C.trainer.traj_gt_train  = load(f"/projects/project-broaddus/rawdata/fly_isbi/traj/Fluo-N3DL-DRO/{train_set}_traj.pkl")
  C.trainer.maxtime_train  = len(list(C.trainer.input_dir.glob("*.tif")))
  _times = np.linspace(0,C.trainer.maxtime_train - 1,8).astype(np.int)
  C.trainer.traintimes     = _times[[2,5]]
  C.trainer.valitimes      = _times[[0,1,3,4,6,7]]

  C.predictor.input_dir    = Path(f"/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/{pred}/")
  C.predictor.train_dir    = C.trainer.train_dir
  C.predictor.traj_gt_pred = load(f"/projects/project-broaddus/rawdata/fly_isbi/traj/Fluo-N3DL-DRO/{pred}_traj.pkl")
  C.predictor.out          = SimpleNamespace()
  C.predictor.out.base     = C.trainer.train_dir
  C.predictor.out.tail     = Path(f"Fluo-N3DL-DRO/{pred}/")
  C.predictor.maxtime_pred = len(C.predictor.traj_gt_pred)
  C.predictor.best_model   = C.trainer.train_dir / ("m/" + BEST_MODEL)
  
  if time=='all':
    C.predictor.predict_times = list(range(C.predictor.maxtime_pred))
  else: 
    C.predictor.predict_times = [int(time)]

  C.evaluator.RAWdir          = Path(f"/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/{pred}/")
  C.evaluator.RESdir          = Path(f"/projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/{pred}_RES/")

  C.evaluator.name_total_scores = C.trainer.train_dir / f"matches/Fluo-N3DL-DRO/{pred}/total.pkl"
  C.evaluator.name_total_traj   = C.trainer.train_dir / f"pts/Fluo-N3DL-DRO/{pred}/traj.pkl"
  C.evaluator.out = C.predictor.out

  C.evaluator.DET_command = f"""
  time /projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure {C.evaluator.RESdir.parent} {pred} 3 0
  cd /projects/project-broaddus/rawdata/fly_isbi/Fluo-N3DL-DRO/{pred}_RES/
  mv DET_log.txt DET_log_{train_set}.txt
  """

  ## NO FILE DEPS

  ## data varies much over time, so we need special sampling
  C.trainer.f_net_args   = ((16,[[1],[1]]), dict(finallayer=torch_models.nn.Sequential)) ## *args and **kwargs
  C.trainer.norm         = lambda img: normalize3(img,2,99.4,clip=True)

  C.trainer.sigmas       = np.array([3,3,3])
  C.trainer.kernel_shape = np.array([43,43,43]) ## 7sigma in each direction
  C.trainer.sampler      = 'content'
  C.trainer.patch_space  = np.array([16,128,128])
  C.trainer.patch_full   = np.array([1,1,16,128,128])
  C.trainer.fg_bg_thresh = 0.01
  C.trainer.bg_weight_multiplier = 0
  C.trainer.weight_decay = False
  C.trainer.rescale_for_matching = (2,1,1)
  C.trainer.i_final      = TRAIN_TIME
  C.trainer.bp_per_epoch = 600

  C.predictor.f_net_args = C.trainer.f_net_args
  C.predictor.norm       = C.trainer.norm
  C.predictor.plm_footprint = np.ones((3,10,10))

  C.evaluator.det_pts_transform = lambda x: x
  
  def pts2lab(pts,shape):
    kerns = [np.zeros((3,10,10)) + j + 1 for j in range(len(pts))]
    lab = conv_at_pts_multikern(pts,kerns,shape)
    lab = lab.astype(np.uint16)
    return lab
  C.evaluator.pts2lab = pts2lab

  return C



notes = """

Tue Feb 18 10:56:18 2020

Ideally I would like to use Snakemake as a purely optional tool for parallelizing and automatizing this project.
Thus I'm worried about concepts from Snakemake sneaking into the d_isbi objects: i.e. wildcards.
Also, I'd like for framework to work on any dataset I'm given that adheres to the ISBI conventions.
So I'm worried about my own nameschemes necessitated by Snakemake sneaking into this module.

TODO: include creation of bash scripts that run prediction on ISBI server in Snakemake.


Wed Feb 19 09:14:36 2020


I'd really like to get the overall design of this project right.
This is more than an api for training and prediction, but includes all the ways we interact with and run code.
The most important ways we do are
1. running code through snakemake
2. running code through ipython

- We should minimize the differences between these different ways of running code
- The internals of the code shouldn't know anything about which way it is being run

Also, we may want to run different amounts of code at different times.
1. at the largest scale, we may want to run the entire experiment: All datasets, all iterations, all predictions and all analysis. (note that figures are a separate issue. here we just generate raw data. maybe that will change in the future.)
2. at a smaller scale we may want to rerun just the training or point detection with new parameters.

- It should be easy to run a piece of the workflow without having to re-run everything that came before. snakemake is designed exactly for this use case, and allows running at all scales by default.

- we don't want to do any serious coding in snakemake. it should basically only call single ipython functions.
- bonus: keeping Snakemake thin minimizes differences between ipython/snakemake interaction modes.

- snakemake will build it's own dependency tree, which we do not have access to.
  - so try not to pass lots of files determined by snakemake directly to your code.
  - it's better to let snakemake determine the wildcards (which are easy to recreate), then the wildcards determine the files via run_isbi.

- keep all snakemake wildcards contained in a single namespace which parameterizes your entire project.

overview:
- snakemake determines wildcards, passes them to (global? or rule specific? dataset specific?) func in run_isbi 
- run_isbi funcs take wildcards and turn them into config namespaces
- the code which does the real work takes config namespaces as arguments
counterexample:
- isbi_predict_generic runs independently for each _timepoint_. We do not want to build a separate config for each timepoint... or do we? just to pass to predict.isbi_single? ... actually maybe we do...
- we could have alternative ways of running this code. the snakemake way: in parallel with a single config per timepoint. or the no-snakemake way... maybe the best thing would be for the config to have a list of timepoints on which to predict...

config namespace separation matches the logical separation of workflow (and also match with snakemake)
functions which turn wildcards into config namespaces must have separate cases for each dataset?
data-specific namespaces can inheret from a shared generic namespace.
And we can build a larger config namespace to represent the entire workflow?
This is only necessary if we want to run the entire workflow from a python function, not via snakemake.
We will never want to do this, although it should be easy.
Snakemake is perfectly capable of running a workflow locally or without GPU if need be.
So we can allow snakemake to be our sole way of running multiple workflow pieces at once, and only have config namespaces for specific workflow pieces...
WARNING: This prevents us from sharing info _in memory_ between different parts of the workflow. i.e. different sections can only communicate through data and shared pieces of configs.
Let's assume this is OK/desireable for now and split up the workflow into pieces....

OK the problems with the cluster have gotten really bad...
I basically can't submit jobs any faster than 1/minute. So Instead of going highly parallel via Snakemake I'm going to reserve four nodes (one for each dataset) and go sequential (via ipython).
That OR I could make snakemake rules that predict on alllll files instead of just one.
Maybe there is even a snakemake command that forces rules to be submitted as group jobs...?
`group` is the most relevant concept here, but unfortunately there is no obvious way to submit many jobs stemming from the same wildcard rule together without tying them _all_ together...

Another useful thing:
`from snakemake.io import expand`
for easier work with wildcards

https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#dynamic-files
alternative to making large lists of files?
NO i don't think so. we need to know all the files ahead of time, so the aggregation rule can wait until they are all present.



Not all snakemake rules need to specify all info necessary to create a d_isbi.
e.g. the training rule doesn't need to specify the value of `pred`.
Should we build the full d_isbi anyways? Is this going to introduce weird bugs later?
The problem is that we deisgned d_isbi to contain all the info necessary to run training (once) and prediction (once), but not more or less.
We could split up d_isbi into smaller pieces, but containing too much info is not a real problem. only too little.
We could make the call from the snakemake rule more specific. Each rule knows whether it is training/predicting/evaluating/etc...
Making each piece smaller would also allow us to share across datasets more easily, if possible.




Sun Feb 23 15:42:54 2020

After fixing a small bug in DET_evaluation command, we can now train and evaluate on all datasets in sequence via the run_everything command.
no! more bugs. you put predictions on 01 and 02 in the same folder, so it's likely that both 01 and 02 have faulty evaluations.
- training and prediction on trib_pred has taken 40 hours so far, and we're not even done with prediction...

OK. I have an idea for what might contribute to the c.elegans large nucleus problem, and how to fix it:
Since we use flat sampling on c. elegans we are much less likely to find a nucleus when we choose t=0 than when we choose t=189.
If we simply use content-based sampling for c. elegans we will oversample nuclei in t=0, which is OK!


"""








