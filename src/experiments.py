from types import SimpleNamespace
from segtools import torch_models
from segtools.numpy_utils import normalize3
from segtools.math_utils import conv_at_pts_multikern
from segtools.ns2dir import save,load
from segtools import point_matcher
from pathlib import Path
import numpy as np
from skimage.measure  import regionprops
from itertools import product



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

import detector
import isbi_tools
import train_StructN2V
import data_specific

def add_defaults_to_namespace(namespace,defaults):
  if type(namespace) is not dict: namespace=namespace.__dict__
  if type(defaults) is not dict:  defaults=defaults.__dict__
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
      #   T = train_detector.init(config.trainer)
      #   train_detector.train(T,config.trainer)
      for pset in pred_sets:
        config = config_builder(train_set=tset,pred=pset)
        
        print(f"Predicting from {raw} {tset} on {pset}")
        predict.isbi_predict(config.predictor)
        
        print(f"Evaluating from {raw} {tset} on {pset}")
        predict.total_matches(config.evaluator)
        predict.rasterize_isbi_detections(config.evaluator)
        predict.evaluate_isbi_DET(config.evaluator)

## snakemake stuff

def _all_matches(wc):
  maxtime    = len(list(Path(f"/projects/project-broaddus/rawdata/{wc.rawdir}/{wc.isbiname}/{wc.pred}_GT/TRA/").glob("man_track*.tif")))
  allmatches = [f"/projects/project-broaddus/devseg_2/ex7/{wc.rawdir}/train{wc.tid}/p{wc.param_id}/{wc.train_set}/matches/{wc.isbiname}/{wc.pred}/t{time:03d}.pkl" for time in range(maxtime)]
  # allmatches = expand(deps.matches_out_wc,**)
  return allmatches

def apply_recursively_to_strings(obj,func,callable_arg=None):
  """
  apply func recursivly to obj, keeping nested structure.
  apply to values of dicts, not keys. 
  apply to all elements of lists.
  evaluate functions (they should return lists), then apply to lists
  """
  if type(obj) is str: return func(obj)

  if callable(obj): obj = obj(callable_arg)
  
  to_sn = False
  if type(obj) is SimpleNamespace:
    obj = obj.__dict__
    to_sn = True

  if type(obj) is dict:
    for k,v in obj.items():
      obj[k] = apply_recursively_to_strings(v,func,callable_arg)
    if to_sn: obj = SimpleNamespace(**obj)
    return obj

  if type(obj) is list:
    obj = [apply_recursively_to_strings(x,func,callable_arg) for x in obj]
    return obj

  return obj

def reify_deps(deps,wildcards):
  "recursively replace wildcards in all strings with their values"
  func = lambda s: s.format(**wildcards.__dict__)
  res = apply_recursively_to_strings(deps,func,callable_arg=wildcards)
  return res 

def build_snakemake_deps():
  deps = SimpleNamespace()

  deps.train = SimpleNamespace()
  deps.train.traindir = "/projects/project-broaddus/devseg_2/ex7/{rawdir}/train{tid}/p{param_id}/{train_set}/"
  deps.train.best_model = deps.train.traindir + "m/net22.pt"
  deps.train.outputs = [deps.train.traindir, deps.train.best_model]

  deps.pred = SimpleNamespace()
  deps.pred.matches    = deps.train.traindir + "matches/{isbiname}/{pred}/t{time}.pkl"
  deps.pred.netpred    = deps.train.traindir + "pred/{isbiname}/{pred}/t{time}.tif"
  deps.pred.netpredmxz = deps.train.traindir + "mxz/{isbiname}/{pred}/t{time}.tif"
  deps.pred.pts        = deps.train.traindir + "pts/{isbiname}/{pred}/t{time}.pkl"
  deps.pred.traj_gt    = "/projects/project-broaddus/rawdata/{rawdir}/traj/{isbiname}/{pred}_traj.pkl"
  deps.pred.raw        = "/projects/project-broaddus/rawdata/{rawdir}/{isbiname}/{pred}/t{time}.tif"
  deps.pred.inputs     = [deps.pred.raw, deps.train.best_model, deps.pred.traj_gt]
  deps.pred.outputs    = [deps.pred.matches, deps.pred.netpred, deps.pred.netpredmxz, deps.pred.pts,]

  deps.eval = SimpleNamespace()
  deps.eval.total_matches = deps.train.traindir + "matches/{isbiname}/{pred}/total_matches.pkl"
  deps.eval.traj    = deps.train.traindir + "pts/{isbiname}/{pred}/traj.pkl"
  deps.eval.RESdir  = "/projects/project-broaddus/rawdata/{rawdir}/{isbiname}/{pred}_RES/"
  deps.eval.DET     = deps.eval.RESdir + "DET_log_ts{train_set}_tid{tid}_p{param_id}.txt"
  deps.eval.inputs  = [_all_matches,]
  deps.eval.outputs = [deps.eval.total_matches, deps.eval.traj, deps.eval.DET]
  
  ## now let's reify target

  p = SimpleNamespace()
  p.map1 = dict(celegans_isbi='Fluo-N3DH-CE',fly_isbi='Fluo-N3DL-DRO',trib_isbi_proj='Fluo-N3DL-TRIC',trib_isbi='Fluo-N3DL-TRIF',A549="Fluo-C3DH-A549",MDA231="Fluo-C3DL-MDA231",)
  # map2 = dict(celegans_isbi=celegans_isbi,fly_isbi=fly_isbi,trib_isbi_proj=trib_isbi_proj,trib_isbi=trib_isbi)
  p.rawdirs   = ['celegans_isbi', 'MDA231'] #, 'fly_isbi', 'trib_isbi_proj', 'trib_isbi','A549', 'MDA231', ]
  # p.isbinames = [p.map1[r] for r in p.rawdirs]
  p.preds     = ['01','02'] #,'02']
  p.trains    = ['02'] #,'02']
  p.kernxy_list  = [7] #[1,3,5,7,9]
  p.kernz_list   = [1,3] #[1,3,5]
  p.tid_list     = [1,]
  p.extra_params = list(product(p.kernxy_list,p.kernz_list))
  p.param_id = np.arange(len(p.extra_params))
  p.times = ['002']

  iterator    = product(p.rawdirs,p.preds,p.trains,p.tid_list,p.param_id,p.times)
  deps.target = [deps.eval.DET.format(rawdir=a,isbiname=p.map1[a],pred=b,train_set=c,tid=d,param_id=e,time=f) for a, b, c, d, e, f in iterator]
  print(deps.target)
  deps.p      = p

  return deps

# Wildcards = namedtuple(?)

def eg_wildcards():
  wildcards = SimpleNamespace(train_set='01',pred='01',time='all',tid=5,kernxy=7,kernz=1,rawdir='A549',isbiname='Fluo-C3DH-A549')
  wildcards = SimpleNamespace(rawdir="MDA231", tid="1", param_id="1", train_set="01", pred='02', isbiname="Fluo-C3DL-MDA231", time='003',)
  wildcards = SimpleNamespace(train_set='02',pred='02',time='003',tid=5,param_id=0,rawdir='MDA231',isbiname='Fluo-C3DL-MDA231',)
  return wildcards

def convert_snakemake_wcs(wildcards):
  deps = build_snakemake_deps()
  wildcards.tid      = int(wildcards.tid)
  wildcards.kernxy   = deps.p.extra_params[int(wildcards.param_id)][0]
  wildcards.kernz    = deps.p.extra_params[int(wildcards.param_id)][1]
  wildcards.isbiname = deps.p.map1[wildcards.rawdir]
  wildcards = add_defaults_to_namespace(wildcards,eg_wildcards()) ## adds time, pred, anything that we're missing
  return wildcards

## Below are entry points from snakemake
def test_isbi_train():
  w = eg_wildcards()
  deps = build_snakemake_deps()
  deps = reify_deps(deps.train,w)
  isbi_train([],deps.outputs,w)
  for d in deps.outputs:
    print(d)
    print(Path(d).exists())

def isbi_train(inputs,outputs,wildcards):
  
  wildcards = convert_snakemake_wcs(wildcards)
  img_meta  = data_specific._get_img_meta(wildcards)
  config    = detector.config(img_meta)
  
  ## update variables we want to control globally via snakemake
  config.savedir = outputs[0] #Path(f"/projects/project-broaddus/devseg_2/ex7/{w.rawdir}/train{w.tid}/{w.kernxy}_{w.kernz}/{w.train_set}/")
  w = wildcards
  config.sigmas  = np.array([w.kernz,w.kernxy,w.kernxy])

  loader = SimpleNamespace()
  loader.input_dir     = Path(f"/projects/project-broaddus/rawdata/{w.rawdir}/{w.isbiname}/{w.train_set}/")
  loader.traj_gt_train = Path(f"/projects/project-broaddus/rawdata/{w.rawdir}/traj/{w.isbiname}/{w.train_set}_traj.pkl")
  maxtime_train        = len(list(loader.input_dir.glob("*.tif")))
  _times = np.linspace(0,maxtime_train - 1,8).astype(np.int)
  loader.valitimes     = _times[[2,5]]
  loader.traintimes    = _times[[0,1,3,4,6,7]]  

  ## update variables in a data-dependent way
  loader,config = data_specific._specialize_train(wildcards,loader,config)

  config.load_train_and_vali_data = lambda _config : isbi_tools.load_isbi_train_and_vali(loader,_config)

  T = detector.train_init(config)
  detector.train(T)

def test_isbi_predict():
  w = eg_wildcards()
  deps = build_snakemake_deps()
  deps = reify_deps(deps.pred,w)
  deps.inputs[1] = Path(deps.inputs[1]).parent / "net22.pt"

  for d in deps.outputs:
    Path(d).unlink()
    print(Path(d).exists())
  isbi_predict(deps.inputs, deps.outputs, w)
  for d in deps.outputs:
    print(d)
    print(Path(d).exists())

def isbi_predict(inputs,outputs,wildcards):
  wildcards = convert_snakemake_wcs(wildcards)
  img_meta  = data_specific._get_img_meta(wildcards)
  config    = detector.config(img_meta)

  ## update config with global vars
  config.best_model = inputs[1]
  
  ## data-dependent updates
  config = data_specific._specialize_predict(wildcards,config)

  pts_gt  = load(inputs[2])[int(wildcards.time)]
  raw     = load(inputs[0])
  net     = detector._load_net(config)
  pred    = detector.predict_raw(config,net,raw)
  pts     = detector.predict_pts(config,pred)
  matches = detector.predict_matches(config,pts_gt,pts)

  out_matches, out_netpred, out_netpredmxz, out_pts, = outputs
  save(pred,out_netpred)
  save(pred.max(0),out_netpredmxz)
  save(pts,out_pts)
  save(matches,out_matches)


def test_isbi_evaluate():
  w = eg_wildcards()
  print(w)
  deps = build_snakemake_deps()
  deps = reify_deps(deps.eval,w)

  # for d in deps.outputs:
  #   # Path(d).unlink()
  #   print(Path(d).exists())
  isbi_evaluate(deps.inputs, deps.outputs, w)
  for d in deps.outputs[0]:
    print(d)
    print(Path(d).exists())

def isbi_evaluate(inputs,outputs,wildcards):
  wildcards = convert_snakemake_wcs(wildcards)
  img_meta  = data_specific._get_img_meta(wildcards)
  config    = detector.config(img_meta)

  # deps = build_snakemake_deps()
  # deps = reify_deps(deps,wildcards)

  match_list   = [load(x) for x in inputs[0]]
  match_scores = point_matcher.listOfMatches_to_Scores(match_list)
  save(match_scores, outputs[0])
  print("SCORES: ", match_scores)  
  traj = [load(x.replace("matches/","pts/")) for x in inputs[0]]
  save(traj, outputs[1])

  w = wildcards
  RESdir  = f"/projects/project-broaddus/rawdata/{w.rawdir}/{w.isbiname}/{w.pred}_RES/"
  eg_img  = f"/projects/project-broaddus/rawdata/{w.rawdir}/{w.isbiname}/{w.pred}/t000.tif"

  detector.rasterize_detections(config, traj, load(eg_img).shape, RESdir, pts_transform = lambda x: x,)
  base_dir = Path(RESdir).parent
  isbi_tools.evaluate_isbi_DET(base_dir,outputs[2],pred=wildcards.pred,fullanno=False)

  # detector.total_matches(config.evaluator)
  # detector.rasterize_isbi_detections(config.evaluator)
  # detector.evaluate_isbi_DET(config.evaluator)

def special():
  print("test!!!!")


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
The problem is that we designed d_isbi to contain all the info necessary to run training (once) and prediction (once), but not more or less.
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

Changed name to "experiments.py". This is a much better fit. This is where we actually apply methods to data and generate results.

What is the interface from the Snakefile?
We have a matching function in experiments for each rule in the Snakemake file. We pass that function the wildcards and maybe the input/output.
should experiments have full control over savedir for pred,pts,matches, etc independently? or should the method decide that?
the function args have been tricky to get right, which makes me think this should be the job of experiments, not of detector.

The "loader" object that experiments uses to communicate with isbi_tools smells bad.
It is usually filled with irrelevant information.
It is buil by isbitools, but then is passed around internally in experiments a lot.
It's purpose overlaps with the purpose of deps, i.e. to manage file names and dependencies between them in a generic way via variables.
It takes the wildcards argument! Only thing in isbi_tools to do that...
Only place where wildcards goes outside of experiments and Snakefile.

only exists for coms between experiments and one function in isbi_tools.

Q: how should we add the experiments on other random datasets to our current set of experiments?
- new experiments file, snakemake wildcard control. vs data-specific config control.
  - data-specific config knows about raw/gt location, image properties, and method's data-specific params.
  - snakemake approach: data-specific config only knows about image properties and method's params, not file locations / dir structure.
  - how do we know how many time points to use? look at gt files, not raw files.
  - what if meta-data is acquisition dependent? (a la in c. elegans where timestep and acquisition length differ?)
- standardize directory structure to match isbi data, add to snakemake
- 


"""








