import detector
import numpy as np
from subprocess import run
from types import SimpleNamespace
from skimage.measure      import regionprops
from pathlib import Path
from segtools.ns2dir import load, save, flatten


def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

def load_isbi_train_and_vali(loader, config,):
  print("Training Times : ", loader.traintimes)
  print("Vali Times : ", loader.valitimes)
  vd = _load_isbi_training_data(loader.valitimes, loader, config)
  td = _load_isbi_training_data(loader.traintimes,loader, config)
  return vd,td

def _load_isbi_training_data(times,loader,config):
  """
  Conforms to the interface necessary for use in detector
  attributes: input,target,gt,axes=='tczyx',dims,in_samples,in_chan,in_space
  """
  d = SimpleNamespace()
  d.input  = np.array([load(loader.input_dir / f"t{n:03d}.tif") for n in times])
  d.input  = config.norm(d.input)
  try:
    d.gt   = load(loader.traj_gt_train)
  except:
    d.gt   = [mantrack2pts(load(loader.TRAdir / f"man_track{n:03d}.tif")) for n in times]
  d.gt     = np.array([config.pt_norm(x) for x in d.gt])
  d.target = detector._pts2target(d.gt,d.input[0].shape,config)
  d.target = d.target[:,None]
  d.input  = d.input[:,None]
  d.axes = "TCZYX"
  d.dims = {k:v for k,v in zip(d.axes, d.input.shape)}
  d.in_samples  = d.input.shape[0]
  d.in_chan     = d.input.shape[1]
  d.in_space    = np.array(d.input.shape[2:])

  return d

def evaluate_isbi_DET(base_dir,detname,pred='01',fullanno=True):
  "evalid is a unique ID that prevents us from overwriting DET_log files from different experiments predicting on the same data."
  fullanno = '' if fullanno else '0'
  DET_command = f"""
  time /projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure {base_dir} {pred} 3 {fullanno}
  cd {base_dir}/{pred}_RES/
  mv DET_log.txt {detname}
  """
  run([DET_command],shell=True)

def rasterize_detections(sigmas, traj, imgshape):
  for i in range(len(traj)):
    pts = traj[i]
    kerns = [np.zeros(3*sigmas) + j + 1 for j in range(len(pts))]
    lab   = conv_at_pts_multikern(pts,kerns,shape).astype(np.uint16)
    return lab





notes = """
We need a better idea for how to load training data.
The interface is going to work like this:
Instead of passing a bunch of args and a func into the config, then having the config call the loader with those args we're just going to pass a closure argument.
config.loader : config -> vd,td,meta
or maybe 
config.loader : () -> vd,td
This is a closure we can build in `experiments`.
Is there any reason to wait to pass config later?
Then we can change, e.g. the kernel size _after_ we've built the loader...

We need a function that builds td (or vd) from list of times & dir
We need a function that builds td & vd
The isbi_loader, load_isbi_train_and_vali, and _load_isbi_training_data need to be a coherent thing.

The loader is really two things with one name.
The loader is for storing the arguments we pass to the load_isbi_train_and_vali function.
But it's also a way of getting names to a variety of isbi things via wildcards.
This second purpose is really the job of deps in experiments. So let's get rid of the isbi_loader function and build just the piece we need in experiments.




"""