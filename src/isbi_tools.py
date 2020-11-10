# import detector
import numpy as np
# from subprocess import run
from types import SimpleNamespace
# from skimage.measure      import regionprops
from pathlib import Path
from segtools.ns2dir import load, save, toarray
import re



# def mantrack2pts(mantrack): return np.array([r.centroid for r in regionprops(mantrack)],np.int)

# def load_isbi_train_and_vali(loader, config,):
#   print("Training Times : ", loader.traintimes)
#   print("Vali Times : ", loader.valitimes)
#   vd = _load_isbi_training_data(loader.valitimes, loader, config)
#   td = _load_isbi_training_data(loader.traintimes,loader, config)
#   return vd,td

# def _load_isbi_training_data(times,loader,config):
#   """
#   Conforms to the interface necessary for use in detector
#   attributes: input,target,gt,axes=='tczyx',dims,in_samples,in_chan,in_space
#   """
#   d = SimpleNamespace()
#   d.input  = np.array([load(loader.input_dir / f"t{n:03d}.tif") for n in times])
#   d.input  = config.norm(d.input)
#   try:
#     d.gt   = load(loader.traj_gt_train)
#   except:
#     d.gt   = [mantrack2pts(load(loader.TRAdir / f"man_track{n:03d}.tif")) for n in times]
#   d.gt     = np.array([config.pt_norm(x) for x in d.gt])
#   d.target = detector._pts2target(d.gt,d.input[0].shape,config)
#   d.target = d.target[:,None]
#   d.input  = d.input[:,None]
#   d.axes = "TCZYX"
#   d.dims = {k:v for k,v in zip(d.axes, d.input.shape)}
#   d.in_samples  = d.input.shape[0]
#   d.in_chan     = d.input.shape[1]
#   d.in_space    = np.array(d.input.shape[2:])
  # return d

# def evaluate_isbi_DET(base_dir,detname,pred='01',fullanno=True):
#   "evalid is a unique ID that prevents us from overwriting DET_log files from different experiments predicting on the same data."
#   fullanno = '' if fullanno else '0'
#   DET_command = f"""
#   time /projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure {base_dir} {pred} 3 {fullanno}
#   cd {base_dir}/{pred}_RES/
#   mv DET_log.txt {detname}
#   """
#   run([DET_command],shell=True)

# def rasterize_detections(sigmas, traj, imgshape):
#   for i in range(len(traj)):
#     pts = traj[i]
#     kerns = [np.zeros(3*sigmas) + j + 1 for j in range(len(pts))]
#     lab   = conv_at_pts_multikern(pts,kerns,shape).astype(np.uint16)
#     return lab


## proof of principle for every dataset
isbi_datasets = [
  ("HSC",             "BF-C2DL-HSC"),
  ("MuSC",            "BF-C2DL-MuSC"),
  ("HeLa",            "DIC-C2DH-HeLa"),
  ("MSC",             "Fluo-C2DL-MSC"),
  ("A549",            "Fluo-C3DH-A549"),
  ("A549-SIM",        "Fluo-C3DH-A549-SIM"),
  ("H157",            "Fluo-C3DH-H157"),
  ("MDA231",          "Fluo-C3DL-MDA231"),
  ("GOWT1",           "Fluo-N2DH-GOWT1"),
  ("SIM+",            "Fluo-N2DH-SIM+"),
  ("HeLa",            "Fluo-N2DL-HeLa"),
  ("celegans_isbi",   "Fluo-N3DH-CE"),
  ("hampster",        "Fluo-N3DH-CHO"),
  ("SIM+",            "Fluo-N3DH-SIM+"),
  ("fly_isbi",        "Fluo-N3DL-DRO"),
  ("trib_isbi_proj",  "Fluo-N3DL-TRIC"),
  ("U373",            "PhC-C2DH-U373"),
  ("PSC",             "PhC-C2DL-PSC"),
  ("trib_isbi/crops_2xDown", "Fluo-N3DL-TRIF"),
  ]

## WARNING: IN XYZ ORDER!!!
isbi_scales = {
  "Fluo-C3DH-A549":      (0.126, 0.126, 1.0),
  "Fluo-C3DH-H157":      (0.126, 0.126, 0.5),
  "Fluo-C3DL-MDA231":    (1.242, 1.242, 6.0),
  "Fluo-N3DH-CE":        (0.09 , 0.09, 1.0),
  "Fluo-N3DH-CHO":       (0.202, 0.202, 1.0),
  "Fluo-N3DL-DRO":       (0.406, 0.406, 2.03),
  "Fluo-N3DL-TRIC":      (1.,1.,1.), # NA dueto cartographic projections
  "Fluo-N3DL-TRIF":      (0.38 , 0.38, 0.38),
  "Fluo-C3DH-A549-SIM":  (0.126, 0.126, 1.0),
  "Fluo-N3DH-SIM+":      (0.125, 0.125, 0.200),
  "BF-C2DL-HSC" :        (0.645 ,0.645),
  "BF-C2DL-MuSC" :       (0.645 ,0.645),
  "DIC-C2DH-HeLa" :      (0.19 ,0.19),
  "Fluo-C2DL-MSC" :      (0.3 ,0.3), # (0.3977 x 0.3977) for dataset 2?,
  "Fluo-N2DH-GOWT1" :    (0.240 ,0.240),
  "Fluo-N2DL-HeLa" :     (0.645 ,0.645),
  "PhC-C2DH-U373" :      (0.65 ,0.65),
  "PhC-C2DL-PSC" :       (1.6 ,1.6),
  "Fluo-N2DH-SIM+" :     (0.125 ,0.125),
  }

def get_isbi_info(myname,isbiname,dataset):
  d = SimpleNamespace()
  d.myname     = myname
  d.isbiname   = isbiname
  d.dataset    = dataset
  d.isbi_dir   = Path(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/")
  d.trackfiles = sorted((d.isbi_dir/(dataset+"_GT/TRA/")).glob("man_track*.tif"))
  d.timebounds = [re.search(r'man_track(\d+)\.tif',str(x)).group(1) for x in d.trackfiles]
  d.ndigits    = len(d.timebounds[0])
  d.shape = load(d.trackfiles[0]).shape
  d.start,d.stop  = int(d.timebounds[0]), int(d.timebounds[-1])+1
  # stop  = 5
  d.ndim = 2 if '2D' in isbiname else 3
  scale = np.array(isbi_scales[isbiname])[::-1]
  d.scale = scale / scale[-1]
  d.ignore_FP = '' if isbiname not in ["Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF"] else '0'

  return d



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

Tue Nov 10 11:10:31 2020

All the loading/centerpoint finding/kern finding, saving/rasterization stuff has been moved into tracking.py
All the DET/TRA evaluation has been moved up into experiments[2].py
But here we store dataset specific info/lists of names and ways of querying a given saved dataset.


"""