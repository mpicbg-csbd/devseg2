"""
Make the bash scripts required for ISBI CSC/CTC compliance from `template.sh`.
Move the weights appropriate for each model into the `models/` folder.
"""

from pathlib import Path
from glob import glob
import os
import shutil
import pickle
from subprocess import run
import numpy as np

# from experiments_common import rchoose, partition, shuffle_and_split, parse_pid, iterdims, savedir_global

def parse_pid(pid_or_params,dims):
  if hasattr(pid_or_params,'__len__') and len(pid_or_params)==len(dims):
    params = pid_or_params
    pid = np.ravel_multi_index(params,dims)
  elif 'int' in str(type(pid_or_params)):
    pid = pid_or_params
    params = np.unravel_index(pid,dims)
  else:
    a = hasattr(pid_or_params,'__len__')
    b = len(pid_or_params)==len(dims)
    print("ERROR", a, b)
    assert False
  return params, pid


isbi_datasets = [
  ("HSC",             "BF-C2DL-HSC"),           #  0     0 
  ("MuSC",            "BF-C2DL-MuSC"),          #  1     1 
  ("HeLa",            "DIC-C2DH-HeLa"),         #  2     2 
  ("MSC",             "Fluo-C2DL-MSC"),         #  3     3 
  ("A549",            "Fluo-C3DH-A549"),        #  4     4 
  ("A549-SIM",        "Fluo-C3DH-A549-SIM"),    #  5    16 
  ("H157",            "Fluo-C3DH-H157"),        #  6     5 
  ("MDA231",          "Fluo-C3DL-MDA231"),      #  7     6 
  ("GOWT1",           "Fluo-N2DH-GOWT1"),       #  8     7 
  ("SIM+",            "Fluo-N2DH-SIM+"),        #  9    17 
  ("HeLa",            "Fluo-N2DL-HeLa"),        # 10     8 
  ("celegans_isbi",   "Fluo-N3DH-CE"),          # 11     9 
  ("hampster",        "Fluo-N3DH-CHO"),         # 12    10 
  ("SIM+",            "Fluo-N3DH-SIM+"),        # 13    18 
  ("fly_isbi",        "Fluo-N3DL-DRO"),         # 14    11 
  ("trib_isbi_proj",  "Fluo-N3DL-TRIC"),        # 15    12 
  ("U373",            "PhC-C2DH-U373"),         # 16    14 
  ("PSC",             "PhC-C2DL-PSC"),          # 17    15 
  # ("trib_isbi", "Fluo-N3DL-TRIF"),      # 18    13 
  ("trib_isbi/crops_2xDown", "Fluo-N3DL-TRIF"), # 18    13 
  ]

## WARNING: IN XYZ ORDER!!!
isbi_scales = {
  "Fluo-C3DH-A549":      (0.126, 0.126, 1.0),
  "Fluo-C3DH-H157":      (0.126, 0.126, 0.5),
  "Fluo-C3DL-MDA231":    (1.242, 1.242, 6.0),
  "Fluo-N3DH-CE":        (0.09 , 0.09, 1.0),
  "Fluo-N3DH-CHO":       (0.202, 0.202, 1.0),
  "Fluo-N3DL-DRO":       (0.406, 0.406, 2.03),
  "Fluo-N3DL-TRIC":      (1.,1.,1.), # NA due to cartographic projections
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

import argparse, sys

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Build bash scripts from templates...')
  parser.add_argument('--target-local',  dest='remote', action='store_false')
  parser.add_argument('--target-remote', dest='remote', action='store_true')
  parser.set_defaults(remote=False)

  args = parser.parse_args()

  target_remote = args.remote
  print(f"target_remote = {target_remote}")

  ## iterate over all challenge datasets
  for i in range(2*19):
    (p0,p1), pid = parse_pid(i, [2,19])
    # _ , pid_train = parse_pid([0,p1], [3,19]) # 
    pid_train = p1

    weightname_in = f"../expr/e26_isbidet/train/pid{pid_train:03d}/m/best_weights_loss.pt"
    isbiname = isbi_datasets[p1][1]
    myname   = isbi_datasets[p1][0]
    dataset_pred  = ["01","02",][p0]
    dataset_train = "01+02"

    # if p0!=0: continue ## TODO: FIXME!!!
    
    weightname_out = f"{isbiname}-{dataset_train}_weights.pt"
    
    # os.remove(f"models/*")
    shutil.copy(weightname_in, "models/"+weightname_out)
    template = open("template.sh",'r').read()
    params = pickle.load(open(f"/projects/project-broaddus/devseg_2/expr/e26_isbidet/train/pid{pid:03d}/params.pkl",'rb'))
    Ndim = len(params.zoom)
    scale = np.array(isbi_scales[isbiname])[::-1]
    scale = scale / scale[-1]


    # for local testing
    indir  = f"/projects/project-broaddus/rawdata/isbi_challenge/{isbiname}/{dataset_pred}"
    outdir = f"/projects/project-broaddus/rawdata/isbi_challenge_out/{isbiname}/{dataset_pred}_RES"
    # for prediction on ISBI server
    if target_remote:
      indir  = f"../{isbiname}/{dataset_pred}"
      outdir = f"../{isbiname}/{dataset_pred}_RES"

    ## Manual extensions
    if isbiname in ["Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF",]:
      mantrack_t0 = str(Path(indir).parent / "01_GT/TRA/man_track000.tif")
    else:
      mantrack_t0 = "None"

    if isbiname == "Fluo-N3DL-TRIF": params.zoom = (0.5 , 0.5 , 0.5)

    temp = {
      "<indir>": indir,
      "<outdir>": outdir,
      "<weightname>":weightname_out,
      "<zoom>": ("{:.3f} "*Ndim).format(*params.zoom), 
      "<nms_footprint>": ("{:3d} "*Ndim).format(*params.nms_footprint),
      "<scale>" : ("{:.3f} "*Ndim).format(*scale),
      "<mantrack_t0>": mantrack_t0,
    }

    for x,y in temp.items():
      template = template.replace(x,y)

    fout = f"{isbiname}-{dataset_pred}.sh"
    with open(fout,'w') as _fi:
      _fi.write(template)

    run([f"chmod +x {fout}"] , shell=True)