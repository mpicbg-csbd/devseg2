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


## (myname, isbiname) | my order | official ISBI order
isbi_names = [
  "BF-C2DL-HSC",           #  0     0 
  "BF-C2DL-MuSC",          #  1     1 
  "DIC-C2DH-HeLa",         #  2     2 
  "Fluo-C2DL-MSC",         #  3     3 
  "Fluo-C3DH-A549",        #  4     4 
  "Fluo-C3DH-A549-SIM",    #  5    16 
  "Fluo-C3DH-H157",        #  6     5 
  "Fluo-C3DL-MDA231",      #  7     6 
  "Fluo-N2DH-GOWT1",       #  8     7 
  "Fluo-N2DH-SIM+",        #  9    17 
  "Fluo-N2DL-HeLa",        # 10     8 
  "Fluo-N3DH-CE",          # 11     9 
  "Fluo-N3DH-CHO",         # 12    10 
  "Fluo-N3DH-SIM+",        # 13    18 
  "Fluo-N3DL-DRO",         # 14    11 
  "Fluo-N3DL-TRIC",        # 15    12 
  "PhC-C2DH-U373",         # 16    14 
  "PhC-C2DL-PSC",          # 17    15 
  "Fluo-N3DL-TRIF",        # 18    13 
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

# import argparse, sys

if __name__=="__main__":

  # parser = argparse.ArgumentParser(description='Build bash scripts from templates...')
  # args = parser.parse_args()

  ## iterate over all challenge datasets
  for i in range(2*19):
    (p0,p1), pid = parse_pid(i, [2,19])
    isbiname = isbi_names[p1]
    dataset_pred  = ["01","02",][p0]
    
    # _ , pid_train = parse_pid([0,p1], [3,19]) # 
    # pid_train = p1 ## This works, because train() uses [3,19] so pid 000..018 are the models trained on both datasets.
    # weightname_in = f"../expr/e26_isbidet/train/pid{pid_train:03d}/m/best_weights_loss.pt"

    
    weightname_out = f"{isbiname}-01+02_weights.pt"
    template = open("template.sh",'r').read()
    params   = pickle.load(open(f"trainparams/{isbiname}-01+02_params.pkl",'rb'))
    Ndim     = len(params.zoom)
    scale    = np.array(isbi_scales[isbiname])[::-1]
    scale    = scale / scale[-1]

    # for local testing
    indir  = f"/projects/project-broaddus/rawdata/isbi_challenge/{isbiname}/{dataset_pred}"
    outdir = f"/projects/project-broaddus/rawdata/isbi_challenge_out/{isbiname}/{dataset_pred}_RES"

    ## Manual extensions
    if isbiname in ["Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF",]:
      mantrack_t0 = str(Path(indir).parent / f"{dataset_pred}_GT/TRA/man_track000.tif")
    else:
      mantrack_t0 = "None"

    if isbiname == "Fluo-N3DL-TRIF": params.zoom = (0.5 , 0.5 , 0.5)


    radius = np.max(np.array(params.nms_footprint) / params.zoom) * 2

    if "PhC-C2DL-PSC" in indir:
      print("PhC-C2DL-PSC : ", params.nms_footprint)
      params.nms_footprint = (3,3)
      radius = 7
    if "Fluo-N3DL-DRO" in indir:
      radius = 7
    if "Fluo-N3DL-TRIF" in indir:
      radius = 10
    if "Fluo-N2DH-SIM+" in indir:
      radius = 20


    ## ignore errors when one object is within evalBorder of XY image boundary
    if isbiname in ["DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-C3DH-H157", "Fluo-N2DH-GOWT1", "Fluo-N3DH-CE", "Fluo-N3DH-CHO", "PhC-C2DH-U373",]:
      params.evalBorder = (0,50,50) if "3D" in isbiname else (50,50)
    elif isbiname in ["BF-C2DL-HSC", "BF-C2DL-MuSC", "Fluo-C3DL-MDA231", "Fluo-N2DL-HeLa", "PhC-C2DL-PSC",]:
      params.evalBorder = (0,25,25) if "3D" in isbiname else (25,25)
    elif isbiname in ["Fluo-C3DH-A549", "Fluo-N2DH-SIM+", "Fluo-N3DH-SIM+", "Fluo-C3DH-A549-SIM", "Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF"]:
      params.evalBorder = (0,0,0) if "3D" in isbiname else (0,0)

    temp = {
      # "<script>" : script,
      "<indir>": indir,
      "<outdir>": outdir,
      "<weightname>":weightname_out,
      "<zoom>": ("{:.3f} "*Ndim).format(*params.zoom), 
      "<nms_footprint>": ("{:3d} "*Ndim).format(*params.nms_footprint),
      "<scale>" : ("{:.3f} "*Ndim).format(*scale),
      "<radius>" : str(radius), 
      "<mantrack_t0>": mantrack_t0,
      "<evalBorder>": ("{:3d} "*Ndim).format(*params.evalBorder),
    }

    for x,y in temp.items():
      template = template.replace(x,y)

    fout = f"{isbiname}-{dataset_pred}.sh"
    with open(fout,'w') as _fi:
      _fi.write(template)

    run([f"chmod +x {fout}"] , shell=True)