from types import SimpleNamespace
import re
from pathlib import Path
import numpy as np
from tifffile import imread


names = ["Fluo-C3DH-A549",
    "Fluo-C3DH-H157",
    "Fluo-C3DL-MDA231",
    "Fluo-N3DH-CE",
    "Fluo-N3DH-CHO",
    "Fluo-N3DL-DRO",
    "Fluo-N3DL-TRIC",
    # "Fluo-N3DL-TRIF",
    "Fluo-C3DH-A549-SIM",
    "Fluo-N3DH-SIM+",
    "BF-C2DL-HSC" ,
    "BF-C2DL-MuSC" ,
    "DIC-C2DH-HeLa" ,
    "Fluo-C2DL-MSC" ,
    "Fluo-N2DH-GOWT1" ,
    "Fluo-N2DL-HeLa" ,
    "PhC-C2DH-U373" ,
    "PhC-C2DL-PSC" ,
    "Fluo-N2DH-SIM+" ,]

def run():
  for name in names:
    for dset in ['01','02']:
      print(name, dset, '  ', end='')
      info = get_isbi_info(name,dset)
      print(info.start, info.stop)

def get_isbi_info(isbiname,dataset):
  d = SimpleNamespace()
  # d.index = [x[1] for x in isbi_datasets].index(isbiname)
  # d.myname     = myname
  d.isbiname   = isbiname
  d.dataset    = dataset
  d.isbi_dir   = Path(f"/projects/project-broaddus/rawdata/isbi_train/{isbiname}/")
  trackfiles   = sorted((d.isbi_dir/(dataset+"_GT/TRA/")).glob("man_track*.tif"))
  # d.rawfiles     = sorted((d.isbi_dir/dataset).glob("t*.tif"))

  d.timebounds = [re.search(r'man_track(\d+)\.tif',str(x)).group(1) for x in trackfiles]
  d.timebounds = [d.timebounds[0],d.timebounds[-1]]
  d.ndigits    = len(d.timebounds[0])
  # d.shape = imread(trackfiles[0]).shape

  n_raw = str(trackfiles[0])[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  n_raw = str(trackfiles[0]) #[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
  # d.shape = imread(n_raw).shape
  
  d.start,d.stop  = int(d.timebounds[0]), int(d.timebounds[-1])+1
  # stop  = 5
  d.ndim = 2 if '2D' in isbiname else 3
  # scale = np.array(isbi_scales[isbiname])[::-1]
  # d.scale = scale / scale[-1]
  d.penalize_FP = '0' if isbiname in ["Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "Fluo-N3DL-TRIF"] else ''
  d.maskname  = "mask{time:03d}.tif" if d.ndigits==3 else "mask{time:04d}.tif"
  d.man_track = "man_track{time:03d}.tif" if d.ndigits==3 else "man_track{time:04d}.tif"
  d.rawname   = "t{time:03d}.tif" if d.ndigits==3 else "t{time:04d}.tif"
  # d.man_seg   = {(3,3): "man_seg_{time:03d}_{z:03d}.tif", 
  # d.raw_full  = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{dataset}/" + d.rawname
  # d.lab_full  = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{dataset}_GT/TRA/" + d.man_track
  return d