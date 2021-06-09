"""
WIP
recursively rescale a folder full of RAW images and associated GT labels.
GT images with uint labels are converted to list-of-dict format with various object properties.
"""


from experiments_common import iterdims
from glob import glob
from matplotlib import pyplot as plt
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import label,zoom
from segtools.ns2dir import load,save,flatten_sn,toarray
from segtools.numpy_utils import normalize3
from segtools.point_tools import trim_images_from_pts2
from skimage.feature  import peak_local_max
from skimage.measure  import regionprops, regionprops_table
from time import time
from types import SimpleNamespace
import ipdb
import itertools
import matplotlib
import numpy as np
import os
import pandas
import re
import shutil
import zarr


try:
    import gputools
except ImportError as e:
    print("Can't import gputools on non-gpu node...\n", e)

savedir_global = Path("/projects/project-broaddus/devseg_2/expr/")



def swap_root(path, oldroot, newroot):
  assert str(oldroot) in str(path)
  path = str(path).replace(str(oldroot),str(newroot))
  return Path(path)

def rescale_dataset(oldroot,newroot,scale=(1,0.5,0.5),crop=False,ltps=None):
  """
  Convert an ISBI dataset from TIFF to ZARR, [crop and] rescale it.
  if ltps exists AND crop==True then use ltps to determine the data crop first...
  """

  oldroot = Path(oldroot)
  newroot = Path(newroot)

  assert oldroot.exists()
  # assert not newroot.exists()
  newroot.mkdir(exist_ok=True, parents=True)

  for _dir,_subdirs,_files in os.walk(oldroot):
    _dir = Path(_dir)
    fs = [x for x in sorted(_files) if x.endswith('.tif')] #[-3:]

    if str(_dir).endswith("RES"): continue

    for _f in fs:
      oldname = _dir / _f
      newname = swap_root(oldname,oldroot,newroot).with_suffix(".zarr")

      print(oldname)
      print(newname)
      if newname.exists(): continue
      x = load(oldname)
      _scale = scale[-x.ndim:]

      _dt = x.dtype
      if 'int' in str(_dt) and 'GT' in str(_dir):
        table_full = DataFrame([{'label':rp.label,'bbox':rp.bbox,'centroid':rp.centroid,'slice':rp.slice, 'area':rp.area} for rp in regionprops(x)])
        # table_full.to_pickle(oldname.with_suffix(".pkl"))

        if x.ndim==3:
          x = gputools.scale(x,_scale,interpolation='nearest').astype(_dt)
        else:
          x = zoom(x,_scale,order=0).astype(_dt)

        zarr.save_array(str(newname),x)
        table_rescaled = DataFrame([{'label':rp.label,'bbox':rp.bbox,'centroid':rp.centroid,'slice':rp.slice, 'area':rp.area} for rp in regionprops(x)])
        # ipdb.set_trace()
        if len(table_rescaled)>0:
          table_rescaled = pandas.merge(table_full,table_rescaled,on='label',how='outer')
        else:
          table_rescaled = table_full.rename(columns = lambda c: c+"_y" if c!='label' else c)

        table_rescaled.to_pickle(newname.with_suffix(".pkl"))
      else:
        ## is RAW

        if x.ndim==3:
          x = gputools.scale(x,_scale,interpolation='linear').astype(_dt)
        else:
          x = zoom(x,_scale,order=0).astype(_dt)

        zarr.save_array(str(newname),x)

def myzoom(img,scale):
  _dt = img.dtype
  if x.ndim==2 and 'int' in str(_dt):
    img = zoom(img,scale,order=0).astype(_dt)
  if x.ndim==2 and 'float' in str(_dt):
    img = zoom(img,scale,order=1).astype(_dt)
  if x.ndim==3 and 'int' in str(_dt):
    img = gputools.scale(img,scale,interpolation='nearest').astype(_dt)
  if x.ndim==3 and 'float' in str(_dt):
    img = gputools.scale(img,scale,interpolation='linear').astype(_dt)
  return img

def test_rescale():
  myname = "H157"
  isbiname = "Fluo-C3DH-H157"
  dataset = "01"

  oldroot = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{dataset}_GT/"
  newroot = f"/projects/project-broaddus/rawdata/{myname}/rescaled/{isbiname}/{dataset}_GT/"

  oldroot = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/"
  newroot = f"/projects/project-broaddus/rawdata/{myname}/rescaled/{isbiname}/"

  # oldroot = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/"
  # newroot = f"/projects/project-broaddus/rawdata/{myname}/rescaled/{isbiname}/"

  tic = time()
  rescale_dataset(oldroot,newroot,scale=(0.5,0.25,0.25))
  print("TIME rescale: ", time()-tic)

  # img = load("/projects/project-broaddus/rawdata/{myname}/{isbiname}/{dataset}_GT/SEG/man_seg122.tif")
  # new = load("/projects/project-broaddus/rawdata/{myname}/rescaled/{isbiname}/{dataset}_GT/SEG/man_seg122.zarr")
  # tab = load("/projects/project-broaddus/rawdata/{myname}/rescaled/{isbiname}/{dataset}_GT/SEG/man_seg122.pkl")

  # img = load("/projects/project-broaddus/rawdata/{myname}/{isbiname}/{dataset}_GT/TRA/man_track250.tif")
  # new = load("/projects/project-broaddus/rawdata/{myname}/rescaled/{isbiname}/{dataset}_GT/TRA/man_track250.zarr")
  # tab = load("/projects/project-broaddus/rawdata/{myname}/rescaled/{isbiname}/{dataset}_GT/TRA/man_track250.pkl")

  # def f(name):
  #   tab = load(name)
  #   tab['name'] = name
  #   return tab

  # tic = time()
  # full = pandas.concat([f(n) for n in sorted(glob(f"/projects/project-broaddus/rawdata/{H157}/rescaled/{isbiname}/{dataset}_GT/TRA/man_track*.pkl"))])
  # print("TIME: ", time()-tic)

  # ipdb.set_trace()

# ['coords_x','coords_y',]

# ['label', 'bbox-0_x', 'bbox-1_x', 'bbox-2_x', 'bbox-3_x', 'coords_x',
#        'slice_x', 'area_x', 'bbox-0_y', 'bbox-1_y', 'bbox-2_y', 'bbox-3_y',
#        'coords_y', 'slice_y', 'area_y']





