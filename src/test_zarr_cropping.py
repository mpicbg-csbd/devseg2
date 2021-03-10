from segtools.ns2dir import load,save,toarray
from segtools.point_tools import trim_images_from_pts2

def run():
  tif = load("/projects/project-broaddus/rawdata/trib_isbi/Fluo-N3DL-TRIF/01/t000.tif")
  img = load("/projects/project-broaddus/rawdata/zarr/trib_isbi/Fluo-N3DL-TRIF/01/t000.zarr")
  pts = load("/projects/project-broaddus/rawdata/trib_isbi/traj/Fluo-N3DL-TRIF/01_traj.pkl")[0]
  pts2,ss = trim_images_from_pts2(pts,border=(10,10,10))
  img2 = img[ss]
  tif2 = tif[ss]
  img2b = img[ss]
  return img
