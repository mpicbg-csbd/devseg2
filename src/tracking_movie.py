# from segtools.rsync import rsync_pull2

from types import SimpleNamespace
from segtools.ns2dir import load,save,toarray
import numpy as np
import matplotlib.pyplot as plt
# plt.ion(); plt.show()
# import ipdb
from pathlib import Path
# import viewer_tools

# import pandas as pd
# import altair

# import napari

# from isbi_score_downloader import isbi_data

def norm_affine01(x,lo,hi):
  if hi==lo: 
    return x-hi
  else: 
    return (x-lo)/(hi-lo)

def norm_percentile01(x,p0,p1):
  lo,hi = np.percentile(x,[p0,p1])
  return norm_affine01(x,lo,hi)

def zoom_pts(pts,scale):
  """
  rescale pts to be consistent with scipy.ndimage.zoom(img,scale)
  """
  # assert type(pts) is np.ndarray
  pts = pts+0.5                         ## move origin from middle of first bin to left boundary (array index convention)
  pts = pts * scale                     ## rescale
  pts = pts-0.5                         ## move origin back to middle of first bin
  pts = np.round(pts).astype(np.uint32) ## binning
  return pts


# d_expr = Path("/Users/broaddus/Desktop/project-broaddus/devseg_2/expr/")

import tracking
from glob import glob
from scipy.ndimage import zoom

def trackingvideo2D():
  # ltpsfile_remote = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N2DH-GOWT1/01_RES/ltps.npy"
  # rsync_pull2(ltpsfile_remote, return_value=False)
  ltpsfile_local = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N2DH-GOWT1/01_RES/ltps.npy"
  rawdir = "/projects/project-broaddus/rawdata/isbi_challenge/Fluo-N2DH-GOWT1/01/"



"""
Generate a three-panel movie of the tracking with colored cell centerpoint trajectories.
"""
def trackingvideo(CONTINUE=False):

  outdir = Path("/projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N3DL-DRO/01/trackingvideo/")
  outdir.mkdir(exist_ok=True, parents=True)

  # ltpsfile_remote = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N2DH-GOWT1/01_RES/ltps.npy"
  # rsync_pull2(ltpsfile_remote, return_value=False)
  ltpsfile_local = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N3DL-DRO/01_RES/ltps/ltps.npy"
  rawdir = "/projects/project-broaddus/rawdata/isbi_challenge/Fluo-N3DL-DRO/01/"

  # N = len(ltps)
  # files = glob("/Users/broaddus/Desktop/project-broaddus/rawdata/isbi_challenge/Fluo-N3DL-DRO/02/t*.tif")
  # N = len(files)
  N     = 20
  DT    = 3 # >= 1
  lw    = 1 # >= 1
  scale = (5,1,1)
  dub   = 20

  ## do the tracking
  ltps = np.load(ltpsfile_local, allow_pickle=True)
  tb   = tracking.nn_tracking_on_ltps(ltps, scale=scale, dub=dub)
  nap  = tracking.tb2nap(tb,ltps)

  bigshape = None

  ## color must be relatively unique for each cell
  colormap = (np.random.rand(100,3)+0.2)/1.2

  ## Save a max projection from each angle
  for i in range(N):
    print(i)
    file = f"{rawdir}/t{i:03d}.tif"

    if not CONTINUE:
      img  = load(file)
      bigshape = img.shape
      imgZ = norm_percentile01(img.max(0), 0, 99)
      save(imgZ, outdir / f"imgZ{i:03d}.png")
      imgY = zoom(norm_percentile01(img.max(1), 0, 99), scale[:2], order=1)
      save(imgY, outdir / f"imgY{i:03d}.png")
      imgX = zoom(norm_percentile01(img.max(2), 0, 99), scale[:2], order=1)
      save(imgX, outdir / f"imgX{i:03d}.png")
    else:
      imgZ = load(outdir / f"imgZ{i:03d}.png")
      imgY = load(outdir / f"imgY{i:03d}.png")
      imgX = load(outdir / f"imgX{i:03d}.png")
      bigshape = (imgY.shape[0], imgZ.shape[0], imgZ.shape[1])
      print(bigshape)

    dy,dx = imgZ.shape
    dpi = 200

    fig,ax = plt.subplots(figsize=(dx/dpi , dy/dpi), dpi=dpi)
    ax.imshow(imgZ,cmap='gray')
    ax.axis('off')
    ax.set_position([0,0,1,1])
    tracklets = nap.tracklets[(i <= nap.tracklets[:,1]) & (nap.tracklets[:,1] <= i+DT)]
    for j in set(tracklets[:,0]):
      subtracklets = tracklets[tracklets[:,0]==j]
      # color = colormap[int(j)%100]
      color = plt.cm.magma(subtracklets[0,2] / bigshape[0])
      ax.plot(subtracklets[:,4] , subtracklets[:,3], c=color, lw=lw) ## plot position only 
    plt.savefig(outdir / f"trackZ{i:03d}.png")
    plt.close()

    dy,dx = imgY.shape
    fig,ax = plt.subplots(figsize=(dx/dpi , dy/dpi), dpi=dpi)
    ax.imshow(imgY,cmap='gray')
    ax.axis('off')
    ax.set_position([0,0,1,1])
    tracklets = nap.tracklets[(i <= nap.tracklets[:,1]) & (nap.tracklets[:,1] <= i+DT)]
    for j in set(tracklets[:,0]):
      subtracklets = tracklets[tracklets[:,0]==j]
      # color = colormap[int(j)%100]
      color = plt.cm.magma(subtracklets[0,3] / bigshape[1])
      ax.plot(subtracklets[:,4] , scale[0]*subtracklets[:,2], c=color, lw=lw) ## plot position only 
    plt.savefig(outdir / f"trackY{i:03d}.png")
    plt.close()

    dy,dx = imgX.shape
    fig,ax = plt.subplots(figsize=(dx/dpi , dy/dpi), dpi=dpi)
    ax.imshow(imgX,cmap='gray')
    ax.axis('off')
    ax.set_position([0,0,1,1])
    tracklets = nap.tracklets[(i <= nap.tracklets[:,1]) & (nap.tracklets[:,1] <= i+DT)]
    for j in set(tracklets[:,0]):
      subtracklets = tracklets[tracklets[:,0]==j]
      # color = colormap[int(j)%100]
      color = plt.cm.magma(subtracklets[0,4] / bigshape[2])
      ax.plot(subtracklets[:,3] , scale[0]*subtracklets[:,2], c=color, lw=lw) ## plot position only 
    plt.savefig(outdir / f"trackX{i:03d}.png")
    plt.close()

    imgZ = load(outdir / f"trackZ{i:03d}.png")
    imgY = load(outdir / f"trackY{i:03d}.png")
    imgX = load(outdir / f"trackX{i:03d}.png")
    dy,dx,chan1   = imgZ.shape
    dz,dx2,chan2  = imgY.shape
    dz2,dy2,chan3 = imgX.shape
    assert dy==dy2 and dx==dx2 and dz==dz2 and chan1==chan2==chan3

    panel = np.zeros((dy+dz , dx+dz , chan1),)

    panel[:dy , :dx] = imgZ
    panel[dy: , :dx] = imgY
    panel[:dy , dx:] = imgX.transpose([1,0,2])

    print(imgZ.shape)
    print(imgY.shape)
    print(imgX.shape)

    save(panel , outdir / f"frames/frame{i:03d}.png")
