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
def trackingvideo(
    filenames,
    ltps,
    outdir = "/projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N3DL-DRO/01/trackingvideo/",
    DT    = 3,   # >= 1 
    lw    = 1,   # >= 1 
    scale = (5,1,1) ,
    dub   = 20 ,
    CONTINUE=False):

  outdir = Path(outdir)
  outdir.mkdir(exist_ok=True, parents=True)

  # ltpsfile_remote = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N2DH-GOWT1/01_RES/ltps.npy"
  # rsync_pull2(ltpsfile_remote, return_value=False)
  # files = sorted(Path(rawdir).glob("t*.tif"))
  N = len(filenames)
  # N = 7

  tb   = tracking.nn_tracking_on_ltps(ltps, scale=scale, dub=dub)
  nap  = tracking.tb2nap(tb,ltps)

  bigshape = None

  ## color must be relatively unique for each cell
  colormap = (np.random.rand(100,3)+0.2)/1.2

  ## Save a max projection from each angle
  for i in range(N):
    # file = f"{rawdir}/t{i:03d}.tif"
    print(i)
    file = filenames[i]

    if not CONTINUE:
      if str(file)[-3:]=='raw':
        img = imreadraw(file)
      else:
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

import argparse


def run_standard(isbiname='Fluo-N3DL-TRIF', dataset='01',):
  scale = isbi_tools.isbi_scales[isbiname]
  scale = (np.array(scale)/scale[0])[::-1]
  scale = tuple(scale)

  rawdir = f"/projects/project-broaddus/rawdata/isbi_challenge/{isbiname}/{dataset}/"
  filenames = sorted(Path(rawdir).glob("t*.tif"))
  ltps = np.load(f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/{isbiname}/{dataset}_RES/ltps/ltps.npy",allow_pickle=1)

  trackingvideo(
    filenames,
    ltps,
    outdir = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/{isbiname}/{dataset}/trackingvideo/",
    DT    = 3,
    lw    = 1,
    scale = scale,
    dub   = 20 ,
    CONTINUE=False)  


import isbi_tools
from subprocess import Popen

def myrun_slurm():
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  # _gpu  = "-p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 "    ## TODO: more cores?
  _cpu  = "-n 1 -t 3:00:00 -c 1 --mem 128000 "
  Popen("cp *.py temp/",shell=True)
  slurm = """
cd temp
sbatch -J png_{name} {_resources} -o ../slurm/png_{name}.out -e ../slurm/png_{name}.err \
<< EOF
#!/bin/bash 
python3 -c \'import png_tracking as A; A.run_standard(\"{isbiname}\",\"{dataset}\")\'
EOF
"""
  slurm = slurm.replace("{_resources}",_cpu) ## you can't partially format(), but you can replace().

  isbiname = 'Fluo-N3DL-TRIC'
  dataset  = '01'
  Popen(slurm.format(name=isbiname[-6:],isbiname=isbiname,dataset=dataset),shell=True)
  
  # for isbiname in isbi_tools.isbi_names:
  #   if '3D' not in isbiname: continue
  #   for dataset in ["01","02"]:
  #     # print(slurm.format(name=isbiname[-6:],isbiname=isbiname,dataset=dataset),flush=True)
  #     Popen(slurm.format(name=isbiname[-6:],isbiname=isbiname,dataset=dataset),shell=True)


def imreadraw(name):
  return np.fromfile(name,dtype='uint16').reshape(134,1024,512)

def runDaniela():

  # rawdir = f"/projects/project-broaddus/rawdata/isbi_challenge/{isbiname}/{dataset}/"
  rawdir = "/projects/project-broaddus/rawdata/daniela/2019-12-11-10-39-07-98-Trier_Tribolium_nGFP_window_highres/stacks/C0opticsprefused/"
  filenames = sorted(Path(rawdir).glob("*.raw"))
  ltps  = np.load("/projects/project-broaddus/rawdata/daniela/pred/ltps/ltps.npy",allow_pickle=1)

  trackingvideo(
    filenames,
    ltps,
    outdir = "/projects/project-broaddus/rawdata/daniela/pred/trackingvideo/",
    DT    = 3,   # >= 1 
    lw    = 1,   # >= 1 
    scale = (4,1,1) ,
    dub   = 20 ,
    CONTINUE=False,
    )






# if __name__=="__main__":

#   parser = argparse.ArgumentParser(description='Create folder of PNGs with cell tracking solutions from results.')
#   parser.add_argument('-r','--rawdir',)
#   parser.add_argument('-d','--detections', )
#   parser.add_argument('-o','--outdir',)
#   parser.add_argument('--dt',type=int,default=3)
#   parser.add_argument('--scale', type=float, nargs='*', default=[1.,1.])

#   # parser.add_argument('-c','--continue', default=False)
#   # parser.set_defaults(remote=False)

#   args = parser.parse_args()
  
#   trackingvideo(
#     rawdir = args.rawdir,
#     ltpsfile_local = args.detections,
#     outdir = args.outdir,
#     DT    = args.dt,   # >= 1 
#     lw    = 1,   # >= 1 
#     scale = tuple(args.scale) ,
#     dub   = 20 ,
#     CONTINUE=False)




"""
sbatch -J Fluo-N3DL-TRIF-01 \
-p gpu --gres gpu:1 -n 1 -c 1 -t 12:00:00 --mem 128000 \
-o slurm_out/png_tracking/Fluo-N3DL-TRIF-01.txt \
-e slurm_err/png_tracking/Fluo-N3DL-TRIF-01.txt \
<< EOF
#!/bin/bash
/bin/time -v python png_tracking.py \
-r /projects/project-broaddus/rawdata/isbi_challenge/Fluo-N3DL-TRIF/01 \
-d /projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N3DL-TRIF/01_RES/ltps/ltps.npy \
-o /projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N3DL-TRIF/01/trackingvideo/ \
--dt 3 \
--scale 1 1 1
EOF


sbatch -J Fluo-N3DL-TRIF-02 \
-p gpu --gres gpu:1 -n 1 -c 1 -t 12:00:00 --mem 128000 \
-o slurm_out/png_tracking/Fluo-N3DL-TRIF-02.txt \
-e slurm_err/png_tracking/Fluo-N3DL-TRIF-02.txt \
<< EOF
#!/bin/bash
/bin/time -v python png_tracking.py \
-r /projects/project-broaddus/rawdata/isbi_challenge/Fluo-N3DL-TRIF/02 \
-d /projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N3DL-TRIF/02_RES/ltps/ltps.npy \
-o /projects/project-broaddus/rawdata/isbi_challenge_out_extra/Fluo-N3DL-TRIF/02/trackingvideo/ \
--dt 3 \
--scale 1 1 1
EOF


sbatch -J Png-DRO-02 -n 1 -c 1 -t 12:00:00 --mem 16000 \
-o slurm_out/png_tracking/Png-DRO-02.txt \
-e slurm_err/png_tracking/Png-DRO-02.txt \
<<EOF
#!/bin/bash
/bin/time -v python -c 'import png_tracking as A; A.run_standard("Fluo-N3DL-DRO","02",5);'
EOF

sbatch -J Png-DRO-02 -n 1 -c 1 -t 12:00:00 --mem 16000 \
-o slurm_out/png_tracking/Png-DRO-02.txt \
-e slurm_err/png_tracking/Png-DRO-02.txt \
<<EOF
#!/bin/bash
/bin/time -v python -c 'import png_tracking as A; A.run_standard("Fluo-N3DL-DRO","02",5);'
EOF


sbatch -J png-DRO-02 -n 1 -c 1 -t 12:00:00 --mem 16000 -o slurm/png-DRO-02.out -e slurm/png-DRO-02.err 
<<EOF
#!/bin/bash
/bin/time -v python -c 'import png_tracking as A; A.run_standard("Fluo-N3DL-DRO","02",5);'
EOF


"""