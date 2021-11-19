from types import SimpleNamespace
# from segtools.ns2dir import load,save,toarray
from tifffile import imread
from skimage.io import imsave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from glob import glob



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

"""
c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
"""
def norm_minmax01(x):
  mx = x.max()
  mn = x.min()
  if mx==mn: 
    return x-mx
  else: 
    return (x-mn)/(mx-mn)

def norm_affine01(x,lo,hi):
  if hi==lo: 
    return x-hi
  else: 
    return (x-lo)/(hi-lo)

def norm_percentile01(x,p0,p1):
  lo,hi = np.percentile(x,[p0,p1])
  return norm_affine01(x,lo,hi)

def img2png(x,colors=None):

  if 'float' in str(x.dtype) and colors:
    # assert type(colors) is matplotlib.colors.ListedColormap
    cmap = colors
  elif 'float' in str(x.dtype) and not colors:
    # assert type(colors) is matplotlib.colors.ListedColormap
    cmap = plt.cm.gray
  elif 'int' in str(x.dtype) and type(colors) is list:
    cmap = np.array([(0,0,0)] + colors*256)[:256]
    cmap = matplotlib.colors.ListedColormap(cmap)
  elif 'int' in str(x.dtype) and type(colors) is matplotlib.colors.ListedColormap:
    cmap = colors
  elif 'int' in str(x.dtype) and colors is None:
    cmap = np.random.rand(256,3).clip(min=0.2)
    cmap[0] = (0,0,0)
    cmap = matplotlib.colors.ListedColormap(cmap)

  def _colorseg(seg):
    m = seg!=0
    seg[m] %= 255 ## we need to save a color for black==0
    seg[seg==0] = 255
    seg[~m] = 0
    rgb = cmap(seg)
    return rgb

  _dtype = x.dtype
  D = x.ndim

  if D==3:
    a,b,c = x.shape
    yx = x.max(0)
    zx = x.max(1)
    zy = x.max(2)
    x0 = np.zeros((a,a), dtype=x.dtype)
    x  = np.zeros((b+a+1,c+a+1), dtype=x.dtype)
    x[:b,:c] = yx
    x[b+1:,:c] = zx
    x[:b,c+1:] = zy.T
    x[b+1:,c+1:] = x0

  assert x.dtype == _dtype

  # ipdb.set_trace()

  if 'int' in str(x.dtype):
    x = _colorseg(x)
  else:
    x = norm_minmax01(x)
    x = cmap(x)
  
  x = (x*255).astype(np.uint8)

  if D==3:
    x[b,:] = 255 # white line
    x[:,c] = 255 # white line

  return x

def blendRawLabPngs(pngraw,pnglab):
  m = pnglab[:,:,:3]==0
  m = np.all(m,-1)
  pngraw = pngraw.copy()
  pngraw[~m] = pnglab[~m]
  # pngraw[~m] = (0.5*pngraw+0.5*pnglab)[~m]
  return pngraw


import os
from subprocess import run, Popen
from pathlib import Path

def makemovie(
  isbiname="DIC-C2DH-HeLa",
  dataset = "01",
  ):
  savedir = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/{isbiname}/{dataset}/vidz/"
  savedir_common = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/vidz/"
  Path(savedir).mkdir(parents=1,exist_ok=1)
  Path(savedir_common).mkdir(parents=1,exist_ok=1)
  raws  = sorted(glob(f"/projects/project-broaddus/rawdata/isbi_challenge/{isbiname}/{dataset}/t*.tif"))
  masks = sorted(glob(f"/projects/project-broaddus/rawdata/isbi_challenge_out/{isbiname}/{dataset}_RES/mask*.tif"))
  
  assert len(raws)>0 , "Empty raw dir"
  assert len(masks)>0 , "Empty mask dir"
  assert len(raws)==len(masks) , "Raw and Mask not consistent"

  cmap = np.random.rand(256,3).clip(min=0.2)
  cmap[0] = (0,0,0)
  cmap = matplotlib.colors.ListedColormap(cmap)

  for i in range(len(raws)):
    print(f"Saving png {i}/{len(raws)} .", end='\r', flush=True)
    pngraw  = img2png(imread(raws[i]).astype(np.float32))
    pngmask = img2png(imread(masks[i]), colors=cmap)
    # imsave(os.path.join(savedir,f"raw{i:04d}.png") , pngraw)
    # imsave(os.path.join(savedir,f"mask{i:04d}.png") , pngmask)
    blend = blendRawLabPngs(pngraw,pngmask)
    imsave(os.path.join(savedir,f"blend{i:04d}.png") , blend)

  run([f'ffmpeg -y -i {savedir}/blend%04d.png -c:v libx264 -vf "fps=55,format=yuv420p" {savedir_common}/{isbiname}-{dataset}.mp4'], shell=True)
  run([f'rm {savedir}/*.png'], shell=True)



def pid2params(pid):
  [p0,p1], pid = parse_pid(pid,[19,2])
  isbiname = isbi_names[p0]
  dataset  = ["01","02"][p1]
  return isbiname,dataset

def make_movie_pid(pid):
  isbiname,dataset = pid2params(pid)
  makemovie(isbiname,dataset)

def make_all_movies():
  _cpu  = "-n 1 -t 4:00:00 -c 1 --mem 128000 "
  slurm = """
  sbatch -J {name} {_resources} -o {name}.out -e {name}.err --wrap \'/bin/time -v python3 -c \"import mask_video as A; A.make_movie_pid({pid})\"\' 
  """
  slurm = slurm.replace("{_resources}",_cpu) ## you can't partially format(), but you can replace().
  
  for pid in range(19*2):
    isbiname, dataset = pid2params(pid)
    if not "TRIF" in isbiname:
      name = isbiname.split("-")[-1] + dataset
      job  = slurm.format(name=name, pid=pid)
      print(job)
      Popen(job,shell=True)

if __name__=="__main__":
  make_all_movies()


## Timing

# MDA23101      Elapsed (wall clock) time (h:mm:ss or m:ss): 0:07.97
# MDA23102      Elapsed (wall clock) time (h:mm:ss or m:ss): 0:07.07
# SIM01         Elapsed (wall clock) time (h:mm:ss or m:ss): 0:09.12
# A54901        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:11.76
# A54902        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:13.16
# SIM02         Elapsed (wall clock) time (h:mm:ss or m:ss): 0:13.28
# CHO01         Elapsed (wall clock) time (h:mm:ss or m:ss): 0:39.86
# CHO02         Elapsed (wall clock) time (h:mm:ss or m:ss): 0:41.83
# HeLa02        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:55.95
# HeLa01        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:01.56
# MSC01         Elapsed (wall clock) time (h:mm:ss or m:ss): 1:08.68
# MSC02         Elapsed (wall clock) time (h:mm:ss or m:ss): 1:09.41
# CE02          Elapsed (wall clock) time (h:mm:ss or m:ss): 1:42.55
# U37301        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:47.54
# U37302        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:51.75
# SIMerr:       Elapsed (wall clock) time (h:mm:ss or m:ss): 1:58.18
# PSC01         Elapsed (wall clock) time (h:mm:ss or m:ss): 2:11.39
# PSC02         Elapsed (wall clock) time (h:mm:ss or m:ss): 2:14.20
# CE01          Elapsed (wall clock) time (h:mm:ss or m:ss): 2:23.32
# DRO02         Elapsed (wall clock) time (h:mm:ss or m:ss): 2:45.15
# DRO01         Elapsed (wall clock) time (h:mm:ss or m:ss): 2:46.02
# SIMerr:       Elapsed (wall clock) time (h:mm:ss or m:ss): 2:47.00
# H15701        Elapsed (wall clock) time (h:mm:ss or m:ss): 3:01.67
# HeLa02        Elapsed (wall clock) time (h:mm:ss or m:ss): 3:06.58
# HeLa01        Elapsed (wall clock) time (h:mm:ss or m:ss): 3:23.93
# GOWT102       Elapsed (wall clock) time (h:mm:ss or m:ss): 3:51.02
# SIMerr:       Elapsed (wall clock) time (h:mm:ss or m:ss): 3:53.93
# GOWT101       Elapsed (wall clock) time (h:mm:ss or m:ss): 3:54.41
# H15702        Elapsed (wall clock) time (h:mm:ss or m:ss): 5:26.89
# SIMerr:       Elapsed (wall clock) time (h:mm:ss or m:ss): 6:06.28
# TRIC01        Elapsed (wall clock) time (h:mm:ss or m:ss): 21:25.32
# TRIC02        Elapsed (wall clock) time (h:mm:ss or m:ss): 35:16.18
# MuSC02        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:15:09
# MuSC01        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:16:23
# HSC02         Elapsed (wall clock) time (h:mm:ss or m:ss): 1:19:09
# HSC01         Elapsed (wall clock) time (h:mm:ss or m:ss): 1:31:05
