from types import SimpleNamespace
# from segtools.ns2dir import load,save,toarray
# from tifffile import imread
from skimage.io import imread, imsave
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
  

  ## TODO: toggle for dense predictions

  raws  = sorted(glob(f"/projects/project-broaddus/rawdata/isbi_challenge/{isbiname}/{dataset}/t*.tif"))
  savedir_common = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/vidz/"

  ## NON-DENSE predictions (standard)
  savedir = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/{isbiname}/{dataset}/vidz/"
  masks = sorted(glob(f"/projects/project-broaddus/rawdata/isbi_challenge_out/{isbiname}/{dataset}_RES/mask*.tif"))
  savefile = f"{savedir_common}/{isbiname}-{dataset}.mp4"
  
  ## DENSE predictions
  # savedir = f"/projects/project-broaddus/rawdata/isbi_challenge_out_extra/{isbiname}/{dataset}-dense/vidz/"
  # masks = sorted(glob(f"/projects/project-broaddus/rawdata/isbi_challenge_out_dense/{isbiname}/{dataset}_RES/mask*.tif"))
  # savefile = f"{savedir_common}/{isbiname}-{dataset}-dense.mp4"

  Path(savedir).mkdir(parents=1,exist_ok=1)
  Path(savedir_common).mkdir(parents=1,exist_ok=1)
  
  assert len(raws)>0 , "Empty raw dir"
  assert len(masks)>0 , "Empty mask dir"
  assert len(raws)==len(masks) , "Raw and Mask not consistent"

  cmap = np.random.rand(256,3).clip(min=0.2)
  cmap[0] = (0,0,0)
  cmap = matplotlib.colors.ListedColormap(cmap)

  run([f'rm {savedir}/*.png'], shell=True)

  for i in range(len(raws)):
    print(f"Saving png {i}/{len(raws)} .", end='\r', flush=True)
    pngraw  = img2png(imread(raws[i]).astype(np.float32))
    pngmask = img2png(imread(masks[i]), colors=cmap)
    # imsave(os.path.join(savedir,f"raw{i:04d}.png") , pngraw)
    # imsave(os.path.join(savedir,f"mask{i:04d}.png") , pngmask)
    blend = blendRawLabPngs(pngraw,pngmask)
    # blend = imread(os.path.join(savedir,f"blend{i:04d}.png"))
    h,w,_ = blend.shape
    blend = np.pad(blend, ((0,h%2), (0,w%2), (0,0)), mode='constant')
    imsave(os.path.join(savedir,f"blend{i:04d}.png") , blend)

  ## The -vf flag is alias for -filter:v which uses the filter language.
  ## To play back at half-speed use setpts=2.0*PTS in filter (Presentation Time Stamp ?)
  ## to determine framerate, bitrate, duration of video run `ffmpeg -i video.mp4`
  ## videos with libx264 encoding have preview in Finder and open in Quicktime.
  ## -crf "constant rate factor" scales quality by a factor, allowing variable bitrate. default is 23 ?
  run([f'ffmpeg -y -i {savedir}/blend%04d.png -c:v libx264 -vf "setpts=2.0*PTS,format=yuv420p" {savefile}'], shell=True)



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
  sbatch -J {name} {_resources} -o slurm/{name}.out -e slurm/{name}.err --wrap \'/bin/time -v python3 -c \"import movies as A; A.make_movie_pid({pid})\"\' 
  """
  slurm = slurm.replace("{_resources}",_cpu) ## you can't partially format(), but you can replace().
  
  for pid in range(19*2):
    isbiname, dataset = pid2params(pid)

    name = isbiname.split("-")[-1] + dataset
    if "SIM+" in name:
      name = "".join(isbiname.split("-")[-2:]) + dataset

    if "TRIF" in name: continue
    # if name not in denselist: continue
    # if name not in redolist: continue
    job  = slurm.format(name=name, pid=pid)
    print(job)
    Popen(job,shell=True)


# redolist = [
#   # "GOWT102",
#   # "CHO02"
#   "PSC01",
#   "PSC02",
# ]

denselist = [
  "DRO01",
  "DRO02",
  "TRIC01",
  "TRIC02",
  "TRIF01",
  "TRIF02",
]


"""bash
sbatch -J HSC01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/HSC01.out -e slurm/HSC01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(0)"'
sbatch -J HSC02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/HSC02.out -e slurm/HSC02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(1)"'
sbatch -J MuSC01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/MuSC01.out -e slurm/MuSC01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(2)"'
sbatch -J MuSC02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/MuSC02.out -e slurm/MuSC02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(3)"'
sbatch -J HeLa01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/HeLa01.out -e slurm/HeLa01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(4)"'
sbatch -J HeLa02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/HeLa02.out -e slurm/HeLa02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(5)"'
sbatch -J MSC01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/MSC01.out -e slurm/MSC01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(6)"'
sbatch -J MSC02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/MSC02.out -e slurm/MSC02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(7)"'
sbatch -J A54901 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/A54901.out -e slurm/A54901.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(8)"'
sbatch -J A54902 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/A54902.out -e slurm/A54902.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(9)"'
sbatch -J SIM01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/SIM01.out -e slurm/SIM01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(10)"'
sbatch -J SIM02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/SIM02.out -e slurm/SIM02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(11)"'
sbatch -J H15701 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/H15701.out -e slurm/H15701.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(12)"'
sbatch -J H15702 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/H15702.out -e slurm/H15702.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(13)"'
sbatch -J MDA23101 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/MDA23101.out -e slurm/MDA23101.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(14)"'
sbatch -J MDA23102 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/MDA23102.out -e slurm/MDA23102.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(15)"'
sbatch -J GOWT101 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/GOWT101.out -e slurm/GOWT101.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(16)"'
sbatch -J GOWT102 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/GOWT102.out -e slurm/GOWT102.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(17)"'
sbatch -J N2DHSIM+01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/N2DHSIM+01.out -e slurm/N2DHSIM+01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(18)"'
sbatch -J N2DHSIM+02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/N2DHSIM+02.out -e slurm/N2DHSIM+02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(19)"'
sbatch -J HeLa01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/HeLa01.out -e slurm/HeLa01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(20)"'
sbatch -J HeLa02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/HeLa02.out -e slurm/HeLa02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(21)"'
sbatch -J CE01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/CE01.out -e slurm/CE01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(22)"'
sbatch -J CE02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/CE02.out -e slurm/CE02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(23)"'
sbatch -J CHO01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/CHO01.out -e slurm/CHO01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(24)"'
sbatch -J CHO02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/CHO02.out -e slurm/CHO02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(25)"'
sbatch -J N3DHSIM+01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/N3DHSIM+01.out -e slurm/N3DHSIM+01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(26)"'
sbatch -J N3DHSIM+02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/N3DHSIM+02.out -e slurm/N3DHSIM+02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(27)"'
sbatch -J DRO01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/DRO01.out -e slurm/DRO01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(28)"'
sbatch -J DRO02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/DRO02.out -e slurm/DRO02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(29)"'
sbatch -J TRIC01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIC01.out -e slurm/TRIC01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(30)"'
sbatch -J TRIC02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIC02.out -e slurm/TRIC02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(31)"'
sbatch -J U37301 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/U37301.out -e slurm/U37301.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(32)"'
sbatch -J U37302 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/U37302.out -e slurm/U37302.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(33)"'
sbatch -J PSC01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/PSC01.out -e slurm/PSC01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(34)"'
sbatch -J PSC02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/PSC02.out -e slurm/PSC02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(35)"'
sbatch -J TRIF01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIF01.out -e slurm/TRIF01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(36)"'
sbatch -J TRIF02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIF02.out -e slurm/TRIF02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(37)"'

# Dense - remember to toggle path names!

sbatch -J dn-DRO01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/DRO01.out -e slurm/DRO01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(28)"'
sbatch -J dn-DRO02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/DRO02.out -e slurm/DRO02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(29)"'
sbatch -J dn-TRIC01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIC01.out -e slurm/TRIC01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(30)"'
sbatch -J dn-TRIC02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIC02.out -e slurm/TRIC02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(31)"'
sbatch -J dn-TRIF01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIF01.out -e slurm/TRIF01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(36)"'
sbatch -J dn-TRIF02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIF02.out -e slurm/TRIF02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(37)"'

"""


## Timing

# MDA23101      Elapsed (wall clock) time (h:mm:ss or m:ss): 0:07
# MDA23102      Elapsed (wall clock) time (h:mm:ss or m:ss): 0:07
# SIM01         Elapsed (wall clock) time (h:mm:ss or m:ss): 0:09
# A54901        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:11
# A54902        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:13
# SIM02         Elapsed (wall clock) time (h:mm:ss or m:ss): 0:13
# CHO01         Elapsed (wall clock) time (h:mm:ss or m:ss): 0:39
# CHO02         Elapsed (wall clock) time (h:mm:ss or m:ss): 0:41
# HeLa02        Elapsed (wall clock) time (h:mm:ss or m:ss): 0:55
# HeLa01        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:01
# MSC01         Elapsed (wall clock) time (h:mm:ss or m:ss): 1:08
# MSC02         Elapsed (wall clock) time (h:mm:ss or m:ss): 1:09
# CE02          Elapsed (wall clock) time (h:mm:ss or m:ss): 1:42
# U37301        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:47
# U37302        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:51
# SIMerr:       Elapsed (wall clock) time (h:mm:ss or m:ss): 1:58
# PSC01         Elapsed (wall clock) time (h:mm:ss or m:ss): 2:11
# PSC02         Elapsed (wall clock) time (h:mm:ss or m:ss): 2:14
# CE01          Elapsed (wall clock) time (h:mm:ss or m:ss): 2:23
# DRO02         Elapsed (wall clock) time (h:mm:ss or m:ss): 2:45
# DRO01         Elapsed (wall clock) time (h:mm:ss or m:ss): 2:46
# SIMerr:       Elapsed (wall clock) time (h:mm:ss or m:ss): 2:47
# H15701        Elapsed (wall clock) time (h:mm:ss or m:ss): 3:01
# HeLa02        Elapsed (wall clock) time (h:mm:ss or m:ss): 3:06
# HeLa01        Elapsed (wall clock) time (h:mm:ss or m:ss): 3:23
# GOWT102       Elapsed (wall clock) time (h:mm:ss or m:ss): 3:51
# SIMerr:       Elapsed (wall clock) time (h:mm:ss or m:ss): 3:53
# GOWT101       Elapsed (wall clock) time (h:mm:ss or m:ss): 3:54
# H15702        Elapsed (wall clock) time (h:mm:ss or m:ss): 5:26
# SIMerr:       Elapsed (wall clock) time (h:mm:ss or m:ss): 6:06
# TRIC01        Elapsed (wall clock) time (h:mm:ss or m:ss): 21:25
# TRIC02        Elapsed (wall clock) time (h:mm:ss or m:ss): 35:16
# MuSC02        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:15:09
# MuSC01        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:16:23
# HSC02         Elapsed (wall clock) time (h:mm:ss or m:ss): 1:19:09
# HSC01         Elapsed (wall clock) time (h:mm:ss or m:ss): 1:31:05



if __name__=="__main__":
  make_all_movies()
