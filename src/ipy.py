import numpy as np
from ns2dir import load,save
import itertools
import tifffile
import files
from pathlib import Path

# def makepieces():
#   tot = []
#   for n,e,i in itertools.product(range(1,8),range(0,36,5),range(4)):
#     f = f"/projects/project-broaddus/devseg_2/e02/t{n}/ta/pred_e{e}_i{i}.tif"
#     img = load(f)
#     tot.append(img[0,16].copy())
#     if i==3:
#       res = np.array(tot)
#       print(res.shape)
#       save(res, str(f).replace('ta','ta/slice'))
#       tot = []

def combine():
  for n,m in itertools.product([1,2,3,4,5,6,7],[1,2]):
    home = Path(f"../e02/t{n}/pts/Fluo-N3DH-CE/0{m}/").resolve()
    res = []
    for d in home.glob('p*'):
      print(d)
      if d.is_dir():
        res.append(load(d/'p1.npy'))
    save(res, home/'traj.pkl')

def maxout():
  maxdir = Path(str(files.raw_ce_train_02[0].parent).replace('isbi/','isbi/mx/'))
  for i,fi in enumerate(files.raw_ce_train_02):
    img = load(fi)
    mx = img.max(0) #; mx *= 255/mx.max(); mx = mx.clip(0,255).astype(np.uint8)
    save(mx, maxdir / f't{i:03d}.png')
    print(i)