import itertools
from math import floor,ceil

import numpy as np
from scipy.ndimage import label,zoom
from skimage.feature  import peak_local_max
from skimage.measure  import regionprops
# import tifffile
from pathlib import Path

from subprocess import Popen,run

import torch
from torch import nn

# import files

from segtools.math_utils import conv_at_pts_multikern
from segtools.numpy_utils import normalize3
from segtools.render import get_fnz_idx2d
from segtools.ns2dir import load,save
from segtools import point_matcher
from segtools import torch_models
# from types import SimpleNamespace

## works with any torch cnn and 3D image (potentially multichannel)

def apply_net_tiled_3d(net,img):
  """
  Does not perform normalization.
  Applies net to image with dims Channels,Z,Y,X.
  Assume 3x or less max pooling layers => (U-net) translational symmetry with period 2^n for n in [0,1,2,3].
  """

  # borders           = [8,24,24] ## border width within each patch that is thrown away after prediction
  # patchshape_padded = [32,240,240] ## the size of the patch that we feed into the net. must be divisible by 8 or net fails.
  # patchshape        = [16,200,200] ## must be divisible by 8 to avoid artifacts.
  # stride            = [16,200,200] ## same as patchshape in this case
  # def g(n,m): return floor(n/m)*m-n ## f(n,m) gives un-padding needed for n to be divisible by m
  def f(n,m): return ceil(n/m)*m-n ## gives padding needed for n to be divisible by m

  a,b,c = img.shape[1:]

  ## extra border needed for stride % 8 = 0. read as e.g. "ImagePad_Z"
  ip_z,ip_y,ip_x = f(a,8),f(b,8),f(c,8) 
  
  # pp_z,pp_y,pp_x = 8,32,32
  # DZ,DY,DX = 16,200,200

  ## max total size with Unet3 16 input channels (64,528,528) = 
  ## padding per patch. must be divisible by 8. read as e.g. "PatchPad_Z"
  pp_z,pp_y,pp_x = 8,64,64
  ## inner patch size (does not include patch border. also will be smaller at boundary)
  DZ,DY,DX = 48,400,400

  img_padded = np.pad(img,[(0,0),(pp_z,pp_z+ip_z),(pp_y,pp_y+ip_y),(pp_x,pp_x+ip_x)],mode='constant')
  output = np.zeros(img.shape)

  ## start coordinates for each patch in the padded input. each stride must be divisible by 8.
  zs = np.r_[:a:DZ]
  ys = np.r_[:b:DY]
  xs = np.r_[:c:DX]

  ## start coordinates of padded input (e.g. top left patch corner)
  for x,y,z in itertools.product(xs,ys,zs):
    ## end coordinates of padded input (including patch border)
    ze,ye,xe = min(z+DZ,a+ip_z) + 2*pp_z, min(y+DY,b+ip_y) + 2*pp_y, min(x+DX,c+ip_x) + 2*pp_x
    patch = img_padded[:,z:ze,y:ye,x:xe]
    with torch.no_grad():
      patch = torch.from_numpy(patch).cuda().float()
      patch = net(patch[None])[0,:,pp_z:-pp_z,pp_y:-pp_y,pp_x:-pp_x].detach().cpu().numpy()
    ## end coordinates of unpadded output (not including border)
    ae,be,ce = min(z+DZ,a),min(y+DY,b),min(x+DX,c)
    ## use "ae-z" because patch size changes near boundary
    output[:,z:ae,y:be,x:ce] = patch[:,:ae-z,:be-y,:ce-x]

  return output

## can be used for any d_isbi dataset

def isbi_predict(predictor):
  args,kwargs = predictor.f_net_args
  net = torch_models.Unet3(*args,**kwargs).cuda()
  net.load_state_dict(torch.load(predictor.best_model))
  o = predictor.out
  print("A OK")

  for time in predictor.predict_times:
    name_img = predictor.input_dir / f't{time:03d}.tif'
    # if (o.pred / name_img.name).exists(): continue
    img = load(name_img)
    img = predictor.norm(img)
    res = apply_net_tiled_3d(net,img[None])[0]
    pts = peak_local_max(res,threshold_abs=0.1,exclude_border=False,footprint=predictor.plm_footprint)
    unambiguous_matches = point_matcher.match_unambiguous_nearestNeib(predictor.traj_gt_pred[time],pts,dub=10)
    print("new {:2d} {:6f} {:6f} {:6f}".format(time, unambiguous_matches.f1,unambiguous_matches.precision,unambiguous_matches.recall))

    res = res.astype(np.float16)
    save(res,                 o.pred / name_img.name)
    save(res.max(0),          o.mx_z / name_img.name)
    save(pts,                 o.pts / name_img.name)
    save(unambiguous_matches, o.matches / (name_img.stem + '.pkl'))

def total_matches(evaluator):
  o = evaluator.out
  match_list = [load(x) for x in o.matches.glob("t*.pkl")]
  match_scores = point_matcher.listOfMatches_to_Scores(match_list)
  save(match_scores, evaluator.name_total_scores)
  print("SCORES: ", match_scores)
  allpts = [load(x) for x in o.pts.glob('t*.tif')]
  save(allpts, evaluator.name_total_traj)

def rasterize_isbi_detections(evaluator):
  traj  = load(evaluator.name_total_traj)
  shape = load(evaluator.RAWdir / 't000.tif').shape
  for i in range(len(traj)):
    pts = evaluator.det_pts_transform(traj[i])
    lab = evaluator.pts2lab(pts,shape)
    save(lab, evaluator.RESdir / f'mask{i:03d}.tif')

def evaluate_isbi_DET(evaluator):
  run([evaluator.DET_command],shell=True)




def test_apply_net_tiled_3d():
  net = torch_models.Unet3(16,finallayer=torch_models.nn.Sequential).cuda()
  img = np.random.rand(1,100,1000,1000)
  print("input", img.shape)
  res = apply_net_tiled_3d(net,img)
  print("output", res.shape)

if __name__ == '__main__': test_apply_net_tiled_3d()


info = """

Thu Feb 20 14:32:36 2020

How should we specify directory and file names in predictor?

Right now we use predictor.RAWdir, predictor.pred_dir, predictor.pred_dir_maxz, and predictor.traj_gt_pred, and more.
There are all directories and filenames.
The predict function should not need to know the stem name or extension of files to load. That should be done by predictor.
Should it be easy to create new output directories that have different types of prediction info (eg. maxz) without having to add new attributes to predictor?
We could do this, but then predictor would not know where to _load_ from those funtions if we needed them later.
But coupling them adds a bit of overhead and some extra names.
What If, in the future, we want to be able to predict on multiple different directories?
We could have a _list_ of directory names instead of a single path. 
Then have a list of list of times for each directory?
This sounds awful...
Better is to have a single list (iterable), just of complete image filenames to read.
Then a second list with corresponding write names.

"""