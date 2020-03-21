import numpy as np
from segtools.numpy_utils import collapse2, normalize3, flatten
from skimage import io
import tifffile
from utils.datagen_common import place_gauss_at_pts, resize_to_fit
from skimage.measure import regionprops
from math import floor,ceil
from utils.readwrite import pklsave

def cat(*args,axis=0): return np.concatenate(args, axis)
def stak(*args,axis=0): return np.stack(args, axis)

def imsave(x, name, **kwargs): return tifffile.imsave(str(name), x, **kwargs)
def imread(name,**kwargs): return tifffile.imread(str(name), **kwargs)


def random_slice(img_size, patch_size):
  assert len(img_size) == len(patch_size)
  def f(d,s):
    if s == -1: return slice(None)
    start = np.random.randint(0,d-s+1)
    end   = start + s
    return slice(start,end)
  return tuple([f(d,s) for d,s in zip(img_size, patch_size)])


def datagen_all_kinds(params={}, savedir=None):
  data = []

  times = np.r_[:190]

  for i in times:
    img = imread(f'/lustre/projects/project-broaddus/devseg_data/raw/celegans_isbi/Fluo-N3DH-CE/01/t{i:03d}.tif')
    lab = imread(f'/lustre/projects/project-broaddus/devseg_data/raw/celegans_isbi/Fluo-N3DH-CE/01_GT/TRA/man_track{i:03d}.tif')

    pmin, pmax = np.random.uniform(1,3), np.random.uniform(99.5,99.8)
    img = normalize3(img,pmin,pmax).astype(np.float32,copy=False)

    pts  = np.array([r.centroid for r in regionprops(lab)]).astype(np.int)

    s = 2
    target,kern1 = place_gauss_at_pts(pts,w=[s/5,s,s])
    target = target[ceil(s/5)*3:,ceil(s)*3:,ceil(s)*3:] ## remove half width of kernel (rounded down) = 3*w
    target1 = resize_to_fit(target, lab.shape)

    s = 4
    target,kern2 = place_gauss_at_pts(pts,w=[s/5,s,s])
    target = target[ceil(s/5)*3:,ceil(s)*3:,ceil(s)*3:] ## remove half width of kernel (rounded down) = 3*w
    target2 = resize_to_fit(target, lab.shape)
    
    s = 6
    target,kern3 = place_gauss_at_pts(pts,w=[s/5,s,s])
    target = target[ceil(s/5)*3:,ceil(s)*3:,ceil(s)*3:] ## remove half width of kernel (rounded down) = 3*w
    target3 = resize_to_fit(target, lab.shape)

    s = pts.shape[0]**(-1/3)*15 ## chosen to give wx,wy=2 at 350 cells
    target,kern4 = place_gauss_at_pts(pts,w=[s/5,s,s])
    target = target[ceil(s/5)*3:,ceil(s)*3:,ceil(s)*3:] ## remove half width of kernel (rounded down) = 3*w
    target4 = resize_to_fit(target, lab.shape)

    slicelist = []
    def random_patch():
      ss = random_slice(img.shape, (32,64,64))
      while (target[ss]>0).sum() < kern1.size/2:
        ss = random_slice(img.shape, (32,64,64))
      x  = img[ss].copy()
      y1 = target1[ss].copy()
      y2 = target2[ss].copy()
      y3 = target3[ss].copy()
      y4 = target4[ss].copy()
      slicelist.append(ss)
  
      ## augment
      noiselevel = 0.2
      x += np.random.uniform(0,noiselevel,(1,)*3)*np.random.uniform(-1,1,x.shape)
      for d in [0,1,2]:
        if np.random.rand() < 0.5:
          x  = np.flip(x,d)
          y1 = np.flip(y1,d)
          y2 = np.flip(y2,d)
          y3 = np.flip(y3,d)
          y4 = np.flip(y4,d)

      return x,y1,y2,y3,y4

    data.append([random_patch() for _ in range(10)]) #ts(xys)czyx

  data = np.array(data)

  print("data.shape: ", data.shape)

  if savedir:
    rgb = collapse2(data[:,:,:,16],'tscyx','ty,sx,c')[...,[0,2,4]]
    rgb = normalize3(rgb)
    io.imsave(savedir/'data_xy.png',rgb)
    rgb = collapse2(data[:,:,:,:,32],'tsczx','tz,sx,c')[...,[0,2,4]]
    rgb = normalize3(rgb)
    io.imsave(savedir/'data_xz.png',rgb)
    np.savez_compressed(savedir/'data.npz',data)
    pklsave(slicelist, savedir/'slicelist.pkl')

  return data

def datagen_self_sup(params={}, savedir=None):
  data = []

  times = np.r_[:190:5]

  for i in times:
    img = imread(f'/lustre/projects/project-broaddus/devseg_data/raw/celegans_isbi/Fluo-N3DH-CE/01/t{i:03d}.tif')


    pmin, pmax = np.random.uniform(1,3), np.random.uniform(99.5,99.8)
    img = normalize3(img,pmin,pmax).astype(np.float32,copy=False)

    slicelist = []
    def random_patch():
      ss = random_slice(img.shape, (32,64,64))
      ## select patches with interesting content. 0.02 is chosen by manual inspection.
      while img[ss].mean() < 0.02:
        ss = random_slice(img.shape, (32,64,64))
      x  = img[ss].copy()
      slicelist.append(ss)
  
      ## augment
      # noiselevel = 0.2
      # x += np.random.uniform(0,noiselevel,(1,)*3)*np.random.uniform(-1,1,x.shape)
      # for d in [0,1,2]:
      #   if np.random.rand() < 0.5:
      #     x  = np.flip(x,d)

      return (x,)

    data.append([random_patch() for _ in range(8)]) #ts(xys)czyx

  data = np.array(data)

  print("data.shape: ", data.shape)

  if savedir:
    rgb = collapse2(data[:,:,:,16],'tscyx','ty,sx,c')[...,[0,0,0]]
    rgb = normalize3(rgb)
    io.imsave(savedir/'data_xy.png',rgb)
    rgb = collapse2(data[:,:,:,:,32],'tsczx','tz,sx,c')[...,[0,0,0]]
    rgb = normalize3(rgb)
    io.imsave(savedir/'data_xz.png',rgb)
    np.savez_compressed(savedir/'data.npz',data)
    pklsave(slicelist, savedir/'slicelist.pkl')

  return data

def datagen_self_sup_artifacts_wingStack1(savedir=None):

  img = imread(f'/lustre/projects/project-broaddus/devseg_data/raw/artifacts/wingStack1.tif')

  pmin, pmax = np.random.uniform(1,3), np.random.uniform(99.5,99.8)
  img = normalize3(img,pmin,pmax).astype(np.float32,copy=False)

  slicelist = []
  def random_patch():
    ss = random_slice(img.shape, (32,64,64))

    ## select patches with interesting content. fix this for wing!
    while img[ss].mean() < 0.02:
      ss = random_slice(img.shape, (32,64,64))
    x  = img[ss].copy()
    slicelist.append(ss)

    ## augment
    # noiselevel = 0.2
    # x += np.random.uniform(0,noiselevel,(1,)*3)*np.random.uniform(-1,1,x.shape)
    # for d in [0,1,2]:
    #   if np.random.rand() < 0.5:
    #     x  = np.flip(x,d)

    return (x,)

  data = np.array([random_patch() for _ in range(100)])

  print("data.shape: ", data.shape)

  #SCZYX
  if savedir:
    rgb = collapse2(data[:,:,::8],'sczyx','sy,zx,c')[...,[0,0,0]]
    rgb = normalize3(rgb)
    io.imsave(savedir/'data_xy.png',rgb)
    rgb = collapse2(data[:,:,:,::8],'sczyx','sy,zx,c')[...,[0,0,0]]
    rgb = normalize3(rgb)
    io.imsave(savedir/'data_xz.png',rgb)
    np.savez_compressed(savedir/'data.npz',data)
    pklsave(slicelist, savedir/'slicelist.pkl')

  return data