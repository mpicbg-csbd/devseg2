import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
try:
    import gputools
except:
    print("Can't import gputools on non-gpu node...\n")

# from joblib import Memory
# location = '/projects/project-broaddus/devseg_2/expr/e26_isbidet/cachedir'
# memory = Memory(location, verbose=0)




## stable. utility funcs

def sample2RawPng(sample):
  pngraw = img2png(sample.raw)
  pnglab = img2png(sample.lab)
  m = pnglab==0
  pngraw[~m] = (0.25*pngraw+0.75*pnglab)[~m]
  return pngraw


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

def take_n_evenly_spaced_items(item_list,N):
  M = len(item_list)
  # assert N<=M
  y = np.linspace(0,M-1,N).astype(np.int)
  # ss = [slice(y[i],y[i+1]) for i in range(M)]
  return np.array(item_list,dtype=np.object)[y]
  # return ss

def divide_evenly_with_min1(n_samples,n_bins):
  N = n_samples
  M = n_bins
  assert N>=M
  y = np.linspace(0,N,M+1).astype(np.int)
  ss = [slice(y[i],y[i+1]) for i in range(M)]
  return ss

def bytes2string(nbytes):
  if nbytes < 10**3:  return f"{nbytes} B"
  if nbytes < 10**6:  return f"{nbytes/10**3} KB"
  if nbytes < 10**9:  return f"{nbytes/10**6} MB"
  if nbytes < 10**12: return f"{nbytes/10**9} GB"

def file_size(root):
  "works for files or directories (recursive)"
  root = Path(root)
  # ipdb.set_trace()
  if root.is_dir():
    # https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python/1392549
    return sum(f.stat().st_size for f in root.glob('**/*') if f.is_file())
  else:
    # https://stackoverflow.com/questions/2104080/how-can-i-check-file-size-in-python
    return os.stat(fname).st_size

def strDiskSizePatchFrame(df):
  _totsize = (2+1+2) * df['shape'].apply(np.prod).sum() ## u16 (raw) + u8 (lab) + u16 (target)
  return bytes2string(_totsize)

def myzoom(img,scale):
  img=img[...]
  _dt = img.dtype
  if img.ndim==2 and 'int' in str(_dt):
    img = zoom(img,scale,order=0).astype(_dt)
  if img.ndim==2 and 'float' in str(_dt):
    img = zoom(img,scale,order=1).astype(_dt)
  if img.ndim==3 and 'int' in str(_dt):
    img = gputools.scale(img,scale,interpolation='nearest').astype(_dt)
  if img.ndim==3 and 'float' in str(_dt):
    img = gputools.scale(img.astype(np.float32),scale,interpolation='linear').astype(_dt)
  return img

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


def blendRawLab(raw,lab,colors=None):
  pngraw = img2png(raw)
  pnglab = img2png(lab,colors=colors)
  m = pnglab[:,:,:3]==0
  m = np.all(m,-1)
  pngraw[~m] = pnglab[~m]
  # pngraw[~m] = (0.5*pngraw+0.5*pnglab)[~m]
  return pngraw

def blendRawLabPngs(pngraw,pnglab):
  m = pnglab[:,:,:3]==0
  m = np.all(m,-1)
  pngraw = pngraw.copy()
  pngraw[~m] = pnglab[~m]
  # pngraw[~m] = (0.5*pngraw+0.5*pnglab)[~m]
  return pngraw


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



def uniqueNdim(a):
  assert a.ndim==2
  uniq   = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
  counts = np.unique(a.view(np.dtype((np.void, a.dtype.itemsize*a.shape[1]))),return_counts=True)[1] #.view(a.dtype).reshape(-1, a.shape[1])
  return uniq, counts



def _png_OLD_DEPRECATED(x,labcolors=None):

  def colorseg(seg):
    if labcolors:
      cmap = np.array([(0,0,0)] + labcolors*256)[:256]
    else:
      cmap = np.random.rand(256,3).clip(min=0.2)
      cmap[0] = (0,0,0)

    cmap = matplotlib.colors.ListedColormap(cmap)

    m = seg!=0
    seg[m] %= 255 ## we need to save a color for black==0
    # seg[m] += 1
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

  if 'int' in str(x.dtype):
    x = colorseg(x)
  else:
    x = norm_minmax01(x)
    x = plt.cm.gray(x)
  
  x = (x*255).astype(np.uint8)

  if D==3:
    x[b,:] = 255 # white line
    x[:,c] = 255 # white line

  return x


