import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from types import SimpleNamespace

import numpy as np
import itertools
from math import ceil

# from torchsummary import summary

# import gc,sys,psutil,os,py


@DeprecationWarning
def conv_many(*cs):
  N = len(cs)-1
  convs = [nn.Conv3d(cs[i],cs[i+1],(3,5,5),padding=(1,2,2)) for i in range(N)]
  relus = [nn.ReLU() for i in range(N)]
  res = [0]*N*2
  res[::2] = convs
  res[1::2] = relus
  return nn.Sequential(*res)


def receptivefield(net,kern=(3,5,5)):
  """
  calculate receptive field / receptive kernel of Conv net
  WARNING: overwrites net weights. save weights first!
  WARNING: only works with relu activations, pooling and upsampling.
  """
  def rfweights(m):
    if type(m) in [nn.Conv3d,nn.Conv2d]:
      m.weight.data.fill_(1/np.prod(kern)) ## conv kernel 3*5*5
      m.bias.data.fill_(0.0)
  net.apply(rfweights);
  if len(kern)==3:
    x0 = np.zeros((128,128,128)); x0[64,64,64]=1;
  elif len(kern)==2:
    x0 = np.zeros((256,256)); x0[128,128]=1;
  xout = net.cuda()(torch.from_numpy(x0)[None,None].float().cuda()).detach().cpu().numpy()
  return xout

def init_weights(net):
  def f(m):
    if type(m) == nn.Conv3d:
      torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
      m.bias.data.fill_(0.05)
  net.apply(f);

def test_weights(net):
  table = ["weight mean,weight std,bias mean,bias std".split(',')]
  def f(m):
    if type(m) == nn.Conv3d:
      table.append([float(m.weight.data.mean()),float(m.weight.data.std()),float(m.bias.data.mean()),float(m.bias.data.std())])
      print(m)
  net.apply(f)
  return table


def n_conv(chans, kernsize, padding):
  conv = {2:nn.Conv2d,3:nn.Conv3d}[len(kernsize)]
  res = []
  for i in range(len(chans)-1):
    res.append(conv(chans[i],chans[i+1],kernsize, padding=padding))
    res.append(nn.ReLU())
  res = nn.Sequential(*res)
  return res

class Unet3(nn.Module):
  """
  Unet with 3 pooling steps.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU, pool=(1,2,2), kernsize=(3,5,5)):
    super(Unet3, self).__init__()

    self.pool = pool
    self.kernsize = kernsize
    self.finallayer = finallayer
    pad = tuple(x//2 for x in kernsize)

    self.l_ab = n_conv([io[0][0], c, c,], kernsize, padding=pad)
    self.l_cd = n_conv([1*c, 2*c,  2*c,], kernsize, padding=pad)
    self.l_ef = n_conv([2*c, 4*c,  4*c,], kernsize, padding=pad)
    self.l_gh = n_conv([4*c, 8*c,  4*c,], kernsize, padding=pad)
    self.l_ij = n_conv([8*c, 4*c,  2*c,], kernsize, padding=pad)
    self.l_kl = n_conv([4*c, 2*c,  1*c,], kernsize, padding=pad)
    self.l_mn = n_conv([2*c, 1*c,  1*c,], kernsize, padding=pad)

    conv = {2:nn.Conv2d,3:nn.Conv3d}[len(kernsize)]
    self.l_o  = nn.Sequential(conv(1*c,io[1][0],(1,)*len(kernsize),padding=0), finallayer())

  def forward(self, x):

    maxpool = {2:nn.MaxPool2d,3:nn.MaxPool3d}[len(self.kernsize)]

    c1 = self.l_ab(x)
    c2 = maxpool(self.pool)(c1)
    c2 = self.l_cd(c2)
    c3 = maxpool(self.pool)(c2)
    c3 = self.l_ef(c3)
    c4 = maxpool(self.pool)(c3)
    c4 = self.l_gh(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c3],1)
    c4 = self.l_ij(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c2],1)
    c4 = self.l_kl(c4)
    c4 = F.interpolate(c4,scale_factor=self.pool)
    c4 = torch.cat([c4,c1],1)
    c4 = self.l_mn(c4)
    out1 = self.l_o(c4)

    return out1

class Unet2(nn.Module):
  """
  The same exact shape we used for models submitted to ISBI. 2,767,777 params.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU, pool=(1,2,2), kernsize=(3,5,5)):
    super(Unet2, self).__init__()

    self.pool = pool
    self.kernsize = kernsize
    self.finallayer = finallayer
    pad = tuple(x//2 for x in kernsize)

    self.l_ab = n_conv([io[0][0], c, c,], kernsize, padding=pad)
    self.l_cd = n_conv([1*c, 2*c,  2*c,], kernsize, padding=pad)
    self.l_ef = n_conv([2*c, 4*c,  2*c,], kernsize, padding=pad)
    self.l_gh = n_conv([4*c, 2*c,  1*c,], kernsize, padding=pad)
    self.l_ij = n_conv([2*c, 1*c,  1*c,], kernsize, padding=pad)

    conv = {2:nn.Conv2d,3:nn.Conv3d}[len(kernsize)]
    self.l_o  = nn.Sequential(conv(1*c,io[1][0],(1,)*len(kernsize),padding=0), finallayer())

  def forward(self, x):

    maxpool = {2:nn.MaxPool2d,3:nn.MaxPool3d}[len(self.kernsize)]

    c1 = self.l_ab(x)
    c2 = maxpool(self.pool)(c1)
    c2 = self.l_cd(c2)
    c3 = maxpool(self.pool)(c2)
    c3 = self.l_ef(c3)
    c3 = F.interpolate(c3,scale_factor=self.pool)
    c3 = torch.cat([c3,c2],1) # concat on channels
    c3 = self.l_gh(c3)
    c3 = F.interpolate(c3,scale_factor=self.pool)
    c3 = torch.cat([c3,c1],1)
    c3 = self.l_ij(c3)
    out1 = self.l_o(c3)

    return out1

class Unet1(nn.Module):
  """
  Small Unet for tiny data.
  """

  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU, pool=(1,2,2), kernsize=(3,5,5)):
    super(Unet1, self).__init__()

    self.pool = pool
    self.kernsize = kernsize
    self.finallayer = finallayer
    pad = tuple(x//2 for x in kernsize)

    self.l_ab = n_conv([io[0][0], c, c,], kernsize, padding=pad)
    self.l_cd = n_conv([1*c, 2*c,  1*c,], kernsize, padding=pad)
    self.l_ef = n_conv([2*c, 1*c,  1*c,], kernsize, padding=pad)

    conv = {2:nn.Conv2d,3:nn.Conv3d}[len(kernsize)]
    self.l_o  = nn.Sequential(conv(1*c,io[1][0],(1,)*len(kernsize),padding=0), finallayer())

  def forward(self, x):

    maxpool = {2:nn.MaxPool2d,3:nn.MaxPool3d}[len(self.kernsize)]

    c1 = self.l_ab(x)
    c2 = maxpool(self.pool)(c1)
    c2 = self.l_cd(c2)
    c3 = F.interpolate(c2,scale_factor=self.pool)
    c3 = torch.cat([c3,c1],1)
    c3 = self.l_ef(c3)
    out1 = self.l_o(c3)

    return out1

## resnets are incomplete

def conv_res(c0,c1,c2):
  return nn.Sequential(
    nn.Conv3d(c0,c1,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    nn.Conv3d(c1,c2,(3,5,5),padding=(1,2,2)),
    # nn.BatchNorm3d(c2),
    # nn.ReLU(),
    # nn.Dropout3d(p=0.1),
    )

class Res1(nn.Module):
  def __init__(self, c=32, io=[[1],[1]], finallayer=nn.LeakyReLU):
    super(Unet2, self).__init__()

    self.l_ab = conv2(io[0][0] ,1*c, 1*c)
    self.l_cd = conv2(1*c, 2*c, 2*c)
    self.l_ef = conv2(2*c, 1*c, 1*c)
    # self.l_gh = conv2(4*c, 2*c, 1*c)
    # self.l_ij = conv2(2*c, 1*c, 1*c)
    
    self.l_k  = nn.Sequential(nn.Conv3d(1*c,io[1][0],(1,1,1),padding=0), finallayer())

  def forward(self, x):

    c1 = nn.Relu()(self._lab(x)  + x)
    c2 = nn.Relu()(self._lcd(c1) + c1)
    c3 = nn.Relu()(self._lef(c2) + c2)
    out1 = self.l_k(c3)

    return out1


## utils

def pretty_size(size):
  """Pretty prints a torch.Size object"""
  assert(isinstance(size, torch.Size))
  return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
  """Prints a list of the Tensors being tracked by the garbage collector."""
  total_size = 0
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj):
        if not gpu_only or obj.is_cuda:
          print("%s:%s%s %s" % (type(obj).__name__, 
                      " GPU" if obj.is_cuda else "",
                      " pinned" if obj.is_pinned else "",
                      pretty_size(obj.size())))
          total_size += obj.numel()
      elif hasattr(obj, "data") and torch.is_tensor(obj.data):
        if not gpu_only or obj.is_cuda:
          print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
                           type(obj.data).__name__, 
                           " GPU" if obj.is_cuda else "",
                           " pinned" if obj.data.is_pinned else "",
                           " grad" if obj.requires_grad else "", 
                           " volatile" if obj.volatile else "",
                           pretty_size(obj.data.size())))
          total_size += obj.data.numel()
    except Exception as e:
      pass        
  print("Total size:", total_size)

def memReport():
  totalsize = 0
  for obj in gc.get_objects():
    if torch.is_tensor(obj):
      print(type(obj), obj.size(), obj.dtype)
      totalsize += obj.size().numel()

  print("Total Size: ", totalsize)
    
# def cpuStats():
#   print(sys.version)
#   print(psutil.cpu_percent())
#   print(psutil.virtual_memory())  # physical memory usage
#   pid = os.getpid()
#   py = psutil.Process(pid)
#   memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
#   print('memory GB:', memoryUse)

## prediction with tiling


def apply_net_2d(net,img,outchan=1,patch_boundary=(128,128),patch_inner=(256,256)):
  """
  Turns off gradients.
  Does not perform normalization.
  Applies net to image with dims Channels,Z,Y,X.
  Assume 3x or less max pooling layers => (U-net) discrete translational symmetry with period 2^n for n in [0,1,2,3].
  """

  # borders           = [8,24,24] ## border width within each patch that is thrown away after prediction
  # patchshape_padded = [32,240,240] ## the size of the patch that we feed into the net. must be divisible by 8 or net fails.
  # patchshape        = [16,200,200] ## must be divisible by 8 to avoid artifacts.
  # stride            = [16,200,200] ## same as patchshape in this case
  # def g(n,m): return floor(n/m)*m-n ## f(n,m) gives un-padding needed for n to be divisible by m

  # a,b = img.shape[-2:]
  # if a<800 and b<800:
  #   with torch.no_grad():
  #     x = torch.from_numpy(img).cuda().float()
  #     return net(x[None])[0].detach().cpu().numpy()


  def f(n,m): return ceil(n/m)*m-n ## gives padding needed for n to be divisible by m

  assert img.ndim==3
  b,c = img.shape[1:]

  ## extra border needed for stride % 8 = 0. read as e.g. "ImagePad_Z"
  ip_y,ip_x = f(b,8),f(c,8)
  
  # pp_z,pp_y,pp_x = 8,32,32
  # DZ,DY,DX = 16,200,200

  ## max total size with Unet3 16 input channels (64,528,528) = 
  ## padding per patch. must be divisible by 8. read as e.g. "PatchPad_Z"
  # pp_z,pp_y,pp_x = 8,64,64
  pp_y,pp_x = patch_boundary
  # assert all([x%8==0 for x in patch_boundary])
  ## inner patch size (does not include patch border. also will be smaller at boundary)
  DY,DX = patch_inner
  # assert all([x%8==0 for x in patch_inner])

  img_padded = np.pad(img,[(0,0),(pp_y,pp_y+ip_y),(pp_x,pp_x+ip_x)],mode='constant')
  output = np.zeros((outchan,b,c))

  ## start coordinates for each patch in the padded input. each stride must be divisible by 8.
  # zs = np.r_[:a:DZ]
  ys = np.r_[:b:DY]
  xs = np.r_[:c:DX]

  ## start coordinates of padded input (e.g. top left patch corner)
  for x,y in itertools.product(xs,ys):
    ## end coordinates of padded input (including patch border)
    ye,xe = min(y+DY,b+ip_y) + 2*pp_y, min(x+DX,c+ip_x) + 2*pp_x
    patch = img_padded[:,y:ye,x:xe]
    with torch.no_grad():
      patch = torch.from_numpy(patch).float().cuda()
      patch = net(patch[None])[0,:,pp_y:-pp_y,pp_x:-pp_x].detach().cpu().numpy()
    ## end coordinates of unpadded output (not including border)
    be,ce = min(y+DY,b),min(x+DX,c)
    ## use "ae-z" because patch size changes near boundary
    output[:,y:be,x:ce] = patch[:,:be-y,:ce-x]

  return output

# def apply_net_2d_noborder(net,img,outchan=1,patch_boundary=(128,128),patch_inner=(256,256)):
#   """
#   Turns off gradients.
#   Does not perform normalization.
#   Applies net to image with dims Channels,Z,Y,X.
#   Assume 3x or less max pooling layers => (U-net) discrete translational symmetry with period 2^n for n in [0,1,2,3].
#   """

#   # borders           = [8,24,24] ## border width within each patch that is thrown away after prediction
#   # patchshape_padded = [32,240,240] ## the size of the patch that we feed into the net. must be divisible by 8 or net fails.
#   # patchshape        = [16,200,200] ## must be divisible by 8 to avoid artifacts.
#   # stride            = [16,200,200] ## same as patchshape in this case
#   # def g(n,m): return floor(n/m)*m-n ## f(n,m) gives un-padding needed for n to be divisible by m

#   # a,b = img.shape[-2:]
#   # if a<800 and b<800:
#   #   with torch.no_grad():
#   #     x = torch.from_numpy(img).cuda().float()
#   #     return net(x[None])[0].detach().cpu().numpy()


#   def f(n,m): return ceil(n/m)*m-n ## gives padding needed for n to be divisible by m

#   assert img.ndim==3
#   b,c = img.shape[1:]

#   ## extra border needed for stride % 8 = 0. read as e.g. "ImagePad_Z"
#   # ip_y,ip_x = f(b,8),f(c,8)
  
#   # pp_z,pp_y,pp_x = 8,32,32
#   # DZ,DY,DX = 16,200,200

#   ## max total size with Unet3 16 input channels (64,528,528) = 
#   ## padding per patch. must be divisible by 8. read as e.g. "PatchPad_Z"
#   # pp_z,pp_y,pp_x = 8,64,64
#   pp_y,pp_x = patch_boundary
#   # assert all([x%8==0 for x in patch_boundary])
#   ## inner patch size (does not include patch border. also will be smaller at boundary)
#   DY,DX = patch_inner
#   ps_y,ps_x = np.array(patch_boundary)*2 + patch_inner
#   # assert all([x%8==0 for x in patch_inner])

#   # img_padded = np.pad(img,[(0,0),(pp_y,pp_y+ip_y),(pp_x,pp_x+ip_x)],mode='constant')
#   output = np.zeros((outchan,b,c))

#   ## start coordinates for each patch in the padded input. each stride must be divisible by 8.
#   # zs = np.r_[:a:DZ]

#   ys  = np.r_[:b-ps_y:DY, b-ps_y] ## patch-start coordinate
#   ys2 = ys+pp_y; ys2[0]=0 ## start-coord of inner slice (inclusive)
#   ys3 = ys+ps_y-pp_y; ys3[-1]=b ## end-coordinate of inner slice (exclusive)

#   # for y in ys:
#   # xs = np.r_[:c-ps_x:DX, c-DX]
  
#   # iterdims()

#   ## start coordinates of padded input (e.g. top left patch corner)
#   for x,y in itertools.product(xs,ys):
#     ## end coordinates of padded input (including patch border)
#     ye,xe = min(y+DY,b+ip_y) + 2*pp_y, min(x+DX,c+ip_x) + 2*pp_x
#     patch = img_padded[:,y:ye,x:xe]
#     with torch.no_grad():
#       patch = torch.from_numpy(patch).float().cuda()
#       patch = net(patch[None])[0,:,pp_y:-pp_y,pp_x:-pp_x].detach().cpu().numpy()
#     ## end coordinates of unpadded output (not including border)
#     be,ce = min(y+DY,b),min(x+DX,c)
#     ## use "ae-z" because patch size changes near boundary
#     output[:,y:be,x:ce] = patch[:,:be-y,:ce-x]

#   return output

def apply_net_tiled_3d(net,img,outchan=1,pp_zyx=(8,64,64), D_zyx=(48,400,400)):
  """
  Turns off gradients.
  Does not perform normalization.
  Applies net to image with dims Channels,Z,Y,X.
  Assume 3x or less max pooling layers => (U-net) discrete translational symmetry with period 2^n for n in [0,1,2,3].
  """

  # borders           = [8,24,24] ## border width within each patch that is thrown away after prediction
  # patchshape_padded = [32,240,240] ## the size of the patch that we feed into the net. must be divisible by 8 or net fails.
  # patchshape        = [16,200,200] ## must be divisible by 8 to avoid artifacts.
  # stride            = [16,200,200] ## same as patchshape in this case
  # def g(n,m): return floor(n/m)*m-n ## f(n,m) gives un-padding needed for n to be divisible by m
  def f(n,m): return ceil(n/m)*m-n ## gives padding needed for n to be divisible by m

  assert img.ndim==4
  a,b,c = img.shape[1:]

  ## extra border needed for stride % 8 = 0. read as e.g. "ImagePad_Z"
  ip_z,ip_y,ip_x = f(a,8),f(b,8),f(c,8) 
  
  # pp_z,pp_y,pp_x = 8,32,32
  # DZ,DY,DX = 16,200,200

  ## max total size with Unet3 16 input channels (64,528,528) = 
  ## padding per patch. must be divisible by 8. read as e.g. "PatchPad_Z"
  # pp_z,pp_y,pp_x = 8,64,64
  pp_z,pp_y,pp_x = pp_zyx
  # assert all([x%8==0 for x in pp_zyx])
  ## inner patch size (does not include patch border. also will be smaller at boundary)
  DZ,DY,DX = D_zyx
  # assert all([x%8==0 for x in D_zyx])

  img_padded = np.pad(img,[(0,0),(pp_z,pp_z+ip_z),(pp_y,pp_y+ip_y),(pp_x,pp_x+ip_x)],mode='constant')
  # outchan = 1
  output = np.zeros((outchan,a,b,c))

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
      patch = torch.from_numpy(patch).float().cuda()
      patch = net(patch[None])[0,:,pp_z:-pp_z,pp_y:-pp_y,pp_x:-pp_x].detach().cpu().numpy()
    ## end coordinates of unpadded output (not including border)
    ae,be,ce = min(z+DZ,a),min(y+DY,b),min(x+DX,c)
    ## use "ae-z" because patch size changes near boundary
    output[:,z:ae,y:be,x:ce] = patch[:,:ae-z,:be-y,:ce-x]

  return output

def predict_raw(net,img,dims,**kwargs):
  """
  each elem of N dimension sent to gpu separately.
  When possible, try to make the output dimensions match the input dimensions by e.g. removing singleton dims.
  """
  assert dims in ["NCYX","NBCYX","CYX","ZYX","CZYX","NCZYX","NZYX","YX"]

  with torch.no_grad():
    if dims=="NCYX":
      # def f(i): return net(torch.from_numpy(img[[i]]).cuda().float()).cpu().numpy()[0]
      def f(i): return apply_net_2d(net,img[[i]],**kwargs)
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="NBCYX":
      # def f(i): return net(torch.from_numpy(img[i]).cuda().float()).cpu().numpy()
      def f(i): return apply_net_2d(net,img[i],**kwargs)
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="CYX":
      res = apply_net_2d(net,img,**kwargs)
      # res = net(torch.from_numpy(img[None]).cuda().float()).cpu().numpy()[0]
    if dims=="YX":
      res = apply_net_2d(net,img[None],**kwargs)[0]
      # res = net(torch.from_numpy(img[None,None]).cuda().float()).cpu().numpy()[0,0]
    if dims=="ZYX":
      ## assume 1 channel. remove after prediction.
      res = apply_net_tiled_3d(net,img[None],**kwargs)[0]
    if dims=="CZYX":
      res = apply_net_tiled_3d(net,img)
    if dims=="NCZYX":
      def f(i): return apply_net_tiled_3d(net,img[i],**kwargs)[0]
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="NZYX":
      def f(i): return apply_net_tiled_3d(net,img[i,None],**kwargs)[0]
      res = np.array([f(i) for i in range(img.shape[0])])
  return res

def predict_keepchan(net,img,dims,**kwargs):
  """
  each elem of N dimension sent to gpu separately.
  When possible, try to make the output dimensions match the input dimensions by e.g. removing singleton dims.
  """
  assert dims in ["NCYX","NBCYX","CYX","ZYX","CZYX","NCZYX","NZYX","YX"]

  with torch.no_grad():
    if dims=="NCYX":
      # def f(i): return net(torch.from_numpy(img[[i]]).cuda().float()).cpu().numpy()[0]
      def f(i): return apply_net_2d(net,img[[i]],**kwargs)
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="NBCYX":
      # def f(i): return net(torch.from_numpy(img[i]).cuda().float()).cpu().numpy()
      def f(i): return apply_net_2d(net,img[i],**kwargs)
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="CYX":
      res = apply_net_2d(net,img,**kwargs)
      # res = net(torch.from_numpy(img[None]).cuda().float()).cpu().numpy()[0]
    if dims=="YX":
      res = apply_net_2d(net,img[None],**kwargs)
      # [0]
      # res = net(torch.from_numpy(img[None,None]).cuda().float()).cpu().numpy()[0,0]
    if dims=="ZYX":
      ## assume 1 channel. remove after prediction.
      res = apply_net_tiled_3d(net,img[None],**kwargs) #[0]
    if dims=="CZYX":
      res = apply_net_tiled_3d(net,img)
    if dims=="NCZYX":
      def f(i): return apply_net_tiled_3d(net,img[i],**kwargs)[0]
      res = np.array([f(i) for i in range(img.shape[0])])
    if dims=="NZYX":
      def f(i): return apply_net_tiled_3d(net,img[i,None],**kwargs)[0]
      res = np.array([f(i) for i in range(img.shape[0])])
  return res








