import numpy as np
from types import SimpleNamespace

def _get_img_meta(wildcards):
  img_meta = SimpleNamespace()
  if wildcards.isbiname=="Fluo-C3DH-A549":
    img_meta.voxel_size = np.array([1.0,0.126,0.126]) ## um
    img_meta.time_step  = 2 ## minutes
  if wildcards.isbiname=="Fluo-C3DL-MDA231":
    img_meta.voxel_size = np.array([6.0,1.242,1.242]) ## um
    img_meta.time_step  = 80 ## minutes
  if wildcards.isbiname=="Fluo-N3DH-CE":
    img_meta.voxel_size = np.array([1.0,0.09,0.09]) ## um
    img_meta.time_step  = 1 ## 1.5 for second dataset? ## minutes
  if wildcards.isbiname=="Fluo-N3DL-DRO":
    img_meta.voxel_size = np.array([2.03,0.406,0.406]) ## um
    img_meta.time_step  = 0.5 ## minutes
  if wildcards.isbiname=="Fluo-N3DL-TRIC": ## 2d
    img_meta.voxel_size = np.array([1,.3,.3]) # 'varying' across xy, because of projection. no units. just a guess
    img_meta.time_step  = 1.5 ## minutes
  if wildcards.isbiname=="Fluo-N3DL-TRIF": ## 3d
    img_meta.voxel_size  = np.array([0.38,0.38,0.38]) ## um
    img_meta.time_step   = 1.5 ## minutes
  return img_meta

def _specialize_predict(wildcards,config):
  if wildcards.isbiname=="Fluo-C3DH-A549":
    config.norm = lambda img: img/2000
  if wildcards.isbiname=="Fluo-C3DL-MDA231":
    config.norm = lambda img: img/4000
  if wildcards.isbiname=="Fluo-N3DH-CE":
    pass
  if wildcards.isbiname=="Fluo-N3DL-DRO":
    pass
  if wildcards.isbiname=="Fluo-N3DL-TRIC": ## 2d
    pass
  if wildcards.isbiname=="Fluo-N3DL-TRIF": ## 3d
    pass
    # config.patch_space = np.array([64,64,64])
  return config

def _specialize_train(wildcards,loader,config):
  """
  All the data-specific param changes live in here
  """
  if wildcards.isbiname=="Fluo-C3DH-A549":
    config.norm = lambda img: img/2000
    config.bg_weight_multiplier = 0.1
  if wildcards.isbiname=="Fluo-C3DL-MDA231":
    config.norm = lambda img: img/4000
  if wildcards.isbiname=="Fluo-N3DH-CE":
    loader.traintimes     = [0,5,33,100,189]
    loader.valitimes      = [0,1,180]
    config.bg_weight_multiplier = 0.2
  if wildcards.isbiname=="Fluo-N3DL-DRO":
    pass
  if wildcards.isbiname=="Fluo-N3DL-TRIC": ## 2d
    pass
  if wildcards.isbiname=="Fluo-N3DL-TRIF": ## 3d
    pass
    # config.patch_space = np.array([64,64,64])
  return loader,config
