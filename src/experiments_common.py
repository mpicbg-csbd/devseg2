"""
## blocking cuda enables straightforward time profiling
export CUDA_LAUNCH_BLOCKING=1
ipython

import denoiser, detector, tracking
import networkx as nx
#import numpy_indexed as ndi
import numpy as np
from segtools.ns2dir import load,save,toarray
#import experiments2 as ex
import analysis2, ipy
import e21_isbidet as e21
%load_ext line_profiler
"""

import ipdb
import itertools
from math import floor,ceil
import numpy as np
# from scipy.ndimage import label,zoom
from skimage.feature  import peak_local_max
from skimage.measure  import regionprops
from pathlib import Path
from segtools.ns2dir import load,save,flatten_sn,toarray
from segtools import torch_models
from types import SimpleNamespace
import torch
# from torch import nn
from segtools.numpy_utils import normalize3, perm2, collapse2, splt
from segtools import point_matcher
from subprocess import run, Popen
import shutil
from segtools.point_tools import trim_images_from_pts2
from scipy.ndimage import zoom
import json
from scipy.ndimage.morphology import binary_dilation

import tracking
import denoiser, denoise_utils
import detector #, detect_utils
import detector2

from isbi_tools import get_isbi_info, isbi_datasets, isbi_scales
from glob import glob
import os
import re
from skimage.util import view_as_windows
from expand_labels_scikit import expand_labels

from scipy.ndimage.morphology import distance_transform_edt
import datagen

from datagen import mantrack2pts, place_gaussian_at_pts, normalize3, sample_flat, sample_content, augment, weights
from segtools.point_matcher import match_points_single, match_unambiguous_nearestNeib
from tracking import nn_tracking_on_ltps, random_tracking_on_ltps



savedir = Path('/projects/project-broaddus/devseg_2/expr/')

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

def iterdims(shape):
  return itertools.product(*[range(x) for x in shape])
