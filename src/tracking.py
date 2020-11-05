"""
single cell tracking example from `https://napari.org/tutorials/applications/cell_tracking`
"""

import os
# import napari
import ipdb
from pathlib import Path

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.measure import regionprops_table

from segtools.ns2dir import load, save, toarray
from segtools.math_utils import conv_at_pts_multikern

import networkx as nx
import matplotlib
from matplotlib import pyplot as plt

from segtools.color import relabel_from_mapping
from segtools import point_tools
import numpy_indexed as ndi
from pykdtree.kdtree import KDTree as pyKDTree
import re

from types import SimpleNamespace

COLUMNS = ['label', 'frame', 'centroid-0', 'centroid-1', 'centroid-2']


def pad_and_stack_arrays(list_of_arrays, align_pt=None):
  """
  arrays don't need to be the same shape
  align_pt is list of pointers to array align_pt. to be aligned.
  TODO: make it work for arrays with channels, etc.
  """
  if align_pt is None: align_pt = np.zeros([len(list_of_arrays), list_of_arrays[0].ndim])
  far_corner = np.array([x.shape for x in list_of_arrays])
  leftpad  = align_pt.max(0) - align_pt
  far_corner = far_corner + leftpad
  rightpad = far_corner.max(0) - far_corner
  def f(_r):
    x,lp,rp = _r
    p = np.stack([lp,rp],axis=1)
    r = np.pad(x,pad_width=p)
    return r
  res = np.stack([f(_r) for _r in zip(list_of_arrays,leftpad,rightpad)])
  return res

def evaluate_isbi(base_dir,detname,pred='01',fullanno=True):
  "evalid is a unique ID that prevents us from overwriting DET_log files from different experiments predicting on the same data."
  fullanno = '' if fullanno else '0'

  cmd = dict(
    localTRA="/Users/broaddus/Downloads/EvaluationSoftware_1/Mac/TRAMeasure",
    remoteDET="/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/DETMeasure",
  )[cmd]

  DET_command = f"""
  time {cmd} {base_dir} {pred} 3 {fullanno}
  cd {base_dir}/{pred}_RES/
  mv DET_log.txt {detname}
  """
  run([DET_command],shell=True)

def test_simple_tracking():
  res = []
  x = np.random.rand(100,2) * 100
  for i in range(20):
    x += np.random.rand(100,2)*2
    res.append(x.copy())
  return res

def test_cele_simple_tracking(tracklets):
  ltps = list(ndi.group_by(tracklets[:,1]).split(tracklets[:,2:]))
  dists, parents = nn_tracking(ltps)

def draw(tb):
  pos  = nx.multipartite_layout(tb,subset_key='time')
  cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))
  # colors = np.array([tb.edges[e]['track'] for e in tb.edges])
  colors = np.array([tb.nodes[n]['track'] for n in tb.nodes])
  # return pos
  # plt.scatter(pos)
  # nx.draw(tb,pos=pos,node_size=30,edge_color=cmap(colors))
  nx.draw(tb,pos=pos,node_size=30,node_color=cmap(colors))





def nn_tracking_on_ltps(ltps=None, scale=(1,1,1), dub=None):
  """
  ltps should be time-ordered list of ndarrays w shape (N,M) with M in [2,3].
  x[t+1] matches to first nearest neib of x[t].
  
  points can be indexed by (time,local index ∈ [0,N_t])
  """

  x = ltps
  dists, parents = [],[]

  for i in range(len(x)-1):
    kdt = pyKDTree(x[i]*scale)
    _dis, _ind = kdt.query(x[i+1]*scale, k=1, distance_upper_bound=dub)
    dists.append(_dis)
    parents.append(_ind)
  tb = _parents2tb(parents,ltps)
  return tb

def random_tracking_on_ltps(ltps,):
  """
  connect points at random. first try to use up all the parents, then loop back and try again.
  """
  labels  = [np.arange(len(_x)) for _x in ltps]
  parents = []
  for _time in range(len(ltps)-1):
    l = labels[_time]
    np.random.shuffle(l)
    parents.append(np.resize(l,len(labels[_time+1])))
  tb = _parents2tb(parents,ltps)
  return tb




def nap2lbep(nap):
  tracklets,graph,properties = nap.tracklets,nap.graph,nap.properties

  full_graph = graph.copy()
  for k in set(properties['root_id']): full_graph[k] = 0
  lbep = np.zeros([len(full_graph),4])

  # group by trackid
  for i,track_group in enumerate(ndi.group_by(tracklets[:,0]).split(tracklets)):
    track_id = track_group[0,0]
    lbep[i,0] = track_id
    lbep[i,1] = track_group[:,1].min() # min time for given track
    lbep[i,2] = track_group[:,1].max() # max time for given track
    x = full_graph[track_id]; #print(x)
    lbep[i,3] = x if 'int' in str(type(x)) else x[0]
  lbep = lbep.astype(np.uint16)
  return lbep

def _tb_add_orig_labels(tb,nap):
  origlabelmap = dict()
  tracklets    = nap.tracklets
  # group by time
  for _time,sub_tracklets in enumerate(ndi.group_by(tracklets[:,1]).split(tracklets)):
    for _idx,row in enumerate(sub_tracklets):
      origlabelmap[(_time,_idx)] = row[0] ## track id
  for n in tb.nodes:
    tb.nodes[n]['orig_trackid'] = origlabelmap[n]

def _tb_add_track_labels(tb):
  track_id = 1
  # source = [n for n in tb.nodes if tb.nodes[n]['time']==0]
  source = [n for n,d in tb.in_degree if d==0]
  for s in source:
    for v in nx.dfs_preorder_nodes(tb,source=s):
      tb.nodes[v]['track'] = track_id
      tb.nodes[v]['root']  = s
      if tb.out_degree[v] != 1: track_id+=1
    # tb.nodes[s]['root']  = 0

def _parents2tb(parents,ltps):
  list_of_edges = []
  for time,layer in enumerate(parents):
    list_of_edges.extend([((time+1,n),(time,parent_id)) for n,parent_id in enumerate(layer)])
  tb = nx.from_edgelist(list_of_edges, nx.DiGraph())
  tb = tb.reverse()

  all_nodes = [(_time,idx) for _time in np.r_[:len(ltps)] for idx in np.r_[:len(ltps[_time])]]
  tb.add_nodes_from(all_nodes)
  
  for v in tb.nodes: tb.nodes[v]['time'] = v[0]
  _tb_add_track_labels(tb)
  return tb

def tb2nap(tb,ltps):
  _ltps     = np.concatenate(ltps,axis=0)
  trackid = np.array([n + (tb.nodes[n]['track'],) for n in tb.nodes])
  nodes,trackid = trackid[:,:2],trackid[:,[2]]
  idx     = np.lexsort(nodes.T[[1,0]])
  nodes,trackid = nodes[idx],trackid[idx]
  tracklets = np.concatenate([trackid, nodes[:,[0]], _ltps],axis=1).astype(np.uint)
  idx = np.lexsort(tracklets[:,[1,0]].T)
  tracklets = tracklets[idx]

  graph = dict()
  for e in tb.edges:
    l0 = tb.nodes[e[0]]['track']
    l1 = tb.nodes[e[1]]['track']
    # print(l0)
    if l0!=l1: graph[l1]=l0

  root_id = []
  for n in nodes:
    r = tb.nodes[tuple(n)]['root']
    assert r!=0
    track = tb.nodes[r]['track']
    root_id.append(track)
  properties = dict(root_id=root_id)

  nap = SimpleNamespace()
  nap.tracklets = tracklets
  nap.graph = graph
  nap.properties = properties
  return nap

def nap2ltps(nap):
  ltps = list(ndi.group_by(nap.tracklets[:,1]).split(nap.tracklets[:,2:]))
  return ltps

from collections import defaultdict

def tb2lbep(tb):
  times = defaultdict(list)
  trackset = {tb.nodes[n]['track'] for n in tb.nodes}
  parent   = {t:0 for t in trackset}

  for n in tb.nodes:
    track = tb.nodes[n]['track']
    ps = list(tb.pred[n])
    if ps: ## if has parent
      parent_track = tb.nodes[ps[0]]['track'] 
      if track != parent_track:
        parent[track] = parent_track

    times[track].append(n[0])
  
  lbep = [[track, min(times[track]), max(times[track]), parent[track]] for track in sorted(trackset)]
  lbep = np.array(lbep)
  # idx = np.lexsort(lbep[:,[0]].T)
  # lbep = lbep[idx]
  return lbep

### Reading/Writing to disk


def _load_mantrack(path,dset,idx):

  try:
    stak = load(path + dset + f"_GT/TRA/man_track{idx:03d}.tif")
  except:
    stak = load(path + dset + f"_GT/TRA/man_track{idx:04d}.tif")

  props = regionprops_table(stak, properties=('label', 'bbox', 'image', 'area'))
  
  try:
    # subtract 1 because the upper bbox is not included in the block (as is typical of python ranges)
    props['centroid-0'] = (props['bbox-3']+props['bbox-0']-1)/2
    props['centroid-1'] = (props['bbox-4']+props['bbox-1']-1)/2
    props['centroid-2'] = (props['bbox-5']+props['bbox-2']-1)/2
  except:
    props['centroid-0'] = (props['bbox-2']+props['bbox-0']-1)/2
    props['centroid-1'] = (props['bbox-3']+props['bbox-1']-1)/2

  props['frame'] = np.full(props['label'].shape, idx)
  props = pd.DataFrame(props)

  return props

def load_isbi2nap(path,dset,ntimes,):

  data_df_raw = pd.concat([_load_mantrack(path,dset,idx) for idx in range(ntimes[0],ntimes[1])]).reset_index(drop=True)
  data_df = data_df_raw.sort_values(['label', 'frame'], ignore_index=True)
  
  kern = data_df.sort_values(['area'], ascending=False, ignore_index=True).loc[0,'image']
  align_pt = np.array([x.shape for x in data_df['image']])//2
  kern = pad_and_stack_arrays(data_df['image'], align_pt)
  avg_kern = kern.mean(0) > 0.5

  cols = COLUMNS
  if '2D' in path[-20:]: cols = cols[:-1]
  tracklets = data_df.loc[:,cols].to_numpy().astype(np.uint)

  lbep = np.loadtxt(os.path.join(path, dset+'_GT/TRA', 'man_track.txt'), dtype=np.uint)
  if lbep.ndim==1: lbep=lbep.reshape([1,4])
  full_graph = dict(lbep[:, [0, 3]])
  graph = {k: v for k, v in full_graph.items() if v != 0} ## weird. graph keys != set of all labels (missing root nodes). can only get root nodes from properties... i would prefer to work with tracklets + full_graph?

  def _root(node: int):
    """Recursive function to determine the root node of each independent component."""
    if full_graph[node] == 0:  # we found the root
        return node
    return _root(full_graph[node])
  roots = {k: _root(k) for k in full_graph.keys()}
  properties = {'root_id': [roots[idx] for idx in tracklets[:, 0]]}

  nap = SimpleNamespace()
  nap.tracklets = tracklets
  nap.graph = graph
  nap.properties = properties
  nap.kern = kern
  nap.avg_kern = avg_kern
  return nap

def save_isbi(nap, _kern=None, shape=(35, 512, 708), savedir="napri2isbi_test/"):
  """
  the inverse of `isbifiles_to_napari`
  sort the tracklets by time
  for each time rasterize detections using the correct labels
  write the man_tracks.txt from properties alone.
  """
  tracklets,graph,properties = nap.tracklets,nap.graph,nap.properties

  savedir = Path(savedir)
  savedir.mkdir(parents=True,exist_ok=True)
  for x in savedir.glob('*'): x.unlink()

  lbep = nap2lbep(nap)
  np.savetxt(savedir / "res_track.txt",lbep,fmt='%d')

  ndigits = 4 if tracklets[:,1].max() > 1000 else 3
  labelset = []
  stackset = []
  for sub_tracklets in ndi.group_by(tracklets[:,1]).split(tracklets):
    time   = sub_tracklets[0,1]
    labels = sub_tracklets[:,0]
    labelset.append(labels)
    pts    = sub_tracklets[:,2:].astype(np.int)
    kerns  = [_kern * _id for _id in labels]
    stack  = conv_at_pts_multikern(pts,kerns,shape).astype(np.uint16)
    stackset.append(set(np.unique(stack)))
    if ndigits==3: 
      save(stack, savedir / f"mask{time:03d}.tif")
    elif ndigits==4:
      save(stack, savedir / f"mask{time:04d}.tif")
  return lbep, labelset, stackset

def save_permute_existing(tb, path, dset, ntimes, savedir="napri2isbi_test"):
  savedir = Path(savedir)
  path    = Path(path)

  time_offset = None
  for _time, name in enumerate(sorted((path/(dset+"_GT/TRA/")).glob("*.tif"))):
    ## WARNING: _time is pseudotime, which doesn't correspond with actual timestring for PSC PhC-C2DL-PSC datasets!!
    name = str(name)
    lab = load(name)
    mapping = {tb.nodes[n]['orig_trackid']:tb.nodes[n]['track'] for n in tb.nodes if n[0]==_time}
    lab2 = relabel_from_mapping(lab,mapping).astype(np.uint16)

    timestring = re.search(r"(\d{3,4})\.tif",name).group(1)
    if time_offset is None: time_offset=int(timestring)
    save(lab2, savedir / f"mask{timestring}.tif")

  # lbep = nap2lbep(tb2nap(tb,nap2ltps(nap)))
  lbep = tb2lbep(tb)
  lbep[:,[1,2]] = lbep[:,[1,2]]+time_offset
  np.savetxt(savedir / "res_track.txt",lbep,fmt='%d')
  return lbep



def compare_all_labelsets(nap=None,lbep=None,tradir=None,tb=None):
  
  all_lsds = dict()

  resdir = Path(resdir)
  if tradir:
    lbep2 = np.loadtxt(resdir / 'res_track.txt').astype(np.uint16)

  lbep_lsd = lbep2lsd(lbep)
  tradir_lsd = TRAdir2lsd(resdir)

def compare_lsds(lsd1,lsd2):
  for k in sorted(set.union(set(lsd1.keys()),set(lsd2.keys()))):
    print(k, len(lsd1[k]-lsd2[k]), len(lsd2[k]-lsd1[k]))

def nap2lsd(nap):
  tracklets = nap.tracklets
  lsd = dict()
  for x in ndi.group_by(tracklets[:,1]).split(tracklets):
    lsd[x[0,1]] = set(x[:,0])
  return lsd

def TRAdir2lsd(tradir,):
  tradir = Path(tradir)
  lsd = dict()
  for name in sorted(tradir.glob("*.tif")):
    time = int(re.search(r"(\d{3,4})\.tif", str(name)).group(1))
    img  = load(name)
    lsd[time] = set(np.unique(img))
  return lsd

def load_lbep(name):
  lbep = np.loadtxt(name).astype(np.uint8)
  if lbep.ndim == 1: lbep = lbep.reshape([1,4])
  return lbep

def lbep2lsd(lbep):
  """
  lbep -> label set dict
  """
  from collections import defaultdict
  mintime, maxtime = lbep[:,1].min(), lbep[:,2].max()
  lsd = defaultdict(set)
  # for time in np.r_[mintime:maxtime+1]:
  #   lsd[time] = {x[0] for x in lbep if x[1] <= time <= x[1]}
  for row in lbep:
    for time in np.r_[row[1]:row[2]+1]:
      lsd[time].add(row[0])

  return lsd




"""bash
isbidir=/Users/broaddus/Desktop/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/
localtra=/Users/broaddus/Downloads/EvaluationSoftware_1/Mac/TRAMeasure
rm "$isbidir"01_RES/*
cp -r napri2isbi_test/* "$isbidir"01_RES/
time $localtra $isbidir 01 3

isbidir=/projects/project-broaddus/rawdata/celegans_isbi/Fluo-N3DH-CE/
localtra=/projects/project-broaddus/comparison_methods/EvaluationSoftware/Linux/TRAMeasure
cp -r nap2isbi/ "$isbidir"01_RES/
time $localtra $isbidir 01 3
"""



"""
TODO: 
- make it work for 2D/3D better
- replace pandas with numpy_indexed?
- 

ways to specify nodes:
- tuple of (time, local label in lab-img) [potentially local label correspond to tracks across time]
- global index into row of tracklets
- 



"""