import numpy as np
import re
from types import SimpleNamespace
from isbi_tools import get_isbi_info, isbi_datasets
import json
from experiments_common import parse_pid
from glob import glob
from skimage.measure  import regionprops
from segtools.ns2dir import load,save
from pathlib import Path


def segdata(info):
  labnames = sorted(glob(f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/SEG/*.tif"))
  def f(n_lab):
    _d = info.ndigits
    _time,_zpos = re.search(r"man_seg_?(\d{3,4})_?(\d{3,4})?\.tif",n_lab).groups()
    _time = int(_time)
    if _zpos: _zpos = int(_zpos)
    n_raw = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname.format(time=_time)
    lab = load(n_lab)
    rp = regionprops(lab)
    rp = SimpleNamespace(bbox=np.array([x['bbox'] for x in rp]),centroid=np.array([x['centroid'] for x in rp]))
    D = lab.ndim
    rp.boxdims = rp.bbox[:,D:] - rp.bbox[:,:D]
    return SimpleNamespace(raw=n_raw,lab=n_lab,time=_time,zpos=_zpos,rp=rp)
  return [f(x) for x in labnames]

def run(pid=0):
  (p0,p1),pid = parse_pid(pid,[19,2])
  # savedir_local = savedir / f'e21_isbidet/v09/pid{pid:03d}/'
  savedir_local = Path("/projects/project-broaddus/devseg_2/data/seginfo/")
  (savedir_local / 'm').mkdir(exist_ok=True,parents=True) ## place to save models
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  # P = _init_params(info.ndim)
  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)
  seginfo = segdata(info)
  
  save(seginfo, savedir_local / f"seginfo-{isbiname}-{trainset}.pkl")



  # for x in seginfo:
  #   print(x.rp.boxdims.mean(0),x.rp.centroid.mean(0))
