from experiments_common import rchoose, partition, shuffle_and_split, parse_pid, iterdims, savedir_global
import datagen

from pykdtree.kdtree import KDTree as pyKDTree
from numpy import r_,s_

from models import CenterpointModel, SegmentationModel
from segtools.render import rgb_max
from segtools.point_tools import patches_from_centerpoints
from segtools import point_matcher
from segtools import torch_models
from augmend import Augmend,FlipRot90,Elastic,Rotate,Scale
import torch
from segtools.ns2dir import load,save
import numpy as np
from segtools.numpy_utils import normalize3
from scipy.ndimage import zoom

from types import SimpleNamespace
from skimage.feature  import peak_local_max
import detector2

savedir = savedir_global()
print("savedir:", savedir)

_gpu  = "-p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 "
_cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
slurm = 'sbatch -J e23_{pid:03d} {_resources} -o slurm/e23_pid{pid:03d}.out -e slurm/e23_pid{pid:03d}.err --wrap \'python3 -c \"import e23_mauricio_copy as ex; ex.slurm_entry({pid})\"\' '
slurm = slurm.replace("{_resources}",_gpu)


def run_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.
  shutil.copy("/projects/project-broaddus/devseg_2/src/e23_mauricio.py", "/projects/project-broaddus/devseg_2/src/e23_mauricio_copy.py")
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)

def slurm_entry(pid=0):
  run(0)

def run(pid=0):
  """
  v01 : new
  """

  (p0,p1,p2,p3),pid = parse_pid(pid,[2,19,2,5])
  savedir_local = savedir / f'e23_mauricio/v01/pid{pid:03d}/'
  print("Running e23 with savedir: \n", savedir_local, flush=True)
  
  CPNet = CPNetMau(savedir_local)

  if 1:
    CPNet.dataloader()
    CPNet.train_cfig.time_total = 30_000
    # T = detector2.train_init(CPNet.train_cfig)
    # T = detector2.train_continue(CPNet.train_cfig, CPNet.savedir / 'm/best_weights_loss.pt')
    # detector2.train(T)
    CPNet.train(_continue=1)

  CPNet.net.load_state_dict(torch.load(CPNet.savedir / "m/best_weights_loss.pt"))
  ltps = pred_centerpoint(CPNet,[0,109],dirname='pred_all',savedir=CPNet.savedir)
  # ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  # tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  # scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()

class CPNetMau(CenterpointModel):
  def __init__(self,savedir,):
    super().__init__(savedir)
    self._init_params(3) #
    self.patch = (8,64,64)
    self.net = torch_models.Unet3(16, [[2],[1]], pool=(1,2,2),   kernsize=(3,5,5),   finallayer=torch_models.nn.Sequential).cuda()
    self.train_cfig.net = self.net
    self.train_cfig.sample = self.sample
    # self.info = info
    self.ndim = 3

    aug = Augmend()
    ax = {2:(0,1), 3:(1,2)}[self.ndim]
    if self.ndim==3:
      aug.add([FlipRot90(axis=1), FlipRot90(axis=0),], probability=1)
      aug.add([FlipRot90(axis=(2,3)), FlipRot90(axis=(1,2)),], probability=1)
    else:
      aug.add([FlipRot90(axis=(0,1)), FlipRot90(axis=(0,1)),], probability=1)
    # aug.add([Elastic(axis=ax, amount=5, order=1),
    #          Elastic(axis=ax, amount=5, order=1),],
    #         probability=0.5)
    # aug.add([Rotate(axis=ax, order=1),
    #          Rotate(axis=ax, order=1),],
    #         probability=0.5)
    self.aug = lambda x: x #aug

  def _init_params(self,ndim):
    # P = SimpleNamespace()
    if ndim==2:
      self.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
      self.zoom  = (1,1) #(0.5,0.5)
      self.kern  = [5,5]
      self.patch = (512,512)
      self.nms_footprint = [9,9]
    elif ndim==3:
      self.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
      self.zoom   = (1,1,1) #(1,0.5,0.5)
      self.kern   = [2,5,5]
      self.patch  = (16,128,128)
      self.nms_footprint = [3,9,9]
    self.patch = np.array(self.patch)

  def dataloader(self):
    # info  = self.info
    n_raw  = "/projects/project-broaddus/rawdata/ZFishMau2021/coleman/2021_01_21_localphototoxicity_h2brfp_lap2bgfp_G2_Subset_Average_DualSideFusion_max_Subset_forcoleman_T{time}.tif"
    n_pts  = "/projects/project-broaddus/rawdata/ZFishMau2021/anno/t{time:03d}.pkl"
    n_class  = "/projects/project-broaddus/rawdata/ZFishMau2021/anno/class{time}.pkl"

    def f(i):
      raw = load(n_raw.format(time=i)).transpose([1,0,2,3])
      raw = zoom(raw,(1,) + self.zoom,order=1)
      raw = normalize3(raw,2,99.4,axs=(1,2,3),clip=False)
      pts = load(n_pts.format(time=i))
      classes = load(n_class.format(time=i))
      pts = [p for i,p in enumerate(pts) if classes[i] in ['p','pm']]
      pts = (np.array(pts) * self.zoom).astype(np.int)
      target = datagen.place_gaussian_at_pts(pts,raw.shape[1:],self.kern)
      slices = datagen.shape2slicelist(raw.shape[1:],self.patch,divisible=(1,8,8)[-self.ndim:])
      hi,low = partition(lambda s: target[s].max()>0.99, slices)
      return SimpleNamespace(raw=raw,pts=pts,target=target,hi=hi,low=low)

    self.data = [f(i) for i in [0,109]]

    hi  = [(i,x) for i,d in enumerate(self.data) for x in d.hi]
    low = [(i,x) for i,d in enumerate(self.data) for x in d.low]
    t_hi,v_hi = shuffle_and_split(hi,valifrac=1/8)
    t_low,v_low = shuffle_and_split(low,valifrac=1/8)
    self.slices_train = SimpleNamespace(hi=t_hi,low=t_low)
    self.slices_vali  = SimpleNamespace(hi=v_hi,low=v_low)

  def sample(self,time,train_mode=True):

    # slices = self.train_slices if train_mode else self.vali_slices
    # i,ss = slices[np.random.choice(len(slices))]
    slices = self.slices_train if train_mode else self.slices_vali
    i,ss   = rchoose(slices.hi) if np.random.rand()<0.99 else rchoose(slices.low)
    # d    = rchoose(data) ## choose time
    # ss   = rchoose(d.content_slices) if np.random.rand()<0.5 else rchoose(d.nondividing_slices)
    # pt   = d.pts[np.random.choice(d.pts.shape[0])] ## choose point
    # ss   = jitter_center_inbounds(pt,self.patch,d.raw.shape,jitter=0.1)
    
    d  = self.data[i]

    x  = d.raw[(slice(None),)+ss].copy()
    yt = d.target[ss].copy()

    if train_mode:
      x,yt = self.aug([x,yt])

    # ipdb.set_trace()
    x  = x.copy()
    yt = yt.copy()
    
    # if self.extern.info.myname=='fly_isbi':
    #   w = weights__decaying_bg_multiplier(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=0.0)
    # else:
    w = datagen.weights__decaying_bg_multiplier(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=1/5)
    # w = np.ones_like(yt)

    s = SimpleNamespace(x=x,yt=yt,w=w)
    ## prevent overdetect on peaks EVEN WITH FOOTPRINT because they have exactly the same value
    # s.yt_pts = peak_local_max(yt+np.random.rand(*yt.shape)*1e-5,threshold_abs=.2,exclude_border=False,footprint=np.ones(self.nms_footprint))
    s.yt_pts = peak_local_max(yt,threshold_abs=.99,exclude_border=False,footprint=np.ones(self.nms_footprint))
    return s

  def mse_loss(self,net,sample):
    s  = sample
    x  = torch.from_numpy(s.x).float().cuda()
    yt = torch.from_numpy(s.yt).float().cuda()
    w  = torch.from_numpy(s.w).float().cuda()
    y  = net(x[None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss

def pred_centerpoint(CPNet,times,dirname='pred',savedir=None):
  # gtpts = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
  # gtpts = [d.pts for d in data]
  # dims = "ZYX" if data[0].raw.ndim==3 else "YX"
  dims = "CZYX"

  def _single(i):
    "i is time"
    print(i)
    name = f"/projects/project-broaddus/rawdata/ZFishMau2021/coleman/2021_01_21_localphototoxicity_h2brfp_lap2bgfp_G2_Subset_Average_DualSideFusion_max_Subset_forcoleman_T{i}.tif"
    gtpts = load(f"/projects/project-broaddus/rawdata/ZFishMau2021/anno/t{i:03d}.pkl").astype(np.int)
    raw = load(name).transpose([1,0,2,3]) #[:,1]
    raw = normalize3(raw,2,99.4,axs=(1,2,3),clip=False)
    x   = zoom(raw,(1,)+CPNet.zoom,order=1)
    pred = torch_models.predict_raw(CPNet.net,x,dims=dims,D_zyx=(24,256,256)).astype(np.float32)[0]
    height = pred.max()
    pred = pred / pred.max() ## 
    pts = peak_local_max(pred,threshold_abs=.2,exclude_border=False,footprint=np.ones(CPNet.nms_footprint))
    pts = pts/CPNet.zoom
    matching = point_matcher.match_unambiguous_nearestNeib(gtpts,pts,dub=100,scale=[3,1,1])
    # matching = match(gtpts,pts)
    print(matching.f1)
    pred = zoom(pred, 1/np.array(CPNet.zoom), order=1)
    # fp, fn = find_errors(raw,matching)
    # if savedir: save(pts,savedir / "predpts/pts{i:04d}.pkl")
    target = datagen.place_gaussian_at_pts(gtpts,raw.shape[1:],CPNet.kern)
    mse = np.mean((pred-target)**2)
    scores = dict(f1=matching.f1,precision=matching.precision,recall=matching.recall,mse=mse)
    cropped_yp = patches_from_centerpoints(raw[1],pts,(5,32,32),bounds='constant')
    # cropped_yp = np.array([x.max(0) for x in cropped_yp])
    cropped_yp = cropped_yp.max(1)
    cropped_gt = patches_from_centerpoints(raw[1],gtpts,(5,32,32),bounds='constant')
    # cropped_gt = np.array([x.max(0) for x in cropped_gt])
    cropped_gt = cropped_gt.max(1)
    return SimpleNamespace(**locals())

  def _save_preds(d,i):
    # _best_f1_score = matching.f1
    save(d.raw.astype(np.float16).max(0), savedir/f"{dirname}/d{i:04d}/raw.tif")
    save(d.pred.astype(np.float16).max(0), savedir/f"{dirname}/d{i:04d}/pred.tif")
    save(d.target.astype(np.float16).max(0), savedir/f"{dirname}/d{i:04d}/target.tif")
    save(d.cropped_yp.astype(np.float16), savedir/f"{dirname}/d{i:04d}/cropped_yp.tif")
    save(d.cropped_gt.astype(np.float16), savedir/f"{dirname}/d{i:04d}/cropped_gt.tif")
    save(d.pts, savedir/f"{dirname}/d{i:04d}/pts.pkl")
    save(d.gtpts, savedir/f"{dirname}/d{i:04d}/pts_gt.pkl")
    save(d.scores, savedir/f"{dirname}/d{i:04d}/scores.pkl")
    save(d.matching, savedir/f"{dirname}/d{i:04d}/matching.pkl")
    # save(fp, savedir/f"errors/t{i:04d}/fp.pkl")
    # save(fn, savedir/f"errors/t{i:04d}/fn.pkl")

  def _f(i): ## i is time
    d = _single(i)
    _save_preds(d,i)
    return d.scores,d.pts,d.height

  scores,ltps,height = map(list,zip(*[_f(i) for i in times]))
  save(scores, savedir/f"{dirname}/scores.pkl")
  save(height, savedir/f"{dirname}/height.pkl")
  save(ltps, savedir/f"{dirname}/ltps.pkl")

  return ltps

if __name__=="__main__":
  run()