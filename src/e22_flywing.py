"""
Detect and Track cells in Suzanne Eaton's flywing data, which we got from Alex Dibrov.
Dection uses CPNet and results go in to detection chapter of my thesis.
"""

from experiments_common import *
from scipy.ndimage import label

print(savedir)

_gpu  = "-p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 "
_cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
slurm = 'sbatch -J e22_{pid:03d} {_resources} -o slurm/e22_pid{pid:03d}.out -e slurm/e22_pid{pid:03d}.err --wrap \'python3 -c \"import e22_flywing_copy as ex; ex.slurm_entry({pid})\"\' '
slurm = slurm.replace("{_resources}",_cpu)


def run_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.
  shutil.copy("/projects/project-broaddus/devseg_2/src/e22_flywing.py", "/projects/project-broaddus/devseg_2/src/e22_flywing_copy.py")
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)

def slurm_entry(pid=0):
  run(0)

def _init_params(ndim):
  P = SimpleNamespace()
  if ndim==2:
    P.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(2,2),   kernsize=(5,5),   finallayer=torch_models.nn.Sequential)
    P.zoom  = (1,1) #(0.5,0.5)
    P.kern  = [5,5]
    P.patch = (512,512)
    P.nms_footprint = [9,9]
  elif ndim==3:
    P.getnet = lambda : torch_models.Unet3(16, [[1],[1]], pool=(1,2,2), kernsize=(3,5,5), finallayer=torch_models.nn.Sequential)
    P.zoom   = (1,1,1) #(1,0.5,0.5)
    P.kern   = [2,5,5]
    P.patch  = (16,128,128)
    P.nms_footprint = [3,9,9]
  P.patch = np.array(P.patch)
  return P

def zmax(x):
  if x.ndim==3: 
    # return np.argmax(x,axis=0)
    return x.max(0)
  return x

def run(pid=0):
  """
  v01 : new
  """

  (p0,p1,p2,p3),pid = parse_pid(pid,[2,19,2,5])
  P = _init_params(2)
  savedir_local = savedir / f'e22_flywing/v01/pid{pid:03d}/'

  _traintimes = np.r_[:15]

  class StandardGen(object):
    def __init__(self):
      # data   = np.array([SimpleNamespace(raw=zoom(load(r),P.zoom,order=1),lab=zoom(load(l),P.zoom,order=0)) for r,l in filenames],dtype=np.object)
      data = SimpleNamespace()
      raw = list(load("/projects/project-broaddus/rawdata/care_flywing_crops/gt_initial_stack.tif"))
      lab = list(load("/projects/project-broaddus/rawdata/care_flywing_crops/gt_truecontours.tif"))
      lab = [label(~x[...,0])[0] for x in lab]
      # ndim   = data[0].raw.ndim
      pts = [mantrack2pts(x) for x in lab]
      target = [place_gaussian_at_pts(p,lab[0].shape,P.kern) for p in pts]
      raw = [normalize3(x,2,99.4,clip=False) for x in raw]
      data = np.array([SimpleNamespace(raw=a,lab=b,pts=c,target=d) for a,b,c,d in zip(raw,lab,pts,target)])
      self.data  = data

    def sample(self,time,train_mode=True):
      N = len(self.data)
      N = 20
      Nvali  = ceil(N/8)
      Ntrain = N-Nvali
      idxs = np.r_[:Ntrain] if train_mode else np.r_[Ntrain:N]
      sampler = [sample_flat, sample_content][p2]
      x,yt = sampler(self.data[idxs],P.patch)
      if train_mode:
        x,yt = augment(x,yt)
      # w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,)
      w = np.ones_like(yt)
      s = SimpleNamespace(x=x,yt=yt,w=w)
      s.yt_pts = peak_local_max(yt,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
      return s

    def sampleMax(self,t):
      s = self.sample(t)
      s.x = s.x.max(0)
      s.yt = s.yt.max(0)
      s.w = s.w.max(0)
      return s

    def pred_many(self,net,savedir=None):
      # gtpts = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
      gtpts = [d.pts for d in self.data]
      ltps  = []
      # _best_f1_score = 0.0
      # dims = "ZYX" if info.ndim==3 else "YX"
      dims = "YX"
      for i in range(len(self.data)):
        print(i)
        # x = zoom(load(filenames[i]),P.zoom,order=1)
        x = self.data[i].raw.copy()
        x = normalize3(x,2,99.4,clip=False)
        res = torch_models.predict_raw(net,x,dims=dims).astype(np.float32)
        pts = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
        pts = pts/P.zoom
        scores = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=[1,1])
        print(scores.f1)
        if savedir: save(pts,savedir / "predpts/pts{i:04d}.pkl")
        # crop_errors_from_matching(x,res,scores)
        _traintimes = np.r_[:20]
        # if savedir and scores.f1>_best_f1_score and i not in _traintimes:
        if savedir:
          # and i not in _traintimes:
          # _best_f1_score = scores.f1
          save(x, savedir/f"pred/d{i:04d}/raw.tif")
          save(res, savedir/f"pred/d{i:04d}/pred.tif")
          save(pts, savedir/f"pred/d{i:04d}/pts.pkl")
          save(gtpts[i], savedir/f"pred/d{i:04d}/pts_gt.pkl")
        ltps.append(pts)
      return ltps

  def crop_errors_from_matching(x,res,scores):
    from segtools.point_tools import patches_from_centerpoints
    ipdb.set_trace()
    gt = scores.pts_gt
    yp = scores.pts_yp
    # gt
    x_patches = patches_from_centerpoints(x, centerpoints, patchsize=(32,32))


  def _loss(net,sample):
    s  = sample
    x  = torch.from_numpy(s.x).float().cuda()
    yt = torch.from_numpy(s.yt).float().cuda()
    w  = torch.from_numpy(s.w).float().cuda()
    y  = net(x[None,None])[0,0]
    loss = torch.abs((w*(y-yt)**2)).mean() #+ weight*torch.abs((y-global_avg)**2).mean()
    return y,loss

  def height(y,sample):
    return y.max()

  def point_match(y,sample):
    s = sample
    pts   = peak_local_max(y,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
    score = match_unambiguous_nearestNeib(s.yt_pts,pts,dub=10,scale=[1,1])
    return score.f1

  def _config():
    cfig = SimpleNamespace()
    cfig.getnet = P.getnet
    cfig.time_validate = 100
    cfig.time_total = 10_000 #if info.ndim==3 else 15_000 ## # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr on 3D data (10x faster on 2D?)
    cfig.n_vali_samples = 10
    cfig.lr = 4e-4
    cfig.savedir = savedir_local
    cfig.loss = _loss
    cfig.vali_metrics = [height, point_match]
    cfig.vali_minmax  = [None,np.max]
    return cfig

  



  cfig = _config()

  print("Running e22 with savedir: \n", cfig.savedir, flush=True)

  dg = StandardGen(); cfig.datagen = dg
  # save([dg.sampleMax(0) for _ in range(10)],cfig.savedir/"traindata.pkl")
  
  # T = detector2.train_init(cfig)
  # T = detector2.train_continue(cfig,cfig.savedir / 'm/best_weights_loss.pt')
  # detector2.train(T)

  net = cfig.getnet().cuda()
  net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_loss.pt"))
  # prednames = [f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=i) for i in range(info.start,info.stop)]
  ltps = dg.pred_many(net,savedir=cfig.savedir)
  # ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  # tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  # scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()
