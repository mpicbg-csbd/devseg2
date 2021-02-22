from experiments_common import *
from pykdtree.kdtree import KDTree as pyKDTree
from segtools.render import rgb_max
from numpy import r_,s_
from segtools.point_tools import patches_from_centerpoints

print(savedir)

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

def match(yt_pts,pts):
  return match_unambiguous_nearestNeib(yt_pts,pts,dub=30,scale=[3,1,1])


def run(pid=0):
  """
  v01 : new
  """

  (p0,p1,p2,p3),pid = parse_pid(pid,[2,19,2,5])
  P = _init_params(3)
  savedir_local = savedir / f'e23_mauricio/v01/pid{pid:03d}/'

  _traintimes = np.r_[:15]
  _traintimes = [109]

  def loaddata():
    # data   = np.array([SimpleNamespace(raw=zoom(load(r),P.zoom,order=1),lab=zoom(load(l),P.zoom,order=0)) for r,l in filenames],dtype=np.object)
    np.random.seed(0)
    def _f(t):
      raw  = load(f"/projects/project-broaddus/rawdata/ZFishMau2021/coleman/2021_01_21_localphototoxicity_h2brfp_lap2bgfp_G2_Subset_Average_DualSideFusion_max_Subset_forcoleman_T{t}.tif")[:,1]
      pts  = np.array(load(f"/projects/project-broaddus/rawdata/ZFishMau2021/anno/t{t:03d}.pkl")).astype(np.int)
      np.random.shuffle(pts)
      target = place_gaussian_at_pts(pts,raw.shape,P.kern)
      raw  = normalize3(raw,2,99.4,clip=False)
      return SimpleNamespace(**locals())

    data = [_f(t) for t in [0,109]]
    return data

  data = loaddata()

  def sample(time,train_mode=True):
    # N = len(data)
    _i = np.random.randint(len(data))
    d  = data[_i]

    N = d.pts.shape[0]
    Nvali  = ceil(N/8)
    Ntrain = N-Nvali
    # idxs   = np.r_[:Ntrain] if train_mode else np.r_[Ntrain:N]
    # sampler = [sample_flat, sample_content][p2]
    # x,yt = sampler(self.data[idxs],P.patch)

    ## sample a point or sample from the left
    # if np.random.rand()<0.75:
    n0,n1 = (0,Ntrain) if train_mode else (Ntrain,None)
    pt = d.pts[np.random.randint(n0,n1)]
    ss = jitter_center_inbounds(pt,P.patch,d.raw.shape,jitter=0.1)
    # else:
    # pt = (np.random.rand(3)*(d.raw.shape - _patch)).astype(int)
    # _x = d.raw.shape[2]
    # x0,x1 = (0,_x//2) if train_mode else (_x//2, _x[2])
    # pt[2] = np.random.randint(x0,x1)
    # ss = tuple([slice(pt[i],pt[i]+P.patch[i]) for i in range(len(P.patch))])

    x  = d.raw[ss].copy()
    yt = d.target[ss].copy()
    # w  = d.weights[ss].copy()

    if train_mode:
      x,yt = augment(x,yt)

    ## inefficient rejection sampling
    if yt.max()<0.9:
      return sample(time,train_mode)

    # w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,)
    w = np.ones_like(yt)
    s = SimpleNamespace(x=x,yt=yt,w=w)
    s.yt_pts = peak_local_max(yt,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
    return s


  def pred_centerpoint(net,times,dirname='pred',savedir=None):
    # gtpts = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
    # gtpts = [d.pts for d in data]
    # dims = "ZYX" if data[0].raw.ndim==3 else "YX"
    dims = "ZYX"

    def _single(i):
      "i is time"
      print(i)
      name = f"/projects/project-broaddus/rawdata/ZFishMau2021/coleman/2021_01_21_localphototoxicity_h2brfp_lap2bgfp_G2_Subset_Average_DualSideFusion_max_Subset_forcoleman_T{i}.tif"
      gtpts = load(f"/projects/project-broaddus/rawdata/ZFishMau2021/anno/t{i:03d}.pkl").astype(np.int)
      raw = load(name)[:,1]
      raw = normalize3(raw,2,99.4,clip=False)
      x   = zoom(raw,P.zoom,order=1)
      pred = torch_models.predict_raw(net,x,dims=dims).astype(np.float32)
      height = pred.max()
      pred = pred / pred.max() ## 
      pts = peak_local_max(pred,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
      pts = pts/P.zoom
      # matching = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=[3,1,1])
      matching = match(gtpts,pts)
      print(matching.f1)
      pred = zoom(pred, 1/np.array(P.zoom), order=1)
      # fp, fn = find_errors(raw,matching)
      # if savedir: save(pts,savedir / "predpts/pts{i:04d}.pkl")
      target = place_gaussian_at_pts(gtpts,raw.shape,P.kern)
      mse = np.mean((pred-target)**2)
      scores = dict(f1=matching.f1,precision=matching.precision,recall=matching.recall,mse=mse)
      cropped_yp = patches_from_centerpoints(raw,pts,(5,32,32),bounds='constant')
      # cropped_yp = np.array([x.max(0) for x in cropped_yp])
      cropped_yp = cropped_yp.max(1)
      cropped_gt = patches_from_centerpoints(raw,gtpts,(5,32,32),bounds='constant')
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

  # def crop_errors_from_matching(x,res,scores):
  #   ipdb.set_trace()
  #   gt = scores.pts_gt
  #   yp = scores.pts_yp
  #   # gt
  #   x_patches = patches_from_centerpoints(x, centerpoints, patchsize=(32,32))

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
    cfig.time_total = 60_000 #if info.ndim==3 else 15_000 ## # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr on 3D data (10x faster on 2D?)
    cfig.n_vali_samples = 10
    cfig.save_every_n = 1
    cfig.lr = 4e-4
    cfig.savedir = savedir_local
    cfig.loss = _loss
    cfig.vali_metrics = [height] #[height, point_match]
    cfig.vali_minmax  = [None] #[None,np.max]
    return cfig

  
  cfig = _config()

  print("Running e23 with savedir: \n", cfig.savedir, flush=True)

  # dg = StandardGen(); 
  cfig.sample = sample
  # save([dg.sampleMax(0) for _ in range(10)],cfig.savedir/"traindata.pkl")
  
  if 1:
    T = detector2.train_init(cfig)
    # T = detector2.train_continue(cfig,cfig.savedir / 'm/best_weights_loss.pt')
    detector2.train(T)

  net = cfig.getnet().cuda()
  net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_loss.pt"))
  # prednames = [f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=i) for i in range(info.start,info.stop)]
  ltps = pred_centerpoint(net,[0,109],dirname='pred_all',savedir=cfig.savedir)
  # ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  # tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  # scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()
