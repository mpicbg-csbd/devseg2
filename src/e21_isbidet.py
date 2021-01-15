from experiments_common import *
from pykdtree.kdtree import KDTree as pyKDTree
from segtools.render import rgb_max

print(savedir)

_gpu  = "-p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 "
_cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
slurm = 'sbatch -J e21_{pid:03d} {_resources} -o slurm/e21_pid{pid:03d}.out -e slurm/e21_pid{pid:03d}.err --wrap \'python3 -c \"import e21_isbidet_copy as ex; ex.slurm_entry({pid})\"\' '
slurm = slurm.replace("{_resources}",_cpu)


def run_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.
  shutil.copy("/projects/project-broaddus/devseg_2/src/e21_isbidet.py", "/projects/project-broaddus/devseg_2/src/e21_isbidet_copy.py")
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)

def slurm_entry(pid=0):
  (p0,p1),pid = parse_pid(pid,[2,19])
  for p2,p3 in iterdims([2,5]):
    try:
      run([p0,p1,p2,p3])
    except:
      print("FAIL on pids", [p0,p1,p2,p3])


def _traintimes(info):
  t0,t1 = info.start, info.stop
  N = t1 - t0
  # pix_needed   = time_total * np.prod(batch_shape)
  # pix_i_have   = N*info.shape
  # cells_i_have = len(np.concatenate(ltps))

  # Nsamples = int(N**0.5)
  # if info.ndim==2: Nsamples = 2*Nsamples
  Nsamples = 5 if info.ndim==3 else min(N//2,100)
  gap = ceil(N/Nsamples)
  # dt = gap//2
  # train_times = np.r_[t0:t1:gap]
  # vali_times  = np.r_[t0+dt:t1:3*gap]
  train_times = np.r_[t0:t1:gap]

  # pred_times  = np.r_[t0:t1]
  # assert np.in1d(train_times,vali_times).sum()==0
  # return train_times, vali_times, pred_times
  return train_times

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

def _specialize(P,myname,info):
  if myname in ["celegans_isbi","A549","A549-SIM","H157","hampster","Fluo-N3DH-SIM+"]:
      P.zoom = {3:(1,0.5,0.5), 2:(0.5,0.5)}[info.ndim]
  if myname=="trib_isbi":
    P.kern = [3,3,3]
  if myname=="MSC":
    a,b = info.shape
    P.zoom = {'01':(1/4,1/4), '02':(128/a, 200/b)}[info.dataset]
    ## '02' rescaling is almost exactly isotropic while still being divisible by 8.
  if info.isbiname=="DIC-C2DH-HeLa":
    P.kern = [7,7]
    P.zoom = (0.5,0.5)
  if myname=="fly_isbi":
    pass
    # cfig.bg_weight_multiplier=0.0
    # cfig.weight_decay = False

def run(pid=0):
  """
  v01 : refactor of e18. make traindata AOT.
  add `_train` 0 = predict only, 1 = normal init, 2 = continue
  datagen generator -> StandardGen, customizable loss and validation metrics, and full detector2 refactor.
  v02 : optimizing hyperparams. in [2,19,5,2]
    p0: dataset in [01,02]
    p1: isbi dataset in [0..18]
    p2: sample flat vs content [0,1]
    p3: random variation [0..4]
    p4: pixel weights (should be per-dataset?) [0,1]
  v03 : explore kernel size. [2,19,10]
    p2: kernel size

    p3: jitter: directed|random
    p4: jitter magnitude?
    we want to show performance vs jitter. [curve]. shape. 
  """

  (p0,p1,p2),pid = parse_pid(pid,[2,19,10])

  myname, isbiname  = isbi_datasets[p1]
  trainset = ["01","02"][p0]
  kernel_size = p2+1
  info = get_isbi_info(myname,isbiname,trainset)
  P = _init_params(info.ndim)
  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)
  print(_traintimes(info))
  train_data_files = [(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=n),
                       f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}_GT/TRA/" + info.man_track.format(time=n),
            )
          for n in _traintimes(info)]
  _specialize(P,myname,info)

  ## v03 ONLY
  P.kern = np.array(kernel_size)*[1,1]
  P.nms_footprint = [3,3] #np.ones(().astype(np.int))

  savedir_local = savedir / f'e21_isbidet/v03/pid{pid:03d}/'

  class StandardGen(object):
    def __init__(self, filenames):
      data   = np.array([SimpleNamespace(raw=zoom(load(r),P.zoom,order=1),lab=zoom(load(l),P.zoom,order=0)) for r,l in filenames],dtype=np.object)
      ndim   = data[0].raw.ndim
      for d in data: d.pts = mantrack2pts(d.lab)
      for d in data: d.target = place_gaussian_at_pts(d.pts,d.lab.shape,P.kern)
      for d in data: d.raw = normalize3(d.raw,2,99.4,clip=False)
      self.data  = data

    def sample(self,time,train_mode=True):
      N = len(self.data)
      Nvali  = ceil(N/8)
      Ntrain = N-Nvali
      idxs = np.r_[:Ntrain] if train_mode else np.r_[Ntrain:N]
      sampler = [sample_flat, sample_content][1] #[p2] now fixed. always content sampling.
      x,yt = sampler(self.data[idxs],P.patch)
      if train_mode:
        x,yt = augment(x,yt)
      if myname=='fly_isbi':
        w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=0.0)
      else:
        # w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,)
        w = np.ones_like(yt)
      s = SimpleNamespace(x=x,yt=yt,w=w)
      s.yt_pts = peak_local_max(yt+np.random.rand(*yt.shape)*1e-5,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
      # ipdb.set_trace()
      return s

    def sampleMax(self,t):
      s = self.sample(t)
      if s.x.ndim==3:
        s.x = s.x.max(0)
        s.yt = s.yt.max(0)
        s.w = s.w.max(0)
      return s

    def pred_many(self,net,filenames,savedir=None):
      gtpts = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
      ltps = []
      # _best_f1_score = 0.0
      dims = "ZYX" if info.ndim==3 else "YX"
      # for i in range(len(filenames)):

      s0  = set(_traintimes(info)) #list()
      s1  = set(range(info.start,info.stop)) - s0
      _t0 = list(np.random.choice(list(s0),min(10,len(s0)),replace=False))
      _t1 = list(np.random.choice(list(s1),min(10,len(s1)),replace=False))
      # _t0 = [0, 78] 
      # _t1 = [101, 129]
      # ipdb.set_trace()
      print(_t0,_t1)
      for i in _t0+_t1:
        print(i)
        raw = load(filenames[i])
        raw = normalize3(raw,2,99.4,clip=False)
        x = zoom(raw,P.zoom,order=1)
        res = torch_models.predict_raw(net,x,dims=dims).astype(np.float32)
        pts = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
        pts = pts/P.zoom
        res = zoom(res, 1/np.array(P.zoom), order=1)
        matching = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=info.scale)
        print(matching.f1)
        # fp, fn = find_errors(raw,matching)
        # if savedir: save(pts,savedir / "predpts/pts{i:04d}.pkl")

        if savedir: # and matching.f1>_best_f1_score and i not in _traintimes(info):
          # _best_f1_score = matching.f1
          # ipdb.set_trace()
          save(raw.astype(np.float16), savedir/f"pred/d{i:04d}/raw.tif")
          save(res, savedir/f"pred/d{i:04d}/pred.tif")
          save(pts, savedir/f"pred/d{i:04d}/pts.pkl")
          save(gtpts[i], savedir/f"pred/d{i:04d}/pts_gt.pkl")
          # save(fp, savedir/f"errors/t{i:04d}/fp.pkl")
          # save(fn, savedir/f"errors/t{i:04d}/fn.pkl")
        ltps.append(pts)
      return ltps

  def find_points_within_patches(centerpoints, allpts, _patchsize):
    kdt = pyKDTree(allpts)
    # ipdb.set_trace()
    N,D = centerpoints.shape
    dists, inds = kdt.query(centerpoints, k=10, distance_upper_bound=np.linalg.norm(_patchsize)/2)
    def _test(x,y):
      return (np.abs(x-y) <= np.array(_patchsize)/2).all()
    pts = [[allpts[n] for n in ns if n<len(allpts) and _test(centerpoints[i],allpts[n])] for i,ns in enumerate(inds)]
    return pts

  def find_errors(img,matching):
    from segtools.point_tools import patches_from_centerpoints
    # ipdb.set_trace()
    _patchsize = (7,65,65)
    _patchsize = (33,33)

    pts = matching.pts_yp[~matching.yp_matched_mask]
    patches = patches_from_centerpoints(img, pts, _patchsize)
    yp_in = find_points_within_patches(pts, matching.pts_yp, _patchsize)
    gt_in = find_points_within_patches(pts, matching.pts_gt, _patchsize)
    fp = [SimpleNamespace(pt=pts[i],patch=patches[i],yp=[pts[i]] + yp_in[i],gt=gt_in[i]) for i in range(pts.shape[0])]

    pts = matching.pts_gt[~matching.gt_matched_mask]
    patches = patches_from_centerpoints(img, pts, _patchsize)
    yp_in = find_points_within_patches(pts, matching.pts_yp, _patchsize)
    gt_in = find_points_within_patches(pts, matching.pts_gt, _patchsize)
    fn = [SimpleNamespace(pt=pts[i],patch=patches[i],yp=yp_in[i],gt=[pts[i]] + gt_in[i]) for i in range(pts.shape[0])]

    return fp, fn

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
    score = match_unambiguous_nearestNeib(s.yt_pts,pts,dub=10,scale=info.scale)
    return score.f1

  def _config():
    cfig = SimpleNamespace()
    cfig.getnet = P.getnet
    cfig.time_validate = 100
    cfig.time_total = 10_000 if info.ndim==3 else 15_000 ## # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr on 3D data (10x faster on 2D?)
    cfig.n_vali_samples = 10
    cfig.lr = 4e-4
    cfig.savedir = savedir_local
    cfig.loss = _loss
    cfig.vali_metrics = [height, point_match]
    cfig.vali_minmax  = [None,np.max]
    return cfig

  cfig = _config()

  print("Running e21 with savedir: \n", cfig.savedir, flush=True)

  # x = set(_traintimes(info))
  # y = set(np.r_[:info.stop])
  # print(len(x))
  # return

  dg = StandardGen(train_data_files); cfig.datagen = dg
  # ipdb.set_trace()
  # save(dg.data[0].target.astype(np.float32), cfig.savedir / 'target_t_120.tif')
  
  T = detector2.train_init(cfig)
  save([dg.sampleMax(0) for _ in range(10)],cfig.savedir/"traindata.pkl")
  # T = detector2.train_continue(cfig,cfig.savedir / 'm/best_weights_loss.pt')
  detector2.train(T)

  net = cfig.getnet().cuda()
  net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_loss.pt"))

  prednames = [f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=i) for i in range(info.start,info.stop)]

  # prednames = [f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=i) for i in [120]]
  # raw = load(prednames[0])
  # raw = normalize3(raw,2,99.4,clip=False)
  # x   = zoom(raw,P.zoom,order=1)
  # res = torch_models.predict_raw(net,x,dims="ZYX").astype(np.float32)
  # res = zoom(res,np.array([1])/P.zoom,)
  # save(res,cfig.savedir / 'pred_t_120.tif')
  # return

  ltps = dg.pred_many(net,prednames,savedir=cfig.savedir)
  # ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  # tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  # scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()

def zmax(x):
  if x.ndim==3: 
    # return np.argmax(x,axis=0)
    return x.max(0)
  return x



def old_shit():
  ## After Training, Predict on some stuff

  testset = trainset

  ## Reload best weights
  # net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_point_match.pt"))
  # net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_latest.pt"))


  ## Predict and Evaluate model result on all available data

  gtpts = load(f"/projects/project-broaddus/rawdata/{myname}/traj/{isbiname}/{testset}_traj.pkl")
  # if myname == "celegans_isbi" and testset=='01':
  #   gtpts[6] = ltps[6]
  #   gtpts[7] = ltps[7]

  scores = []
  pred   = []
  raw    = []
  ltps_pred = dict()
  info_test = get_isbi_info(myname,isbiname,testset)
  for i in range(info_test.start, info_test.stop):
    rawname = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{testset}/" + info_test.rawname.format(time=i)
    print(rawname)
    x = load(rawname)
    if P.zoom: x = zoom(x,P.zoom)
    x  = normalize3(x,2,99.4,clip=False)
    res  = torch_models.predict_raw(net,x,dims=dims).astype(np.float32)
    pts  = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
    if P.zoom:
      pts = pts/P.zoom
    score3  = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=3, scale=info.scale)
    score10 = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=info.scale)
    
    print("time", i, "gt", score10.n_gt, "match", score10.n_matched, "prop", score10.n_proposed)
    ltps_pred[i] = pts
    
    # save(res, cfig.savedir / f'pred/t{i:03d}.npy')
    # save(x, cfig.savedir / f'raw/t{i:03d}.npy')
    # save(pts, cfig.savedir / f'pts/t{i:03d}.npy')
    # save(gtpts[i], cfig.savedir / f'gtpts/t{i:03d}.npy')


    # if i==pred_times[0]:
    # s = {3:score3,10:score10}
    # scores.append(s)
    # pred.append(zmax(res))
    # raw.append(zmax(x))

  save(ltps_pred, cfig.savedir / f'ltps_{testset}.pkl')
  # save(scores, cfig.savedir / f'scores_{testset}.pkl')
  # save(np.array(pred).astype(np.float16), cfig.savedir / f"pred_{testset}.npy")
  # save(np.array(raw).astype(np.float16),  cfig.savedir / f"raw_{testset}.npy")