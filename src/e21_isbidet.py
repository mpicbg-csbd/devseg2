from experiments_common import *

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

def specialize(P,myname,info):
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

def slurm_entry(pid=0):
  (p0,p1),pid = parse_pid(pid,[2,19])
  for p2,p3 in iterdims([2,5]):
    try:
      run([p0,p1,p2,p3])
    except:
      print("FAIL on pids", [p0,p1,p2,p3])

def run(pid=0):
  """
  v01 : refactor of e18. make traindata AOT.
  add `_train` 0 = predict only, 1 = normal init, 2 = continue
  datagen generator -> StandardGen, customizable loss and validation metrics, and full detector2 refactor.
  v02 : optimizing hyperparams. 
  """

  (p0,p1,p2,p3),pid = parse_pid(pid,[2,19,2,5])
  # p2 sample flat vs content
  # p3 random variation
  # ~~p4 pixel weights (should be per-dataset?)~~

  myname, isbiname  = isbi_datasets[p1]
  trainset = ["01","02"][p0]
  info = get_isbi_info(myname,isbiname,trainset)
  P = _init_params(info.ndim)
  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)
  train_data_files = [(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=n),
                       f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}_GT/TRA/" + info.man_track.format(time=n),
            )
          for n in _traintimes(info)]

  specialize(P,myname,info)

  savedir_local = savedir / f'e21_isbidet/v02/pid{pid:03d}/'

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
      sampler = [sample_flat, sample_content][p2]
      x,yt = sampler(self.data[idxs],P.patch)
      if train_mode:
        x,yt = augment(x,yt)
      if myname=='fly_isbi':
        w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=0.0)
      else:
        w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,)
      # w = np.ones_like(yt)
      s = SimpleNamespace(x=x,yt=yt,w=w)
      s.yt_pts = peak_local_max(yt,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
      return s

    def sampleMax(self,t):
      s = self.sample(t)
      s.x = s.x.max(0)
      s.yt = s.yt.max(0)
      s.w = s.w.max(0)
      return s

    def pred_many(self,net,filenames,savedir=None):
      ltps = []
      dims = "ZYX" if info.ndim==3 else "YX"
      for i in range(len(filenames)):
        x = zoom(load(filenames[i]),P.zoom,order=1)
        x = normalize3(x,2,99.4,clip=False)
        res = torch_models.predict_raw(net,x,dims=dims).astype(np.float32)
        pts = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
        pts = pts/P.zoom
        if savedir: save(pts,savedir / "pts{i:04d}.pkl")
        ltps.append(pts)
      return ltps

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

  # dg = StandardGen(train_data_files); cfig.datagen = dg
  # save([dg.sampleMax(0) for _ in range(10)],cfig.savedir/"traindata.pkl")
  
  # T = detector2.train_init(cfig)
  # T = detector2.train_continue(cfig,cfig.savedir / 'm/best_weights_loss.pt')
  # detector2.train(T)

  # net = cfig.getnet().cuda()
  # net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_loss.pt"))
  # prednames = [f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=i) for i in range(info.start,info.stop)]
  # ltps = dg.pred_many(net,prednames)
  ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()



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

  def zmax(x):
    if x.ndim==3: 
      # return np.argmax(x,axis=0)
      return x.max(0)
    return x

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