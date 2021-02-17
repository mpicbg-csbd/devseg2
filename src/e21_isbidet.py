from experiments_common import *


from pykdtree.kdtree import KDTree as pyKDTree
from segtools.render import rgb_max
from numpy import r_,s_

print("savedir:", savedir)

_gpu  = "-p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 "
_cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
slurm = 'sbatch -J e21_{pid:03d} {_resources} -o slurm/e21_pid{pid:03d}.out -e slurm/e21_pid{pid:03d}.err --wrap \'python3 -c \"import e21_isbidet_copy as ex; ex.slurm_entry({pid})\"\' '
slurm = slurm.replace("{_resources}",_gpu)


def run_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.
  shutil.copy("/projects/project-broaddus/devseg_2/src/e21_isbidet.py", "/projects/project-broaddus/devseg_2/src/e21_isbidet_copy.py")
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)

def slurm_entry(pid=0):
  run(pid)
  # (p0,p1),pid = parse_pid(pid,[2,19,9])
  # run([p0,p1,p2])
  # for p2,p3 in iterdims([2,5]):
  #   try:
  #   except:
  #     print("FAIL on pids", [p0,p1,p2,p3])

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

def _specialize_isbi(P,myname,info):
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
  datagen generator -> CenterpointGen, customizable loss and validation metrics, and full detector2 refactor.
  v02 : optimizing hyperparams. in [2,19,5,2]
    p0: dataset in [01,02]
    p1: isbi dataset in [0..18]
    p2: sample flat vs content [0,1]
    p3: random variation [0..4]
    p4: pixel weights (should be per-dataset?) [0,1]
  v03 : explore kernel size. [2,50]
    p0: two datasets
    p1: kernel size
  v04 : unbiased jitter! [2,50]
    p0: two datasets (GOWT1/p)
    p1: noise_level: noise_level for random jitter
    we want to show performance vs jitter. [curve]. shape. 
  v05 : redo celegans. test training data size. [10,5]:
    p0 : n training images from _early times_
    p1 : repeats ?
  v06 : redo celegans, just the basic training on single images. NO xy SCALING!
    p0 : timepoint to use
    p1 : repeats
  v07 : sampling methods: does it matter what we use? [2,5]
    p0 : [iterative sampling, content sampling]
    p1 : repeats
  """

  (p0,p1),pid = parse_pid(pid,[2,5])

  myname, isbiname = isbi_datasets[11] # v05
  trainset = '02' #["01","02"][p0] # v06 
  info = get_isbi_info(myname,isbiname,trainset)
  P = _init_params(info.ndim)
  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)
  print(_traintimes(info))
  # _ttimes = [[6,7],[100,101],[180,181],[6,100,180,7,101,181]][p0]
  
  # ## v06 only
  # if p0<3:
  #   _ttimes = [[6,7],[100,101],[180,181]][p0]
  # if p0 in [3,4]:
  #   _ttimes = [6,100,180,7,101,181]
  # if p0==5:
  #   _ttimes = [2,6,30,100,180,1,7,101,181]
  # if p0==6:

  # _ttimes = _traintimes(info)
  _ttimes     = [2,6,30,100,150,180,189,1,7,101,181]
  _testtime   = [5,8,99,102,179,182]
  _testtime02 = [0,12,25,40,65,88,167]

  train_data_files = [(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/"        + info.rawname.format(time=n),
                       f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}_GT/TRA/" + info.man_track.format(time=n),)
          for n in _ttimes]
  _specialize_isbi(P,myname,info)

  # P.zoom = (1,1,1) if p0 in [0,1,2,3] else (1,0.5,0.5) # v06

  def match(yt_pts,pts):
    return match_unambiguous_nearestNeib(yt_pts,pts,dub=100,scale=[3,1,1])

  ## v03 ONLY
  # kernel_size = 256**(p1/49) / 4
  # P.kern = np.array(kernel_size)*[1,1]
  # P.nms_footprint = [5,5] #np.ones(().astype(np.int))
  ## v04 ONLY
  # kernel_size = 4.23 ##
  # P.kern = np.array(kernel_size)*[1,1]
  # P.nms_footprint = [5,5] #np.ones(().astype(np.int))
  # noise_level = (p1/49)*20 #if p0==0 else (p1/49)*20

  savedir_local = savedir / f'e21_isbidet/v07/pid{pid:03d}/'


  class CenterpointGen(object):
    def __init__(self, filenames):
      data   = np.array([SimpleNamespace(raw=zoom(load(r),P.zoom,order=1),lab=zoom(load(l),P.zoom,order=0)) for r,l in filenames],dtype=np.object)
      ndim   = data[0].raw.ndim
      for d in data: d.pts = mantrack2pts(d.lab)
      ## v04 ONLY. simulate noisy annotations.
      # for d in data:
      #   # x = ((np.random.rand(*d.pts.shape) - 0.5)*noise_level*2).astype(np.int) 
      #   # ipdb.set_trace()
      #   # _sh = 10000,2
      #   ## floors towards zero
      #   _sh = d.pts.shape
      #   x = np.random.randn(*_sh)
      #   x = x / np.linalg.norm(x,axis=1)[:,None]
      #   r = np.random.rand(_sh[0])**(1/_sh[1]) * noise_level
      #   x = (x*r[:,None]).astype(np.int)
      #   d.pts += x
      for d in data: d.target = place_gaussian_at_pts(d.pts,d.lab.shape,P.kern)
      for d in data: d.raw = normalize3(d.raw,2,99.4,clip=False)
      self.data = data

      ## train/vali for iterative sampling
      slicelist = shape2slicelist(data[0].raw.shape, P.patch, divisible=(1,8,8))
      slicelist = [(i,ss) for ss in slicelist for i in range(len(data))]
      # slicelist = [ss for ss in slicelist if (self.get_patch(ss).yt>0.5).sum()>0]
      # dist = [self.get_patch(ss).yt.sum() for ss in slicelist] #, [1,10,20,30,40,50,60,70,80,90,99]
      # print(dist)
      np.random.seed(0)
      np.random.shuffle(slicelist)
      N = len(slicelist)
      print("N = ", N)
      Nvali  = ceil(N/8)
      Ntrain = N-Nvali
      self.train_slices = slicelist[:Ntrain]
      self.vali_slices  = slicelist[Ntrain:]

      ## an index into data must be (i∈len(data),ss∈data[0].shape)
      ## we can make a list of non-overlapping slices, filter out slices based on data content?, split slices into train/vali, all without _actually_ shuffling patches or forgetting their orig spacetime location!


    def get_patch(self,idx):
      i,ss = idx
      s = SimpleNamespace()
      s.x = self.data[i].raw[ss].copy()
      s.yt = self.data[i].target[ss].copy()
      return s

    def sample_content(self,train_mode=True):
      N = len(self.data)
      # Nvali  = [1,1,1,3,3,4,4][p0] # ceil(N/8) v06
      Nvali  = 4 # v07
      Ntrain = N-Nvali
      idxs = np.r_[:Ntrain] if train_mode else np.r_[Ntrain:N]      
      x,yt = sample_content(self.data[idxs],P.patch)
      return x,yt

    def sample_iterate(self,time,train_mode=True):
      if train_mode:
        N   = len(self.train_slices)
        _t  = time % N
        idx = self.train_slices[_t]
        s   = self.get_patch(idx)
        if _t == N-1: np.random.shuffle(self.train_slices)
      else:
        N   = len(self.vali_slices)
        _t  = np.random.choice(range(N))
        idx = self.vali_slices[_t]
        s   = self.get_patch(idx)
      return s


    def sample(self,time,train_mode=True):
      if p0==0:
        s = self.sample_iterate(time,train_mode)
        x,yt = s.x, s.yt
      if p0==1:
        x,yt = self.sample_content(train_mode)

      if train_mode:
        x,yt = augment(x,yt)
      if myname=='fly_isbi':
        w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=0.0)
      else:
        w = weights(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=1/5)
        # w = np.ones_like(yt)
      s = SimpleNamespace(x=x,yt=yt,w=w)
      ## prevent overdetect on peaks EVEN WITH FOOTPRINT because they have exactly the same value
      # s.yt_pts = peak_local_max(yt+np.random.rand(*yt.shape)*1e-5,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
      s.yt_pts = peak_local_max(yt,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
      return s

    def sampleMax(self,t):
      s = self.sample(t)
      if s.x.ndim==3:
        s.x = s.x.max(0)
        s.yt = s.yt.max(0)
        s.w = s.w.max(0)
      return s

  def pred_many(net,times,dirname='pred',savedir=None):
    gtpts = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
    dims = "ZYX" if info.ndim==3 else "YX"

    def _single(i):
      "i is time"
      print(i)
      name = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=i)
      raw = load(name)
      raw = normalize3(raw,2,99.4,clip=False)
      x = zoom(raw,P.zoom,order=1)
      res = torch_models.predict_raw(net,x,dims=dims).astype(np.float32)
      height = res.max()
      res = res / res.max() ## 
      pts = peak_local_max(res,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
      pts = pts/P.zoom
      # matching = point_matcher.match_unambiguous_nearestNeib(gtpts[i],pts,dub=10,scale=[3,1,1])
      matching = match(gtpts[i],pts)
      print(matching.f1)
      res = zoom(res, 1/np.array(P.zoom), order=1)
      # fp, fn = find_errors(raw,matching)
      # if savedir: save(pts,savedir / "predpts/pts{i:04d}.pkl")
      target = place_gaussian_at_pts(gtpts[i],raw.shape,P.kern)
      mse = np.mean((res-target)**2)
      scores = dict(f1=matching.f1,precision=matching.precision,recall=matching.recall,mse=mse)
      return SimpleNamespace(**locals())

    def _save_preds(d,i):
      # _best_f1_score = matching.f1
      save(d.raw.astype(np.float16).max(0), savedir/f"{dirname}/d{i:04d}/raw.tif")
      save(d.res.astype(np.float16).max(0), savedir/f"{dirname}/d{i:04d}/pred.tif")
      save(d.target.astype(np.float16).max(0), savedir/f"{dirname}/d{i:04d}/target.tif")
      save(d.pts, savedir/f"{dirname}/d{i:04d}/pts.pkl")
      save(gtpts[i], savedir/f"{dirname}/d{i:04d}/pts_gt.pkl")
      save(d.scores, savedir/f"{dirname}/d{i:04d}/scores.pkl")
      save(d.matching, savedir/f"{dirname}/d{i:04d}/matching.pkl")
      # save(fp, savedir/f"errors/t{i:04d}/fp.pkl")
      # save(fn, savedir/f"errors/t{i:04d}/fn.pkl")

    def _f(i): ## i is time
      d = _single(i)
      if i in _testtime02: _save_preds(d,i)
      return d.scores,d.pts,d.height

    scores,ltps,height = map(list,zip(*[_f(i) for i in times]))
    save(scores, savedir/f"{dirname}/scores.pkl")
    save(height, savedir/f"{dirname}/height.pkl")
    save(ltps, savedir/f"{dirname}/ltps.pkl")

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
    score = match(s.yt_pts,pts)
    return score.f1

  def _config():
    cfig = SimpleNamespace()
    cfig.getnet = P.getnet
    cfig.time_validate = 100
    ## 30_000 for C.Elegans ONLY v07/v06
    cfig.time_total = 30_000 if info.ndim==3 else 15_000 ## # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr on 3D data (10x faster on 2D?)
    cfig.n_vali_samples = 10
    cfig.save_every_n = 5
    cfig.lr = 4e-4
    cfig.savedir = savedir_local
    cfig.loss = _loss
    cfig.vali_metrics = [height, point_match]
    cfig.vali_minmax  = [None,np.max]
    return cfig

  # def optimize_plm(net,data):
  #   def params2score(params):
  #     def f(x):
  #       (raw,gt) = x
  #       # pred = 
  #     avg_score = np.mean([f(x) for x in dataset])
  #     return avg_score
  #   params = #optimize(params0,params2score,)
  #   return params


  """
  EVERYTHING BELOW THIS POINT IS LIKE A WORKSPACE THAT CHANGES RAPIDLY & USES FUNCTIONS DEFINED ABOVE.
  """

  cfig = _config()

  print("Running e21 with savedir: \n", cfig.savedir, flush=True)

  if False:
    dg = CenterpointGen(train_data_files); 
    cfig.sample = dg.sample
    # save(dg.data[0].target.astype(np.float32), cfig.savedir / 'target_t_120.tif')  

    T = detector2.train_init(cfig)
    save([dg.sampleMax(i) for i in range(10)],cfig.savedir/"traindata.pkl")

    # T = detector2.train_continue(cfig,cfig.savedir / 'm/best_weights_loss.pt')
    detector2.train(T)

  net = cfig.getnet().cuda()
  net.load_state_dict(torch.load(cfig.savedir / "m/best_weights_loss.pt"))

  # s0  = set(_traintimes(info)) #list()
  s0  = set(_ttimes) # v05, v06
  s1  = set(range(info.start,info.stop)) # - s0 # v05,v06
  # _t1 = [list(s1)[0]]
  _t1 = list(s1)
  # _t1 = [18]
  # _t1 = _testtime
  _t1 = _testtime02
  # _t1 = [0,1,2,16,]
  # _t0 = list(np.random.choice(list(s0),min(5,len(s0)),replace=False))
  # _t1 = list(np.random.choice(list(s1),min(5,len(s1)),replace=False))

  ltps = pred_many(net,_t1,dirname='pred_all_02',savedir=cfig.savedir)
  # ltps = pred_many(net,_t0,dirname='pred_train',savedir=cfig.savedir)

  # ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  # tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  # scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()

