from experiments_common import *
from pykdtree.kdtree import KDTree as pyKDTree
from segtools.render import rgb_max
from numpy import r_,s_
from cpnet import CenterpointModel, SegmentationModel

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


def specialize_isbi_cpnet(P,info):
  myname = info.myname
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

def add_centerpoint_noise(data):
  """
  v04 only
  """
  for d in data:
    # x = ((np.random.rand(*d.pts.shape) - 0.5)*noise_level*2).astype(np.int) 
    # ipdb.set_trace()
    # _sh = 10000,2
    ## floors towards zero
    _sh = d.pts.shape
    x = np.random.randn(*_sh)
    x = x / np.linalg.norm(x,axis=1)[:,None]
    r = np.random.rand(_sh[0])**(1/_sh[1]) * noise_level
    x = (x*r[:,None]).astype(np.int)
    d.pts += x



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



def get_filenames_cpnet(info):
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
  _testtime   = [5,8,99,102,179,182]
  _testtime02 = [0,12,25,40,65,88,167]
  _ttimes     = [2,6,30,100,150,180,189,1,7,101,181]

  train_data_files = [(f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/"        + info.rawname.format(time=n),
                       f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track.format(time=n),)
          for n in _ttimes]
  return train_data_files

def get_filenames_seg(info):
  # if isbiname == 'Fluo-N3DL-TRIF': _myname = 
  _segnames = sorted(glob(f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/SEG/*.tif"))
  def _f(segname):
    _d = info.ndigits
    _time,_zpos = re.search(r"man_seg_?(\d{3,4})_?(\d{3,4})?\.tif",segname).groups()
    _time = int(_time)
    # if _zpos is not None: _zpos = int(_zpos)
    return f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname.format(time=_time)
  _rawnames = [_f(s) for s in _segnames]
  _filenames_seg = list(zip(_rawnames,_segnames))
  return _filenames_seg

def experiment_specialization(P):
  "P is CPNet"
  P.zoom = (1,1,1) if p0 in [0,1,2,3] else (1,0.5,0.5) # v06
  # v03 ONLY
  kernel_size = 256**(p1/49) / 4
  P.kern = np.array(kernel_size)*[1,1]
  P.nms_footprint = [5,5] #np.ones(().astype(np.int))
  # v04 ONLY
  kernel_size = 4.23 ##
  P.kern = np.array(kernel_size)*[1,1]
  P.nms_footprint = [5,5] #np.ones(().astype(np.int))
  P.extern.noise_level = (p1/49)*20 #if p0==0 else (p1/49)*20


def run(pid=0):
  """
  v01 : refactor of e18. make traindata AOT.
    add `_train` 0 = predict only, 1 = normal init, 2 = continue
    datagen generator -> CenterpointGen, customizable loss and validation metrics, and full detector2 refactor.
  v02 : optimizing hyperparams. in [2,19,5,2] or [2,19,2,5,2] ? 
    p0: dataset in [01,02]
    p1: isbi dataset in [0..18]
    p2: sample flat vs content [0,1] ## ?WAS THIS REALLY HERE?
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
  v08 : segmentation
    p0 : dataset ∈ 0..18
    p1 : acquisition ∈ 0,1
  """

  (p0,p1),pid = parse_pid(pid,[19,2])
  params = SimpleNamespace(p0=p0,p1=p1)

  savedir_local = savedir / f'e21_isbidet/v08/pid{pid:03d}/'
  myname, isbiname = isbi_datasets[p0] # v05
  trainset = ["01","02"][p1] # v06
  info = get_isbi_info(myname,isbiname,trainset)
  params.info = info

  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str), flush=True)

  """
  EVERYTHING BELOW THIS POINT IS LIKE A WORKSPACE THAT CHANGES RAPIDLY & USES FUNCTIONS DEFINED ABOVE.
  """

  print("Running e21 with savedir: \n", savedir_local, flush=True)

  if 0:
    train_data_files = get_filenames_cpnet(info)
    CPNet = CenterpointModel(info.ndim, savedir_local / "cpnet2", params)
    specialize_isbi_cpnet(CPNet,info)
    CPNet.load_data(train_data_files)
    save(CPNet.data[0], CPNet.savedir / 'data0.pkl')  
    save([CPNet.sample(i) for i in range(3)], CPNet.savedir/"traindata_pts.pkl")
    # T = detector2.train_init(CPNet.train_config())
    cfig = CPNet.train_config()
    cfig.time_total = 8_000
    T = detector2.train_continue(cfig,CPNet.savedir / 'm/best_weights_loss.pt')
    detector2.train(T)

  if 0:
    train_data_files = get_filenames_seg(info)
    SEGnet = SegmentationModel(info.ndim,savedir_local / "segment", params)
    SEGnet.load_data(train_data_files)
    # save(dg.data[0].target.astype(np.float32), cfig.savedir / 'target_t_120.tif')  
    T = detector2.train_init(SEGnet.train_config())
    save([SEGnet.sample(i) for i in range(10)],SEGnet.savedir/"traindata_seg.pkl")
    # T = detector2.train_continue(cfig,cfig.savedir / 'm/best_weights_loss.pt')
    detector2.train(T)

  # net = CPNet.getnet().cuda()
  CPNet = CenterpointModel(info.ndim, savedir_local / "cpnet", params)
  CPNet.net.load_state_dict(torch.load(CPNet.savedir / "m/best_weights_loss.pt"))

  SEGnet = SegmentationModel(info.ndim, savedir_local / "segment", params)
  SEGnet.net.load_state_dict(torch.load(SEGnet.savedir / "m/best_weights_loss.pt"))

  N   = 7
  gap = floor((info.stop-info.start)/N)
  predict_times = range(info.start,info.stop,gap)
  savetimes = predict_times
  L    = SimpleNamespace(info=info,CPNet=CPNet,SEGnet=SEGnet,savetimes=savetimes,predict_times=predict_times,)
  ltps = predict_centers(L)

  L.predict_times = range(info.start,info.stop,gap)
  L.savetimes = predict_times
  segs = predict_segments(L)

  # ltps = load(cfig.savedir / f"ltps_{info.dataset}.pkl")
  # tb   = tracking.nn_tracking_on_ltps(ltps,scale=info.scale) # random_tracking_on_ltps(ltps)
  # scores = tracking.eval_tb_isbi(tb,info,savedir=cfig.savedir)
  # ipdb.set_trace()


def _proj(x):
  if x.ndim==3:
    x = x.max(0)
  return x

def predict_centers(L):
  info     = L.info
  savedir  = L.CPNet.savedir / 'pred_all_01'
  trainset = "01"
  pts_gt   = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
  dims     = "ZYX" if info.ndim==3 else "YX"

  def _single(i):
    "i is time"
    print(i)
    name   = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{trainset}/" + info.rawname.format(time=i)
    raw    = load(name)
    S      = L.CPNet.predict_full(raw,dims)

    matching = L.CPNet.match(pts_gt[i],S.pts)
    print(matching.f1)
    # fp, fn = find_errors(raw,matching)
    # if savedir: save(pts,savedir / "predpts/pts{i:04d}.pkl")
    target = place_gaussian_at_pts(pts_gt[i],raw.shape,L.CPNet.kern)
    mse = np.mean((S.pred-target)**2)
    scores = dict(f1=matching.f1,precision=matching.precision,recall=matching.recall,mse=mse)
    return SimpleNamespace(raw=raw,matching=matching,target=target,scores=scores,pts=S.pts,pts_gt=pts_gt[i],pred=S.pred,height=S.height)

  def _save_preds(d,i):
    # _best_f1_score = matching.f1
    save(_proj(d.raw.astype(np.float16)), savedir/f"d{i:04d}/raw.tif")
    save(_proj(d.pred.astype(np.float16)), savedir/f"d{i:04d}/pred.tif")
    save(_proj(d.target.astype(np.float16)), savedir/f"d{i:04d}/target.tif")
    save(d.pts, savedir/f"d{i:04d}/pts.pkl")
    save(d.pts_gt, savedir/f"d{i:04d}/pts_gt.pkl")
    save(d.scores, savedir/f"d{i:04d}/scores.pkl")
    save(d.matching, savedir/f"d{i:04d}/matching.pkl")
    # save(fp, savedir/f"errors/t{i:04d}/fp.pkl")
    # save(fn, savedir/f"errors/t{i:04d}/fn.pkl")

  def _f(i): ## i is time
    d = _single(i)
    if i in L.savetimes: _save_preds(d,i)
    return d.scores,d.pts,d.height

  scores,ltps,height = map(list,zip(*[_f(i) for i in L.predict_times]))
  # save(scores, savedir/f"scores.pkl")
  save(ltps, savedir/f"ltps.pkl")
  # save(height, savedir/f"height.pkl")

  return ltps

def predict_segments(L):
  # gtpts = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
  savedir = L.SEGnet.savedir / "pred_all_01"
  info = L.info
  dims = "ZYX" if info.ndim==3 else "YX"

  def _single(i):
    "i is time"
    print(i)
    name = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname.format(time=i)
    raw  = load(name)
    res  = L.SEGnet.predict_full(raw,dims)

    # lab_gt = load()
    # segscore = 
    return res

  def _save_preds(d,i):
    # _best_f1_score = matching.f1
    save(_proj(d.raw.astype(np.float16)), savedir/f"d{i:04d}/raw.tif")
    save(_proj(d.pred.astype(np.float16)), savedir/f"d{i:04d}/pred.tif")
    save(_proj(d.seg.astype(np.float16)), savedir/f"d{i:04d}/seg.tif")
    # save(d.target.astype(np.float16).max(0), savedir/f"d{i:04d}/target.tif")
    # save(d.pts, savedir/f"d{i:04d}/pts.pkl")
    # save(gtpts[i], savedir/f"d{i:04d}/pts_gt.pkl")
    # save(d.scores, savedir/f"d{i:04d}/scores.pkl")
    # save(d.matching, savedir/f"d{i:04d}/matching.pkl")
    # save(fp, savedir/f"errors/t{i:04d}/fp.pkl")
    # save(fn, savedir/f"errors/t{i:04d}/fn.pkl")

  def _f(i): ## i is time
    d = _single(i)
    if i in L.savetimes: _save_preds(d,i)
    return d.seg
    # return d.scores,d.pts,d.height

  segs = map(list,zip(*[_f(i) for i in L.predict_times]))
  return segs