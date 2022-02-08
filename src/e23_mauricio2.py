
## standard lib
from types import SimpleNamespace
from pathlib import Path
import shutil
import pickle
from time import time
from textwrap import dedent

## standard scipy
import torch
import numpy as np
from scipy.ndimage import zoom
from skimage.feature  import peak_local_max
from matplotlib import pyplot as plt

## 3rd party 
import augmend
from augmend import Augmend, FlipRot90, Elastic, Rotate, Scale, IntensityScaleShift, Identity
from pykdtree.kdtree import KDTree as pyKDTree

## segtools
from segtools.point_tools import patches_from_centerpoints
from segtools import point_matcher
from segtools import torch_models
from segtools.numpy_utils import normalize3
from segtools.ns2dir import load,save ## FIXME

## local
from datagen import place_gaussian_at_pts
from e26_utils import img2png


# savedir = savedir_global()
# print("savedir:", savedir)

# _gpu  = "-p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 "
# _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
# slurm = 'sbatch -J e23_{pid:03d} {_resources} -o slurm/e23_pid{pid:03d}.out -e slurm/e23_pid{pid:03d}.err --wrap \'python3 -c \"import e23_mauricio_copy as ex; ex.slurm_entry({pid})\"\' '
# slurm = slurm.replace("{_resources}",_gpu)

# def run_slurm(pids):
#   ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
#   ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.
#   shutil.copy("/projects/project-broaddus/devseg_2/src/e23_mauricio.py", "/projects/project-broaddus/devseg_2/src/e23_mauricio_copy.py")
#   for pid in pids: Popen(slurm.format(pid=pid),shell=True)

# def slurm_entry(pid=0):
#   run(0)


## RUN ON SLURM
## sbatch -J e23-mau -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/e23-mau.out -e slurm_err/e23-mau.out --wrap '/bin/time -v e23_mauricio.py'

savedir = Path("/projects/project-broaddus/devseg_2/expr/e23_mauricio/v02/")

def wipedir(path):
  path = Path(path)
  if path.exists(): shutil.rmtree(path)
  path.mkdir(parents=True, exist_ok=True)

def data():

  D = SimpleNamespace()
  D.patch = (8,64,64)
  D.zoom  = (1,1,1)
  D.kern  = [2,5,5]
  D.patch = (16,128,128)
  D.nms_footprint = [3,9,9]
  D.ndim  = 3

  n_raw   = "/projects/project-broaddus/rawdata/ZFishMau2021/coleman/2021_01_21_localphototoxicity_h2brfp_lap2bgfp_G2_Subset_Average_DualSideFusion_max_Subset_forcoleman_T{time}.tif"
  n_pts   = "/projects/project-broaddus/rawdata/ZFishMau2021/anno/t{time:03d}.pkl"
  n_class = "/projects/project-broaddus/rawdata/ZFishMau2021/anno/class{time}.pkl"

  ## no overlap. no border. same for raw and target. variable size. no divisibility constraints (enforce at site of net application).
  def shape2slicelist(imgshape,):
    # divisible=(1,4,4)
    ## use `ceil` so patches have a maximum size of `D.patch`
    ns = np.ceil(np.array(imgshape) / D.patch).astype(int)
    ## list of slice starting coordinates
    start = (np.indices(ns).T * D.patch).reshape([-1,D.ndim])
    # pad = [(0,0),(0,0),(0,0)]

    ## make sure slice end is inside shape
    def _f(st,i): 
      low  = st[i]
      high = min(st[i]+D.patch[i],imgshape[i])
      # high = low + floor((high-low)/divisible[i])*divisible[i] ## divisibility constraints
      return slice(low, high)

    ss = [tuple(_f(st,i) for i in range(D.ndim)) for st in start]
    return ss

  def f(i):
    raw = load(n_raw.format(time=i)).transpose([1,0,2,3])
    raw = zoom(raw,(1,) + D.zoom,order=1)
    raw = normalize3(raw,2,99.4,axs=(1,2,3),clip=False)
    pts = load(n_pts.format(time=i))
    classes = load(n_class.format(time=i))
    pts = [p for i,p in enumerate(pts) if classes[i] in ['p','pm']]
    pts = (np.array(pts) * D.zoom).astype(np.int)
    target = place_gaussian_at_pts(pts,raw.shape[1:],D.kern)
    slices = shape2slicelist(raw.shape[1:])
    s_raw    = [raw[1][ss].copy() for ss in slices]
    s_target = [target[ss].copy() for ss in slices]
    tmax = [target[s].max() for s in slices]
    return SimpleNamespace(pts=pts,raw=s_raw,target=s_target,slices=slices,tmax=tmax,time=i)
    # hi,low = partition(lambda s: target[s].max()>0.99, slices)
    # return SimpleNamespace(raw=raw,pts=pts,target=target,hi=hi,low=low)

  return pickle.load(open(str(savedir / 'data/filtered.pkl'), 'rb'))

  D.samples = []
  D.pts = []
  for i in [0,109]:
    dat = f(i)
    D.pts.append(dat.pts)
    for j in range(len(dat.slices)):
      if dat.tmax[j]==0.0: continue ## FILTER
      D.samples.append(SimpleNamespace(raw=dat.raw[j],target=dat.target[j],tmax=dat.tmax[j],time=dat.time))
  D.samples = np.array(D.samples, dtype=object)

  save(D, savedir / f'data/filtered.pkl') ## remove tmax==0
  # save(D, savedir / f'data/filtered-.5.pkl') ## remove tmax<0.5

  ## save train/vali/test data
  wipedir(savedir/"data/filtered-png/")
  for i in range(len(D.samples)):
    s = D.samples[i]
    # l = D.labels[i]
    r = img2png(s.raw)
    t = img2png(s.target, colors=plt.cm.magma)
    composite = r//2 + t//2 
    save(composite, savedir/f'data/filtered-png/t{s.time}/d{i:04d}.png')

  return D

def train(dataset=None,continue_training=False):

  CONTINUE = continue_training
  print("CONTINUE ? : ", bool(CONTINUE))

  print(f"""
    Begin training Mauricio / Retina / Mitosis detection (v01)
    Savedir is {savedir / "train"}
    """)

  D = dataset

  # if D is None:
  #   D = pickle.load(open(str(savedir / "data/filtered.pkl"), 'rb'))

  ## validation params
  P = SimpleNamespace()
  P.nms_footprint = [3,9,9]
  P.border = [0,0,0]
  P.match_dub = 10
  P.match_scale = [5,1,1]

  ## loss and network
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2),   kernsize=(3,5,5),   finallayer=torch_models.nn.Sequential).cuda()
  net = net.to(device)

  ## FIXME
  # net.load_state_dict(torch.load(savedir / f'train/m/best_weights_latest.pt'))
  
  if CONTINUE:
    labels = load(savedir / "train/labels.pkl")
    net.load_state_dict(torch.load(savedir / f'train/m/best_weights_latest.pt'))
    history = load(savedir / 'train/history.pkl')
  else:
    N = len(D.samples)
    a,b = N*5//8,N*7//8  ## MYPARAM train / vali / test fractions
    labels = np.zeros(N,dtype=np.uint8)
    labels[a:b]=1; labels[b:]=2 ## 0=train 1=vali 2=test
    np.random.shuffle(labels)
    save(labels, savedir / "train/labels.pkl")
    history = SimpleNamespace(lossmeans=[],valimeans=[],)
    wipedir(savedir/'train/m')
    wipedir(savedir/"train/glance_output_train/")
    wipedir(savedir/"train/glance_output_vali/")

  ## post-load configuration
  assert len(D.samples)>8
  assert len(labels)==len(D.samples)
  opt = torch.optim.Adam(net.parameters(), lr = 1e-4)
  D.labels = labels

  ## aug acts like a function i.e. `aug(raw,target,weights)`
  def build_augmend(ndim):
    aug = Augmend()
    ax = {2:(0,1), 3:(1,2)}[ndim]
    if ndim==3:
      aug.add([FlipRot90(axis=0), FlipRot90(axis=0), FlipRot90(axis=0),], probability=1)
      aug.add([FlipRot90(axis=(1,2)), FlipRot90(axis=(1,2)), FlipRot90(axis=(1,2))], probability=1)
    else:
      aug.add([FlipRot90(axis=(0,1)), FlipRot90(axis=(0,1)), FlipRot90(axis=(0,1)),], probability=1)

    aug.add([IntensityScaleShift(), Identity(), Identity()], probability=1)

    # aug.add([Rotate(axis=ax, order=1),
    #          Rotate(axis=ax, order=1),],
    #         probability=0.5)
    return aug

  f_aug = build_augmend(D.ndim)

  from skimage.morphology.binary import binary_dilation

  def addweights(D):
    for d in D.samples:
      d.weights = binary_dilation(d.target>0 , np.ones((1,7,7)))
      # d.weights = (d.target > 0)
      # print("{:.3f}".format(d.weights.mean()),end="  ")
  addweights(D)

  # if P.sparse:
  #   # w0 = dgen.weights__decaying_bg_multiplier(s.target,0,thresh=np.exp(-0.5*(3)**2),decayTime=None,bg_weight_multiplier=0.0)
  #   # NOTE: i think this is equivalent to a simple threshold mask @ 3xstddev, i.e.
  #   w0 = (s.target > np.exp(-0.5*(3**2))).astype(np.float32)
  # else:
  #   w0 = np.ones(s.target.shape,dtype=np.float32)
  # return w0
  # df['weights'] = df.apply(addweights,axis=1)

  def mse_loss(x,yt,w):

    x  = torch.from_numpy(x.copy() ).float().to(device, non_blocking=True)
    yt = torch.from_numpy(yt.copy()).float().to(device, non_blocking=True)
    w  = torch.from_numpy(w.copy() ).float().to(device, non_blocking=True)

    y  = net(x[None,None])[0,0]

    ## Introduce ss.b masking to ensure that backproped pixels do not overlap between train/vali/test
    
    loss = torch.abs((w*(y-yt)**2)).mean()
    return y,loss

  # trainset = df[(df.labels==0) & (df.npts>0)] ## MYPARAM subsample trainset ?
  trainset = D.samples[D.labels==0]
  validata = D.samples[D.labels==1]
  N_train  = len(trainset)
  N_vali = len(validata)

  def backprop_n_samples_into_net():
    _losses = []
    idxs = np.arange(N_train)
    np.random.shuffle(idxs)
    tic = time()
    for i in range(N_train):
      # s  = trainset.iloc[idxs[i]]
      s  = trainset[idxs[i]]
      x  = s.raw.copy()
      yt = s.target.copy()
      w  = s.weights.copy()

      ## remove the border regions that make our patches a bad size
      divis = (1,8,8)
      ss = [[None,None,None],[None,None,None],[None,None,None],]
      for n in range(D.ndim):
        rem = x.shape[n]%divis[n]
        if rem != 0:
          # print(f"\nn,rem = {n},{rem}\n")
          ss[n][0] = 0
          ss[n][1]  = -rem
      ss = tuple([slice(a,b,c) for a,b,c in ss])
      x  = x[ss]
      yt = yt[ss]
      w  = w[ss]

      x,yt,w = f_aug([x,yt,w])
      y,l = mse_loss(x,yt,w)
      l.backward()
      opt.step()
      opt.zero_grad()
      _losses.append(float(l.detach().cpu()))
      dt = time()-tic; tic = time()
      print(f"it {i}/{N_train}, dt {dt:5f}, max {float(y.max()):5f}", end='\r',flush=True)

    history.lossmeans.append(np.nanmean(_losses))

  def validate_single(sample):
    s = sample

    x  = s.raw.copy()
    yt = s.target.copy()
    w  = s.weights.copy()

    ## remove the border regions that make our patches a bad size
    divis = (1,8,8)
    ss = [[None,None,None],[None,None,None],[None,None,None],]
    for n in range(D.ndim):
      rem = x.shape[n]%divis[n]
      if rem != 0:
        # print(f"\nn,rem = {n},{rem}\n")
        ss[n][0] = 0
        ss[n][1]  = -rem
    ss = tuple([slice(a,b,c) for a,b,c in ss])
    x  = x[ss]
    yt = yt[ss]
    w  = w[ss]

    # if np.any(np.array(x.shape)%(1,8,8) != (0,0,0)): 
    #   return SimpleNamespace(pred=None,scores=(0,0,0))

    with torch.no_grad(): y,l = mse_loss(x,yt,w)

    y = y.cpu().numpy()
    l = float(l.cpu().numpy())
    
    _peaks = y #.copy() #y/y.max()
    pts      = peak_local_max(_peaks,threshold_abs=.5,exclude_border=False,footprint=np.ones(P.nms_footprint))
    s_pts    = peak_local_max(s.target.astype(np.float32),threshold_abs=.5,exclude_border=False,footprint=np.ones(P.nms_footprint))

    ## filter border points
    patch  = np.array(s.raw.shape)
    pts2   = [p for p in pts if np.all(p%(patch-P.border) > P.border)]
    s_pts2 = [p for p in s_pts if np.all(p%(patch-P.border) > P.border)]

    matching = point_matcher.match_unambiguous_nearestNeib(s_pts2,pts2,dub=P.match_dub,scale=P.match_scale)
    return SimpleNamespace(x=x,yt=yt,pred=y, scores=(l,matching.f1,y.max()))

  def validate_many():
    _valiscores = []

    # idxs = np.arange(N_vali)
    # np.random.shuffle(idxs)

    for i in range(N_vali):
      s = validata[i] ## no idxs
      _scores = validate_single(s).scores
      _valiscores.append(_scores)
      # if i%10==0: print(f"_scores",_scores, end='\n',flush=True)

    history.valimeans.append(np.nanmean(_valiscores,0))

    ## now save (new) best weights

    torch.save(net.state_dict(), savedir / f'train/m/best_weights_latest.pt')

    valikeys   = ['loss','f1','height']
    valiinvert = [1,-1,-1] # minimize, maximize, maximize
    valis = np.array(history.valimeans).reshape([-1,3])*valiinvert

    for i,k in enumerate(valikeys):
      if np.nanmin(valis[:,i])==valis[-1,i]:
        torch.save(net.state_dict(), savedir / f'train/m/best_weights_{k}.pt')

  def pred_glances(time):
    ids = [0,N_train//2,N_train-1]
    for i in ids:
      res = validate_single(trainset[i])
      pred = res.pred
      r = img2png(res.x)
      p = img2png(pred,colors=plt.cm.magma)
      t = img2png((res.yt > 0.9).astype(np.uint8))
      composite = np.round(r/2 + p/2).astype(np.uint8).clip(min=0,max=255)
      m = np.any(t[:,:,:3]!=0 , axis=2)
      composite[m] = t[m]
      save(composite,savedir/f'train/glance_output_train/a{time:03d}_{i:03d}.png')

    # ids = [0,N_vali//2,N_vali-1]
    # ids = [1,N_vali//2-1,N_vali-1-1]
    ids = range(N_vali)
    for i in ids:
      res = validate_single(validata[i])
      pred = res.pred
      r = img2png(res.x)
      p = img2png(pred,colors=plt.cm.magma)
      t = img2png((res.yt > 0.9).astype(np.uint8))
      composite = np.round(r/2 + p/2).astype(np.uint8).clip(min=0,max=255)
      m = np.any(t[:,:,:3]!=0 , axis=2)
      composite[m] = t[m]
      save(composite,savedir/f'train/glance_output_vali/a{time:03d}_{i:03d}.png')

  tic = time()
  n_pix = np.sum([np.prod(d.raw.shape) for d in trainset]) / 1_000_000 ## Megapixels of raw data in trainset
  N_epochs=300 ## MYPARAM
  print(f"Estimated Time: {n_pix} Mpix * 1s/Mpix = {300*n_pix/60:.2f}m = {300*n_pix/60/60:.2f}h \n")
  print(f"\nBegin training for {N_epochs} epochs...\n\n")
  for ep in range(N_epochs):
    backprop_n_samples_into_net()
    validate_many()
    save(history, savedir / "train/history.pkl")
    if ep in range(10) or ep%10==0: pred_glances(ep)
    
    dt  = time() - tic
    tic = time()

    print("\033[F",end='') ## move cursor UP one line 
    print(f"finished epoch {ep+1}/{N_epochs}, loss={history.lossmeans[-1]:4f}, dt={dt:4f}, rate={n_pix/dt:5f} Mpix/s", end='\n',flush=True)

def pred():
  # gtpts = load(f"/projects/project-broaddus/rawdata/{info.myname}/traj/{info.isbiname}/{info.dataset}_traj.pkl")
  # gtpts = [d.pts for d in data]
  # dims = "ZYX" if data[0].raw.ndim==3 else "YX"
  # dims = "ZYX"

  wipedir(savedir / "pred")

  n_raw   = "/projects/project-broaddus/rawdata/ZFishMau2021/coleman/2021_01_21_localphototoxicity_h2brfp_lap2bgfp_G2_Subset_Average_DualSideFusion_max_Subset_forcoleman_T{time}.tif"
  n_pts   = "/projects/project-broaddus/rawdata/ZFishMau2021/anno/t{time:03d}.pkl"
  n_class = "/projects/project-broaddus/rawdata/ZFishMau2021/anno/class{time}.pkl"

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  net = torch_models.Unet3(16, [[1],[1]], pool=(1,2,2),   kernsize=(3,5,5),   finallayer=torch_models.nn.Sequential).cuda()
  net = net.to(device)
  net.load_state_dict(torch.load(savedir / f'train/m/best_weights_latest.pt'))

  D = SimpleNamespace()
  D.patch = (8,64,64)
  D.zoom  = (1,1,1)
  D.kern  = [2,5,5]
  D.patch = (16,128,128)
  D.nms_footprint = [3,9,9]
  D.ndim  = 3

  def predsingle(time):
    raw = load(n_raw.format(time=i)).transpose([1,0,2,3])[1]
    raw = zoom(raw,D.zoom,order=1)
    raw = normalize3(raw,2,99.4,clip=False)
    # raw = zoom(raw,(1,) + D.zoom,order=1)
    # raw = normalize3(raw,2,99.4,axs=(1,2,3),clip=False)
    gtpts = load(n_pts.format(time=i))
    classes = load(n_class.format(time=i))
    gtpts = [p for i,p in enumerate(gtpts) if classes[i] in ['p','pm']]
    gtpts = (np.array(gtpts) * D.zoom).astype(np.int)
    pred = torch_models.predict_raw(net,raw,dims="ZYX",D_zyx=(24,256,256)).astype(np.float32)
    height = pred.max()
    pred = pred / pred.max() ## 
    pts = peak_local_max(pred,threshold_abs=.2,exclude_border=False,footprint=np.ones(D.nms_footprint))
    pts = pts/D.zoom
    matching = point_matcher.match_unambiguous_nearestNeib(gtpts,pts,dub=100,scale=[3,1,1])
    print(dedent(f"""
           time : {time:03d}
             f1 : {matching.f1:.3f}
      precision : {matching.precision:.3f}
         recall : {matching.recall:.3f}
      """))
    return SimpleNamespace(**locals())

  # def _single(i):
  #   "i is time"
  #   print(i)
  #   name = f"/projects/project-broaddus/rawdata/ZFishMau2021/coleman/2021_01_21_localphototoxicity_h2brfp_lap2bgfp_G2_Subset_Average_DualSideFusion_max_Subset_forcoleman_T{i}.tif"
  #   gtpts = load(f"/projects/project-broaddus/rawdata/ZFishMau2021/anno/t{i:03d}.pkl").astype(np.int)
  #   raw = load(name).transpose([1,0,2,3]) #[:,1]
  #   raw = normalize3(raw,2,99.4,axs=(1,2,3),clip=False)
  #   x   = zoom(raw,(1,)+CPNet.zoom,order=1)
  #   pred = torch_models.predict_raw(CPNet.net,x,dims=dims,D_zyx=(24,256,256)).astype(np.float32)[0]
    
  #   height = pred.max()
  #   pred = pred / pred.max() ## 
  #   pts = peak_local_max(pred,threshold_abs=.2,exclude_border=False,footprint=np.ones(CPNet.nms_footprint))
  #   pts = pts/CPNet.zoom
  #   matching = point_matcher.match_unambiguous_nearestNeib(gtpts,pts,dub=100,scale=[3,1,1])
  #   # matching = match(gtpts,pts)
  #   print(matching.f1)
  #   pred = zoom(pred, 1/np.array(CPNet.zoom), order=1)
  #   # fp, fn = find_errors(raw,matching)
  #   # if savedir: save(pts,savedir / "predpts/pts{i:04d}.pkl")
  #   target = datagen.place_gaussian_at_pts(gtpts,raw.shape[1:],CPNet.kern)
  #   mse = np.mean((pred-target)**2)
  #   scores = dict(f1=matching.f1,precision=matching.precision,recall=matching.recall,mse=mse)
  #   cropped_yp = patches_from_centerpoints(raw[1],pts,(5,32,32),bounds='constant')
  #   # cropped_yp = np.array([x.max(0) for x in cropped_yp])
  #   cropped_yp = cropped_yp.max(1)
  #   cropped_gt = patches_from_centerpoints(raw[1],gtpts,(5,32,32),bounds='constant')
  #   # cropped_gt = np.array([x.max(0) for x in cropped_gt])
  #   cropped_gt = cropped_gt.max(1)
  #   return SimpleNamespace(**locals())

  def save_preds(d,i):
    # _best_f1_score = matching.f1
    # save(d.raw.astype(np.float16).max(0), savedir_local/f"t{i:04d}/raw.tif")
    save(img2png(d.pred.max(0)), savedir/f"pred/t{i:04d}.png")
    # save(d.target.astype(np.float16).max(0), savedir_local/f"t{i:04d}/target.tif")
    # save(d.cropped_yp.astype(np.float16), savedir_local/f"t{i:04d}/cropped_yp.tif")
    # save(d.cropped_gt.astype(np.float16), savedir_local/f"t{i:04d}/cropped_gt.tif")
    # save(d.pts, savedir_local/f"t{i:04d}/pts.pkl")
    # save(d.gtpts, savedir_local/f"t{i:04d}/pts_gt.pkl")
    # save(d.scores, savedir_local/f"t{i:04d}/scores.pkl")
    # save(d.matching, savedir_local/f"t{i:04d}/matching.pkl")
    # save(fp, savedir/f"errors/t{i:04d}/fp.pkl")
    # save(fn, savedir/f"errors/t{i:04d}/fn.pkl")


  for i in [0,109]:
    d = predsingle(i)
    save_preds(d,i)

  # def _f(i): ## i is time
  #   d = predsingle(i)
  #   save_preds(d,i)
  # return d.scores,d.pts,d.height

  # scores,ltps,height = map(list,zip(*[_f(i) for i in times]))
  # save(scores, savedir/f"{dirname}/scores.pkl")
  # save(height, savedir/f"{dirname}/height.pkl")
  # save(ltps, savedir/f"{dirname}/ltps.pkl")
  # return ltps

if __name__=="__main__":
  D = data()
  train(D,continue_training=0)
  pred()