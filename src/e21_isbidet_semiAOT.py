


class CPNetISBI2(CenterpointModel):
  def __init__(self,savedir,info):
    self.timings = []
    self.timings.append(time())
    super().__init__(savedir)
    self._init_params(info.ndim)
    self.train_cfig.net = self.net
    self.train_cfig.sample = self.sample
    self.info = info
    self.ndim = info.ndim
    self.aug = self.augmenter()
    cpnet_data_specialization(self,info) ## specialization before dataloader!
    self.timings.append(time())
    self.dataloader()
    self.timings.append(time())

  def augmenter(self):    
    aug = Augmend()
    ax = {2:(0,1), 3:(1,2)}[self.ndim]
    if self.ndim==3:
      aug.add([FlipRot90(axis=0), FlipRot90(axis=0),], probability=1)
      aug.add([FlipRot90(axis=(1,2)), FlipRot90(axis=(1,2)),], probability=1)
    else:
      aug.add([FlipRot90(axis=(0,1)), FlipRot90(axis=(0,1)),], probability=1)
    # aug.add([Elastic(axis=ax, amount=5, order=1),
    #          Elastic(axis=ax, amount=5, order=1),],
    #         probability=0.5)
    aug.add([Rotate(axis=ax, order=1),
             Rotate(axis=ax, order=1),],
            probability=0.5)
    # s = np.random.rand()+0.5
    # aug.add([Scale(axis=ax, amount=(s,s), order=1),
    #          Scale(axis=ax, amount=(s,s), order=1),],
    #         probability=0.5)
    return aug

  def dataloader(self):
    info  = self.info
    n_raw = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}/" + info.rawname
    n_lab = f"/projects/project-broaddus/rawdata/{info.myname}/{info.isbiname}/{info.dataset}_GT/TRA/" + info.man_track

    if info.index in [6,11,12,13,14,15,18]:
      n_raw = n_raw[:-4].replace("rawdata","rawdata/zarr") + ".zarr"
      n_lab = n_lab[:-4].replace("rawdata","rawdata/zarr") + ".zarr"

    def f(i):
      raw = load(n_raw.format(time=i))[...]
      # raw = load(n_raw.format(time=i))
      # raw = zoom(raw,self.zoom,order=1)
      raw = gputools.scale(raw,self.zoom,interpolation='linear')
      p2,p99 = np.percentile(raw[::4,::4,::4],[2,99.4])
      raw = (raw-p2)/(p99-p2)
      # raw = normalize3(raw[::4,::4,::4],2,99.4,clip=False)
      pts = dgen.mantrack2pts(load(n_lab.format(time=i)))
      pts = (np.array(pts) * self.zoom).astype(np.int)
      target = dgen.place_gaussian_at_pts(pts,raw.shape,self.kern)
      slices = dgen.shape2slicelist(raw.shape,self.patch,divisible=(1,8,8)[-self.ndim:])
      return SimpleNamespace(raw=raw,pts=pts,target=target,slices=slices)

      # def rsample():
      #   pt = rchoose(pts)

    _ttimes = _traintimes_cpnet(info)
    self.data_train = [f(i) for i in _ttimes[[0,1]]]
    self.data_vali  = [f(i) for i in _ttimes[[-1,-2]]]

  def data2patch(self,time,train_mode=True):

    data = self.data_train if train_mode else self.data_vali
    d    = rchoose(data)
    pt   = rchoose(d.pts) ## choose point
    ss   = dgen.jitter_center_inbounds(pt,self.patch,d.raw.shape,jitter=0.1)
    
    x  = d.raw[ss].copy()
    yt = d.target[ss].copy()
    s  = SimpleNamespace(x=x,yt=yt,pt=pt)
    return s

  def sample(self,time,train_mode=True):

    s = self.data2patch(time,train_mode=train_mode)
    x,yt = s.x,s.yt

    if train_mode:
      x,yt = self.aug([x,yt])

    x  = x.copy()
    yt = yt.copy()

    # if self.extern.info.myname=='fly_isbi':
    #   w = weights__decaying_bg_multiplier(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=0.0)
    # else:
    w = dgen.weights__decaying_bg_multiplier(yt,time,thresh=np.exp(-4**2/2),decayTime=3*1600,bg_weight_multiplier=1/5)
      # w = np.ones_like(yt)

    s = SimpleNamespace(x=x,yt=yt,w=w)
    ## prevent overdetect on peaks EVEN WITH FOOTPRINT because they have exactly the same value
    # s.yt_pts = peak_local_max(yt+np.random.rand(*yt.shape)*1e-5,threshold_abs=.2,exclude_border=False,footprint=np.ones(P.nms_footprint))
    s.yt_pts = peak_local_max(yt,threshold_abs=.2,exclude_border=False,footprint=np.ones(self.nms_footprint))
    return s
