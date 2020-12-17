## Deprecated

def e20_trainset(pid=0):
  """
  v01: Construct fixed training datasets for each ISBI example.
  Augmentation / content-based sampling, etc goes here.
  We can optionally reconstruct the training data (or not) before each training run, and we'll have a record of scores for each patch.
  uses same pid scheme as e18_isbidet.
  """

  (p0,p1),pid = _parse_pid(pid,[2,19])

  myname, isbiname  = isbi_datasets[p1]
  trainset = ["01","02"][p0]
  info = get_isbi_info(myname,isbiname,trainset)
  print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str))

  res = SimpleNamespace()
  res.zoom = None

  if info.ndim==2:
    res.patch_shape   = [512,512]
    res.nms  = [7,7]
    res.kern = [7,7]
  else:
    res.patch_shape   = [16,128,128]
    res.nms  = [2,7,7]
    res.kern = [2,7,7]

  # ltps = load(f"/projects/project-broaddus/rawdata/{myname}/traj/{isbiname}/{trainset}_traj.pkl")
  # if type(ltps) is dict: 
  #   ltps = [ltps[k] for k in ltps]

  res.train_times = _traintimes()

  def specialize(res):
    if myname=="celegans_isbi":
      # kernel_sigmas = [1,7,7]
      # bg_weight_multiplier = 0.2
      res.kern = [1,5,5]
      res.zoom = (1,0.5,0.5)
    if myname=="trib_isbi":  
      res.kern = [3,3,3]
    if myname=="MSC":
      a,b = info.shape
      if info.dataset=="01": 
        res.zoom=(1/4,1/4)
      else:
        # res.zoom = (256/a, 392/b) ## almost exactly isotropic but divisible by 8!
        res.zoom = (128/a, 200/b) ## almost exactly isotropic but divisible by 8!
    if isbiname=="DIC-C2DH-HeLa":
      res.kern = [11,11]
      res.zoom = (0.5,0.5)
    if myname=="fly_isbi":
      pass
      # cfig.bg_weight_multiplier=0.0
      # cfig.weight_decay = False
    # if myname=="MDA231":     res.kern = [1,3,3]
    if "A549" in myname:
      res.zoom = (1/4)*3
    if myname=="H157":
      res.zoom = (1/4,)*3
    if myname=="hampster":
      # res.kern = [1,7,7]
      res.zoom = (1,0.5,0.5)
    if isbiname=="Fluo-N3DH-SIM+":
      res.zoom = (1,1/2,1/2)

  names = [(f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}/" + info.rawname.format(time=n),
            f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{trainset}_GT/TRA/" + info.man_track.format(time=n),
            )
          for n in res.train_times]

  config_dgen = SimpleNamespace()
  config_dgen.train_mode = 1
  config_dgen.fg_bg_thresh = np.exp(-16/2)
  config_dgen.bg_weight_multiplier = 1.0 #0.2 #1.0
  config_dgen.time_weightdecay = 1600 # for pixelwise weights
  config_dgen.weight_decay = True
  config_dgen.use_weights  = True

  import datagen
  g = datagen.gen_1(names,config_dgen)

  def _config():
    cfig = SimpleNamespace()
    cfig.getnet = _getnet
    cfig.nms_footprint = data.nms
    cfig.rescale_for_matching = list(info.scale)

    cfig.time_validate = Ntrain 
    cfig.time_total = time_total ## # about 12k / hour? 200/min = 5mins/ 1k = 12k/hr on 3D data
    cfig.lr = 4e-4

    return cfig
  cfig = _config()
  
  # cfig.load_train_and_vali_data = _ltvd
  cfig.savedir = savedir / f'e21_isbidet/v01/pid{pid:03d}/'
  return g,config
