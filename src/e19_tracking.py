def e19_tracking(pid=0):
  """
  v01: [3,19,2]
  v02: add CP-net tracking to p0. fix a major bug. split dataset loop into p3. fixed, small kern size. [4,19,2,2]
  v03: loops over p0 and p2 internally to allow parallel execution over p1,p3. [19,2]
  v04: WIP using e18_v03 after big bug fix.
  v05: Transition to using e21v01 output.
  v06: e21v02
  """

  (p1,p3),pid = _parse_pid(pid,[19,2])
  for (p0,p2) in iterdims([4,2]):
    if (p0,p2)==(1,1): continue
    if (p0,p2)!=(3,0): continue
    print(f"\n{pid}: {p0} {p1} {p2} {p3}")

    dataset = ['01','02'][p3]
    myname, isbiname = isbi_datasets[p1]
    isbi_dir = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/"
    
    info = get_isbi_info(myname,isbiname,dataset)
    # print(json.dumps(info.__dict__,sort_keys=True, indent=2, default=str))
    print(isbiname, dataset, sep='\t')

    outdir = savedir/f"e19_tracking/v05/pid{pid:03d}/"
    # outdir = savedir/f"e19_tracking/v02/pid_{p0}_{p1:02d}_{p2}_{p3}/"

    _tracking = [lambda ltps: tracking.nn_tracking_on_ltps(ltps,scale=info.scale),
                 lambda ltps: tracking.random_tracking_on_ltps(ltps)
                ][p2]
    kern = np.ones([3,5,5]) if info.ndim==3 else np.ones([5,5])

    start,stop = info.start,info.stop
    if p0==0: ## permute existing labels via tracking
      nap  = tracking.load_isbi2nap(isbi_dir,dataset,[start,stop])
      ltps = tracking.nap2ltps(nap)
      tb   = _tracking(ltps)
      tracking._tb_add_orig_labels(tb,nap)
      lbep = tracking.save_permute_existing(tb,info,savedir=outdir)
    if p0==1: ## make consistent label shape, but don't change label id's.
      nap  = tracking.load_isbi2nap(isbi_dir,dataset,[start,stop])
      tracking.save_isbi(nap,shape=info.shape,_kern=kern,savedir=outdir)
    if p0==2: ## make consistent label shape AND track
      ltps = load(f"/projects/project-broaddus/rawdata/{myname}/traj/{isbiname}/{dataset}_traj.pkl")
      if type(ltps) is dict:
        ltps = [ltps[k] for k in sorted(ltps.keys())]
      tb   = _tracking(ltps)
      nap  = tracking.tb2nap(tb,ltps)
      nap.tracklets[:,1] += start
      tracking.save_isbi(nap,shape=info.shape,_kern=kern,savedir=outdir)
    if p0==3: ## track using CP-net detections
      oldpid  = _parse_pid([p3,p1],[2,19])[1]
      ltps    = load(f"/projects/project-broaddus/devseg_2/expr/e21_isbidet/v01/pid{oldpid:03d}/ltps_{dataset}.pkl")
      if type(ltps) is dict:
        _ltps = [ltps[k] for k in sorted(ltps.keys())]
      tb      = _tracking(_ltps)

      # ipdb.set_trace()
      if info.penalize_FP=='0':
        nap_orig  = tracking.load_isbi2nap(isbi_dir,dataset,[start,start+1])
        tb = tracking.filter_starting_tracks(tb,ltps,nap_orig)
        tracking.relabel_tracks_from_1(tb)
      
      nap  = tracking.tb2nap(tb,_ltps)
      nap.tracklets[:,1] += start
      tracking.save_isbi(nap,shape=info.shape,_kern=kern,savedir=outdir)

    resdir  = Path(isbi_dir)/(dataset+"_RES")
    bashcmd = f"""
    localtra=/projects/project-broaddus/comparison_methods/EvaluationSoftware2/Linux/TRAMeasure
    localdet=/projects/project-broaddus/comparison_methods/EvaluationSoftware2/Linux/DETMeasure
    mkdir -p {resdir}
    # rm {resdir}/*
    cp -r {outdir}/*.tif {outdir}/res_track.txt {resdir}/
    $localdet {isbi_dir} {dataset} {info.ndigits} {info.penalize_FP} > {outdir}/{dataset}_DET.txt
    cat {outdir}/{dataset}_DET.txt
    $localtra {isbi_dir} {dataset} {info.ndigits} > {outdir}/{dataset}_TRA.txt
    cat {outdir}/{dataset}_TRA.txt
    # rm {resdir}/*
    rm {outdir}/*.tif
    """
    run(bashcmd,shell=True)

    # return tb,nap,ltps