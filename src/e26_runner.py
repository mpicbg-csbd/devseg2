from pathlib import Path

import e26_isbidet_dgen as A
import e26_isbidet_train as B

import joblib

from joblib import Memory
location = '/projects/project-broaddus/devseg_2/expr/e26_isbidet/cachedir'
memory = Memory(location, verbose=0)



def myrun_slurm(pids):
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  shutil.copy(__file__, Path("/projects/project-broaddus/devseg_2/src/temp/") / str(Path(__file__).stem + "_copy.py"))
  shutil.copy(A.__file__, Path("/projects/project-broaddus/devseg_2/src/temp/")) ## becomes a local import 
  shutil.copy(B.__file__, Path("/projects/project-broaddus/devseg_2/src/temp/")) ## becomes a local import 

  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 "    ## TODO: more cores?
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = 'sbatch -J e24_{pid:03d} {_resources} -o slurm/e24_pid{pid:03d}.out -e slurm/e24_pid{pid:03d}.err --wrap \'python3 -c \"import temp.e24_train_on_both_copy as ex; ex.myrun_slurm_entry({pid})\"\' '
  slurm = slurm.replace("{_resources}",_gpu) ## you can't partially format(), but you can replace().
  for pid in pids: Popen(slurm.format(pid=pid),shell=True)


def myrun_slurm_entry(pid=0):

  ds1,params1,_pngs = A.build_patchFrame(pid)
  ds2,params2,_pngs = A.build_patchFrame(pid+1)

  ipdb.set_trace()

  B.train(ds1,params1,continue_training=True)

  # evaluate(pid)
  # evaluate_imgFrame(pid,swap=False)
  # evaluate_imgFrame(pid,swap=True)