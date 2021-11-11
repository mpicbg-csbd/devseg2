from pathlib import Path
from subprocess import Popen

import e26_isbidet_dgen as A
import e26_isbidet_train as B

import joblib

from joblib import Memory
location = '/projects/project-broaddus/devseg_2/expr/e26_isbidet/cachedir'
memory = Memory(location, verbose=0)

from experiments_common import parse_pid

def myrun_slurm():
  ## copy the experiments file to a safe name that you WONT EDIT. If you edit the code while jobs are waiting in the SLURM queue it could cause inconsistencies.
  ## NOTE: here we only copy experiment.py file, but THE SAME IS TRUE FOR ALL DEPENDENCIES.

  _gpu  = "-p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 "    ## TODO: more cores?
  _cpu  = "-n 1 -t 1:00:00 -c 4 --mem 128000 "
  slurm = """
  cp *.py temp
  cd temp
  sbatch -J e26_{pid:03d} {_resources} -o ../slurm/e26_pid{pid:03d}.out -e ../slurm/e26_pid{pid:03d}.err --wrap \'python3 -c \"import e26_runner as ex; ex.myrun_slurm_entry({pid})\"\' 
  """
  slurm = slurm.replace("{_resources}",_gpu) ## you can't partially format(), but you can replace().


  for p1 in [0]:
    for p0 in range(19):
      # if p0 in [3,6]: continue
      (p1,p0,),pid = parse_pid([p1,p0],[3,19])
      Popen(slurm.format(pid=pid),shell=True)

      # try:
      #   myrun_slurm_entry(pid)
      #   print("Worked:", [p1,p0])
      # except:
      #   print("missing", pid)


def myrun_slurm_entry(pid=0):

  # ds1,params1,_pngs = A.build_patchFrame(pid)
  # ds2,params2,_pngs = A.build_patchFrame(pid+1)
  # ipdb.set_trace()
  B.train(pid)
  B.evaluate(pid)
  # B.evaluate_imgFrame(pid)

  # evaluate(pid)
  # evaluate_imgFrame(pid,swap=False)
  # evaluate_imgFrame(pid,swap=True)