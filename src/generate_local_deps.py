## walk over files
## find .py
## add name to list
## add lines containing import to list
## extract module names (gotta run this on source. don't actually import modules!)

import os
from pathlib import Path
import re
import ipdb


# ipdb.set_trace = lambda : None

ignore_folders = ["deprecated","temp",".snakemake"]

def run():
  pyfiles = []
  for _dir,_subdirs,_files in os.walk("."):
    _subdirs[:] = [d for d in _subdirs if Path(d).name not in ignore_folders]

    # ipdb.set_trace()

    for fi in _files:
      if fi[-3:]=='.py':
        pyfiles.append(os.path.join(_dir,fi))

  modulesnames = [Path(fi).stem for fi in pyfiles]
  modulesnames = [m for m in modulesnames if '_copy' not in m]
  ipdb.set_trace()

  listOfImports = []
  for fi in pyfiles:
    listOfImports.extend(analyze_py(fi, modulesnames))

  with open("local_deps.txt",'w') as outfile:
    print(''.join(listOfImports),file=outfile)

def analyze_py(filename, modulesnames):
  lines = open(filename,'r',encoding='utf-8').readlines()
  lines = [l for l in lines if 'import' in l]
  thisname = Path(filename).stem
  lines2 = []
  for currentLine in lines:
    matchedMods = []
    for m in modulesnames:
      pattern = r'\bXXX\b'.replace('XXX',str(m))
      s = re.search(pattern, currentLine)
      ipdb.set_trace()
      if s: matchedMods.append(m)
    if len(matchedMods)>0:
      lines2.append("{:20s}\t{:30s}\t{}".format(thisname, str(matchedMods), currentLine))

  # lines2 = ["{:20s}\t{}".format(thisname, l) for l in lines if any(m in l for m in modulesnames)]
  return lines2



