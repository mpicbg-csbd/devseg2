"""
Make the bash scripts required for ISBI CSC/CTC compliance from `template.sh`.
Move the weights appropriate for each model into the `models/` folder.
"""

from glob import glob
import os
import shutil
import pickle
from subprocess import run

isbi_datasets = [
  ("HSC",             "BF-C2DL-HSC"),           #  0     0 
  ("MuSC",            "BF-C2DL-MuSC"),          #  1     1 
  ("HeLa",            "DIC-C2DH-HeLa"),         #  2     2 
  ("MSC",             "Fluo-C2DL-MSC"),         #  3     3 
  ("A549",            "Fluo-C3DH-A549"),        #  4     4 
  ("A549-SIM",        "Fluo-C3DH-A549-SIM"),    #  5    16 
  ("H157",            "Fluo-C3DH-H157"),        #  6     5 
  ("MDA231",          "Fluo-C3DL-MDA231"),      #  7     6 
  ("GOWT1",           "Fluo-N2DH-GOWT1"),       #  8     7 
  ("SIM+",            "Fluo-N2DH-SIM+"),        #  9    17 
  ("HeLa",            "Fluo-N2DL-HeLa"),        # 10     8 
  ("celegans_isbi",   "Fluo-N3DH-CE"),          # 11     9 
  ("hampster",        "Fluo-N3DH-CHO"),         # 12    10 
  ("SIM+",            "Fluo-N3DH-SIM+"),        # 13    18 
  ("fly_isbi",        "Fluo-N3DL-DRO"),         # 14    11 
  ("trib_isbi_proj",  "Fluo-N3DL-TRIC"),        # 15    12 
  ("U373",            "PhC-C2DH-U373"),         # 16    14 
  ("PSC",             "PhC-C2DL-PSC"),          # 17    15 
  # ("trib_isbi", "Fluo-N3DL-TRIF"), 			# 18    13 
  ("trib_isbi/crops_2xDown", "Fluo-N3DL-TRIF"), # 18    13 
  ]


def main():

	for i in range(38):
		weightname_in = f"../expr/e24_isbidet_AOT/v01/pid{i:03d}/m/best_weights_loss.pt"
		isbiname = isbi_datasets[i//2][1]
		myname   = isbi_datasets[i//2][0]
		dataset  = ["01","02"][i%2]

		if dataset=='02': continue ## TODO: FIXME!!!
		
		weightname_out = f"{isbiname}-{dataset}_weights.pt"
		
		# os.remove(f"models/*")
		shutil.copy(weightname_in, "models/"+weightname_out)

		
		template = open("template.sh",'r').read()

		indir  = f"/projects/project-broaddus/rawdata/{myname}/{isbiname}/{dataset}"
		outdir = f"/projects/project-broaddus/rawdata/isbi_challenge_out/{isbiname}/{dataset}_RES"

		params = pickle.load(open(f"/projects/project-broaddus/devseg_2/expr/e24_isbidet_AOT/v01/pid{i:03d}/params.pkl",'rb'))
		Ndim = len(params.zoom)

		temp = {
			"<indir>": indir,
			"<outdir>": outdir,
			"<weightname>":weightname_out,
			"<zoom>": ("{:.3f} "*Ndim).format(*params.zoom), 
			"<nms_footprint>": ("{:3d} "*Ndim).format(*params.nms_footprint),
		}

		for x,y in temp.items():
			template = template.replace(x,y)

		fout = f"test_{isbiname}-01.sh"
		with open(fout,'w') as _fi:
			_fi.write(template)

		run([f"chmod +x {fout}"],shell=True)

if __name__=="__main__":
	main()