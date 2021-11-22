import os

data = [("BF-C2DL-HSC" , "01") ,
("BF-C2DL-HSC" , "02") ,
("BF-C2DL-MuSC" , "01") ,
("BF-C2DL-MuSC" , "02") ,
("DIC-C2DH-HeLa" , "01") ,
("DIC-C2DH-HeLa" , "02") ,
("Fluo-C2DL-MSC" , "01") ,
("Fluo-C2DL-MSC" , "02") ,
("Fluo-C3DH-A549" , "01") ,
("Fluo-C3DH-A549" , "02") ,
("Fluo-C3DH-A549-SIM" , "01") ,
("Fluo-C3DH-A549-SIM" , "02") ,
("Fluo-C3DH-H157" , "01") ,
("Fluo-C3DH-H157" , "02") ,
("Fluo-C3DL-MDA231" , "01") ,
("Fluo-C3DL-MDA231" , "02") ,
("Fluo-N2DH-GOWT1" , "01") ,
("Fluo-N2DH-GOWT1" , "02") ,
("Fluo-N2DH-SIM+" , "01") ,
("Fluo-N2DH-SIM+" , "02") ,
("Fluo-N2DL-HeLa" , "01") ,
("Fluo-N2DL-HeLa" , "02") ,
("Fluo-N3DH-CE" , "01") ,
("Fluo-N3DH-CE" , "02") ,
("Fluo-N3DH-CHO" , "01") ,
("Fluo-N3DH-CHO" , "02") ,
("Fluo-N3DH-SIM+" , "01") ,
("Fluo-N3DH-SIM+" , "02") ,
("Fluo-N3DL-DRO" , "01") ,
("Fluo-N3DL-DRO" , "02") ,
("Fluo-N3DL-TRIC" , "01") ,
("Fluo-N3DL-TRIC" , "02") ,
("Fluo-N3DL-TRIF" , "01") ,
("Fluo-N3DL-TRIF" , "02") ,
("PhC-C2DH-U373" , "01") ,
("PhC-C2DH-U373" , "02") ,
("PhC-C2DL-PSC" , "01") ,
("PhC-C2DL-PSC" , "02") ,
]

from glob import glob

def testall():
	for isbiname,dataset in data:
		print(isbiname,dataset)
		assert os.path.exists(f"/projects/project-broaddus/rawdata/isbi_challenge_out/{isbiname}/{dataset}_RES/res_track.txt")
		raws  = sorted(glob(f"/projects/project-broaddus/rawdata/isbi_challenge/{isbiname}/{dataset}/t*.tif"))
		masks = sorted(glob(f"/projects/project-broaddus/rawdata/isbi_challenge_out/{isbiname}/{dataset}_RES/mask*.tif"))
		assert len(raws)==len(masks)
		for (r,m) in zip(raws,masks):
			assert r[-7:-4]==m[-7:-4] ## check the actual numbers agree
		print(raws[-1][34:])
		print(masks[-1][34:])











