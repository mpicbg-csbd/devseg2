
## Get all the challenge data from ISBI CTC.
## The training dataset names are identical except "challenge" -> "training"

cd /projects/project-broaddus/rawdata/isbi_challenge/

# Download 2d data from ISBI CTC
# ==============================

# Training dataset: http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip (1.6 GB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-MuSC.zip (1.2 GB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip (37 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip (36 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip (72 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip (53 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip (182 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip (40 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip (124 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip (91 MB)


# (36 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C2DL-Huh7.zip
unzip Fluo-C2DL-Huh7.zip
rm Fluo-C2DL-Huh7.zip

# (1.6 GB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/BF-C2DL-HSC.zip
unzip BF-C2DL-HSC.zip
rm BF-C2DL-HSC.zip

# (1.3 GB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/BF-C2DL-MuSC.zip
unzip BF-C2DL-MuSC.zip
rm BF-C2DL-MuSC.zip

# (41 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/DIC-C2DH-HeLa.zip
unzip DIC-C2DH-HeLa.zip
rm DIC-C2DH-HeLa.zip

# (36 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C2DL-Huh7.zip
unzip Fluo-C2DL-Huh7.zip
rm Fluo-C2DL-Huh7.zip

# (71 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C2DL-MSC.zip
unzip Fluo-C2DL-MSC.zip
rm Fluo-C2DL-MSC.zip

# (46 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-GOWT1.zip
unzip Fluo-N2DH-GOWT1.zip
rm Fluo-N2DH-GOWT1.zip

# (168 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DL-HeLa.zip
unzip Fluo-N2DL-HeLa.zip
rm Fluo-N2DL-HeLa.zip

# (38 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/PhC-C2DH-U373.zip
unzip PhC-C2DH-U373.zip
rm PhC-C2DH-U373.zip

# (106 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/PhC-C2DL-PSC.zip
unzip PhC-C2DL-PSC.zip
rm PhC-C2DL-PSC.zip

# (96 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-SIM+.zip
unzip Fluo-N2DH-SIM+.zip
rm Fluo-N2DH-SIM+.zip



# Download 3D Data from ISBI CTC
# ==============================


# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-C3DH-A549.zip✱ (244 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-C3DH-H157.zip✱ (7.0 GB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-C3DL-MDA231.zip✱ (182 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CE.zip✱ (3.1 GB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CHO.zip✱ (98 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DL-DRO.zip (5.8 GB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DL-TRIC.zip (20.6 GB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DL-TRIF.zip (320 GB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-C3DH-A549-SIM.zip (314 MB)
# Training dataset: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip (3.1 GB)



# (294 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C3DH-A549.zip 
unzip Fluo-C3DH-A549.zip
rm -rf Fluo-C3DH-A549.zip

# (7.1 GB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C3DH-H157.zip 
unzip Fluo-C3DH-H157.zip
rm -rf Fluo-C3DH-H157.zip

# (179 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C3DL-MDA231.zip 
unzip Fluo-C3DL-MDA231.zip
rm -rf Fluo-C3DL-MDA231.zip

# (1.7 GB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DH-CE.zip 
unzip Fluo-N3DH-CE.zip
rm -rf Fluo-N3DH-CE.zip

# (105 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DH-CHO.zip 
unzip Fluo-N3DH-CHO.zip
rm -rf Fluo-N3DH-CHO.zip

# (5.9 GB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DL-DRO.zip 
unzip Fluo-N3DL-DRO.zip
rm -rf Fluo-N3DL-DRO.zip

# (19.9 GB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DL-TRIC.zip 
unzip Fluo-N3DL-TRIC.zip
rm -rf Fluo-N3DL-TRIC.zip

# (467 GB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DL-TRIF.zip 
unzip Fluo-N3DL-TRIF.zip
rm -rf Fluo-N3DL-TRIF.zip

# (327 MB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-C3DH-A549-SIM.zip 
unzip Fluo-C3DH-A549-SIM.zip
rm -rf Fluo-C3DH-A549-SIM.zip

# (5.9 GB)
wget --no-check-certificate http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N3DH-SIM+.zip 
unzip Fluo-N3DH-SIM+.zip
rm -rf Fluo-N3DH-SIM+.zip
