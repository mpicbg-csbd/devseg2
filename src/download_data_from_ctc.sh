
# The 20th dataset!
mkdir -p /projects/project-broaddus/rawdata/Huh7/
cd /projects/project-broaddus/rawdata/Huh7/
curl -L -O http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-Huh7.zip
unzip Fluo-C2DL-Huh7


## We're going to save challenge data in a special folder:
## `/projects/project-broaddus/rawdata/isbi_challenge/`


## GOWT1 Already exists
mkdir -p /projects/project-broaddus/rawdata/isbi_challenge/
cd /projects/project-broaddus/rawdata/isbi_challenge/
curl -L -O http://data.celltrackingchallenge.net/challenge-datasets/Fluo-N2DH-GOWT1.zip
unzip Fluo-N2DH-GOWT1.zip


