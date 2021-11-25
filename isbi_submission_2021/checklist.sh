# Challenge submission checklist
# ==============================


## Assume we've already done all the training...
## Are the models up to date? Trained on both datasets?
ls -lt /projects/project-broaddus/devseg_2/isbi_submission_2021/models

## Ensure local environment is active and proper modules are loaded...
module load gcc/6.2.0
module load cuda/9.2.148
source my_env3/bin/activate

## Copy best-yet weights+params to local models/ and trainparams/ dirs
bash copy_trained_models.sh

## instantiate ISBI format shell scripts from template
python make_templates.py

## Run local prediction (should take 5m for all data except TRIB, which takes 1h ...)
bash slurmcodes.sh
## or single
sbatch -J Fluo-N2DL-HeLa-01     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N2DL-HeLa-01.txt     -e slurm_err/Fluo-N2DL-HeLa-01.txt      --wrap '/bin/time -v ./Fluo-N2DL-HeLa-01.sh    '
sbatch -J Fluo-N3DH-CHO-01      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DH-CHO-01.txt      -e slurm_err/Fluo-N3DH-CHO-01.txt       --wrap '/bin/time -v ./Fluo-N3DH-CHO-01.sh     '
## Check that all jobs succeeded without errors
grep "Exit status: 1" slurm_err/*
ls -lt slurm_err/ ## check date
grep "obj" slurm_out/*GOWT*
ls -lt /projects/project-broaddus/rawdata/isbi_challenge_out/*/*/mask00{0,00}.tif | less -S

## check that raw and mask files match up properly
python -c "import check_predictions as A; A.testall();"

## make videos of each tracking job on SLURM. wait to finish... 30 mins?
python movies.py
grep "Error" slurm/*.err
ls -lt slurm/*.err
ls -lt /projects/project-broaddus/rawdata/isbi_challenge_out_extra/vidz/
ls -lt /projects/project-broaddus/rawdata/isbi_challenge_out_extra/*/*/vidz/

## [x] Perform visual inspection of all .mp4s on local machine

## [x] Run diff on prediction function codes. Make sure remote and local are equivalent.

## [x] Make dense and sparse versions of DRO, TRIC, TRIB. copy/edit shell script, etc.
ls -lt /projects/project-broaddus/rawdata/isbi_challenge_out_dense/*/*/mask00{0,00}.tif | less -S
cp Fluo-N3DL-TRIF-01.sh Fluo-N3DL-TRIF-01-dense.sh
cp Fluo-N3DL-TRIF-02.sh Fluo-N3DL-TRIF-02-dense.sh
cp Fluo-N3DL-TRIC-01.sh Fluo-N3DL-TRIC-01-dense.sh
cp Fluo-N3DL-TRIC-02.sh Fluo-N3DL-TRIC-02-dense.sh
cp Fluo-N3DL-DRO-01.sh  Fluo-N3DL-DRO-01-dense.sh
cp Fluo-N3DL-DRO-02.sh  Fluo-N3DL-DRO-02-dense.sh
sed -i 's/mantrack_t0 .\+/mantrack_t0 "None" \\/g' *dense.sh
sed -i 's/isbi_challenge_out/isbi_challenge_out_dense/g' *dense.sh

## this is only slightly useful...
## toggle DENSE ON in movies.py
sbatch -J DRO01 -n 1  -t 4:00:00 -c 1 --mem 128000  -o slurm/DRO01.out -e slurm/DRO01.err   --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(28)"'
sbatch -J DRO02 -n 1  -t 4:00:00 -c 1 --mem 128000  -o slurm/DRO02.out -e slurm/DRO02.err   --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(29)"'
sbatch -J TRIC01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIC01.out -e slurm/TRIC01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(30)"'
sbatch -J TRIC02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIC02.out -e slurm/TRIC02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(31)"'
sbatch -J TRIF01 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIF01.out -e slurm/TRIF01.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(36)"'
sbatch -J TRIF02 -n 1 -t 4:00:00 -c 1 --mem 128000  -o slurm/TRIF02.out -e slurm/TRIF02.err --wrap '/bin/time -v python3 -c "import movies as A; A.make_movie_pid(37)"'
## toggle DENSE OFF in movies.py


## Package up code and model data for delivery to ISBI servers
bash compress_isbi.sh
zip -r isbi_submission_2021.zip temp/
## Sanity check zip size
du -hs models/
du -hs isbi_submission_2021.zip

cat temp/*.sh
rm -rf temp

## login to the mack server and push all the code and data up to the moon
ssh mack
lftp -u ctc148,$LFTPPASS ftp.celltrackingchallenge.net 
## make sure that cp2remote.lftp doesn't comment out the data we want!
## Manually copy commands from `cp2remote.lftp`

## Congratulations, you're done!
