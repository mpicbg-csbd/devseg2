## move data from fileserver to local rawdata dir
/shared="2019-12-11-10-39-07-98-Trier_Tribolium_nGFP_window_highres/stacks/C0opticsprefused"
cp /fileserver/myersspimdata/IMAGING/archive_lightsheet_data_good/$shared/0000[012]*.raw /projects/project-broaddus/rawdata/daniela/$shared/
## The raw data lives here
ls /projects/project-broaddus/rawdata/daniela/2019-12-11-10-39-07-98-Trier_Tribolium_nGFP_window_highres/stacks/C0opticsprefused/

## turn on local env
source my_env3/bin/activate

## run with 
python predict_daniela_trib.py
## or
sbatch -J daniela -p gpu --gres gpu:1 -n 1 -c 1 -t 36:00:00 --mem 128000 -o slurm/daniela.out -e slurm/daniela.err --wrap 'python predict_daniela_trib.py'

## predicted points live in (points only. no masks.)
ls /projects/project-broaddus/rawdata/daniela/pred/ltps/ltps.npy

## make tracking movie with flow lines
cd ../src
python -c "import png_tracking as A; A.runDaniela()"
cd - 


