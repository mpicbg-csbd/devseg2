# /shared="2019-12-11-10-39-07-98-Trier_Tribolium_nGFP_window_highres/stacks/C0opticsprefused"
# cp /fileserver/myersspimdata/IMAGING/archive_lightsheet_data_good/$shared/0000[012]*.raw /projects/project-broaddus/rawdata/daniela/$shared/

ls /projects/project-broaddus/rawdata/daniela/

# run with 
# sbatch -J daniela -p gpu --gres gpu:1 -n 1 -c 1 -t 36:00:00 --mem 128000 -o slurm/daniela.out -e slurm/daniela.err --wrap 'python predict_stacks_daniela.py'

# --wrap '/bin/time -v ./Fluo-N3DL-TRIF-01.sh    '

# /bin/time -v ./my_env3/bin/python3 predict_stacks_new_local.py \
# 	-i "/projects/project-broaddus/rawdata/daniela/2019-12-11-10-39-07-98-Trier_Tribolium_nGFP_window_highres/stacks/C0opticsprefused/" \
# 	-o "/projects/project-broaddus/rawdata/daniela/pred/" \
# 	--cpnet_weights "models/Fluo-N3DL-TRIF-01+02_weights.pt" \
# 	--zoom 1.000 1.000 1.000  \
# 	--nms_footprint   3   5   5  \
# 	--scale 1.000 1.000 1.000  \

	# --mantrack_t0 "/projects/project-broaddus/rawdata/isbi_challenge/Fluo-N3DL-TRIF/01_GT/TRA/man_track000.tif" \
