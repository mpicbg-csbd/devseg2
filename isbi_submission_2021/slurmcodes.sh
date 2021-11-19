## NOTE: You must run these jobs from an _active_ virtualenv in the local dir.

sbatch -J BF-C2DL-HSC-01        -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/BF-C2DL-HSC-01.txt        -e slurm_err/BF-C2DL-HSC-01.txt         --wrap '/bin/time -v ./BF-C2DL-HSC-01.sh       '
sbatch -J BF-C2DL-HSC-02        -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/BF-C2DL-HSC-02.txt        -e slurm_err/BF-C2DL-HSC-02.txt         --wrap '/bin/time -v ./BF-C2DL-HSC-02.sh       '
sbatch -J BF-C2DL-MuSC-01       -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/BF-C2DL-MuSC-01.txt       -e slurm_err/BF-C2DL-MuSC-01.txt        --wrap '/bin/time -v ./BF-C2DL-MuSC-01.sh      '
sbatch -J BF-C2DL-MuSC-02       -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/BF-C2DL-MuSC-02.txt       -e slurm_err/BF-C2DL-MuSC-02.txt        --wrap '/bin/time -v ./BF-C2DL-MuSC-02.sh      '
sbatch -J DIC-C2DH-HeLa-01      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/DIC-C2DH-HeLa-01.txt      -e slurm_err/DIC-C2DH-HeLa-01.txt       --wrap '/bin/time -v ./DIC-C2DH-HeLa-01.sh     '
sbatch -J DIC-C2DH-HeLa-02      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/DIC-C2DH-HeLa-02.txt      -e slurm_err/DIC-C2DH-HeLa-02.txt       --wrap '/bin/time -v ./DIC-C2DH-HeLa-02.sh     '
sbatch -J Fluo-C2DL-MSC-01      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C2DL-MSC-01.txt      -e slurm_err/Fluo-C2DL-MSC-01.txt       --wrap '/bin/time -v ./Fluo-C2DL-MSC-01.sh     '
sbatch -J Fluo-C2DL-MSC-02      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C2DL-MSC-02.txt      -e slurm_err/Fluo-C2DL-MSC-02.txt       --wrap '/bin/time -v ./Fluo-C2DL-MSC-02.sh     '
sbatch -J Fluo-C3DH-A549-01     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C3DH-A549-01.txt     -e slurm_err/Fluo-C3DH-A549-01.txt      --wrap '/bin/time -v ./Fluo-C3DH-A549-01.sh    '
sbatch -J Fluo-C3DH-A549-02     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C3DH-A549-02.txt     -e slurm_err/Fluo-C3DH-A549-02.txt      --wrap '/bin/time -v ./Fluo-C3DH-A549-02.sh    '
sbatch -J Fluo-C3DH-A549-SIM-01 -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C3DH-A549-SIM-01.txt -e slurm_err/Fluo-C3DH-A549-SIM-01.txt  --wrap '/bin/time -v ./Fluo-C3DH-A549-SIM-01.sh'
sbatch -J Fluo-C3DH-A549-SIM-02 -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C3DH-A549-SIM-02.txt -e slurm_err/Fluo-C3DH-A549-SIM-02.txt  --wrap '/bin/time -v ./Fluo-C3DH-A549-SIM-02.sh'
sbatch -J Fluo-C3DH-H157-01     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C3DH-H157-01.txt     -e slurm_err/Fluo-C3DH-H157-01.txt      --wrap '/bin/time -v ./Fluo-C3DH-H157-01.sh    '
sbatch -J Fluo-C3DH-H157-02     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C3DH-H157-02.txt     -e slurm_err/Fluo-C3DH-H157-02.txt      --wrap '/bin/time -v ./Fluo-C3DH-H157-02.sh    '
sbatch -J Fluo-C3DL-MDA231-01   -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C3DL-MDA231-01.txt   -e slurm_err/Fluo-C3DL-MDA231-01.txt    --wrap '/bin/time -v ./Fluo-C3DL-MDA231-01.sh  '
sbatch -J Fluo-C3DL-MDA231-02   -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-C3DL-MDA231-02.txt   -e slurm_err/Fluo-C3DL-MDA231-02.txt    --wrap '/bin/time -v ./Fluo-C3DL-MDA231-02.sh  '
sbatch -J Fluo-N2DH-GOWT1-01    -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N2DH-GOWT1-01.txt    -e slurm_err/Fluo-N2DH-GOWT1-01.txt     --wrap '/bin/time -v ./Fluo-N2DH-GOWT1-01.sh   '
sbatch -J Fluo-N2DH-GOWT1-02    -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N2DH-GOWT1-02.txt    -e slurm_err/Fluo-N2DH-GOWT1-02.txt     --wrap '/bin/time -v ./Fluo-N2DH-GOWT1-02.sh   '
sbatch -J Fluo-N2DH-SIM+-01     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N2DH-SIM+-01.txt     -e slurm_err/Fluo-N2DH-SIM+-01.txt      --wrap '/bin/time -v ./Fluo-N2DH-SIM+-01.sh    '
sbatch -J Fluo-N2DH-SIM+-02     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N2DH-SIM+-02.txt     -e slurm_err/Fluo-N2DH-SIM+-02.txt      --wrap '/bin/time -v ./Fluo-N2DH-SIM+-02.sh    '
sbatch -J Fluo-N2DL-HeLa-01     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N2DL-HeLa-01.txt     -e slurm_err/Fluo-N2DL-HeLa-01.txt      --wrap '/bin/time -v ./Fluo-N2DL-HeLa-01.sh    '
sbatch -J Fluo-N2DL-HeLa-02     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N2DL-HeLa-02.txt     -e slurm_err/Fluo-N2DL-HeLa-02.txt      --wrap '/bin/time -v ./Fluo-N2DL-HeLa-02.sh    '
sbatch -J Fluo-N3DH-CE-01       -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DH-CE-01.txt       -e slurm_err/Fluo-N3DH-CE-01.txt        --wrap '/bin/time -v ./Fluo-N3DH-CE-01.sh      '
sbatch -J Fluo-N3DH-CE-02       -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DH-CE-02.txt       -e slurm_err/Fluo-N3DH-CE-02.txt        --wrap '/bin/time -v ./Fluo-N3DH-CE-02.sh      '
sbatch -J Fluo-N3DH-CHO-01      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DH-CHO-01.txt      -e slurm_err/Fluo-N3DH-CHO-01.txt       --wrap '/bin/time -v ./Fluo-N3DH-CHO-01.sh     '
sbatch -J Fluo-N3DH-CHO-02      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DH-CHO-02.txt      -e slurm_err/Fluo-N3DH-CHO-02.txt       --wrap '/bin/time -v ./Fluo-N3DH-CHO-02.sh     '
sbatch -J Fluo-N3DH-SIM+-01     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DH-SIM+-01.txt     -e slurm_err/Fluo-N3DH-SIM+-01.txt      --wrap '/bin/time -v ./Fluo-N3DH-SIM+-01.sh    '
sbatch -J Fluo-N3DH-SIM+-02     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DH-SIM+-02.txt     -e slurm_err/Fluo-N3DH-SIM+-02.txt      --wrap '/bin/time -v ./Fluo-N3DH-SIM+-02.sh    '
sbatch -J Fluo-N3DL-DRO-01      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DL-DRO-01.txt      -e slurm_err/Fluo-N3DL-DRO-01.txt       --wrap '/bin/time -v ./Fluo-N3DL-DRO-01.sh     '
sbatch -J Fluo-N3DL-DRO-02      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DL-DRO-02.txt      -e slurm_err/Fluo-N3DL-DRO-02.txt       --wrap '/bin/time -v ./Fluo-N3DL-DRO-02.sh     '
sbatch -J Fluo-N3DL-TRIC-01     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DL-TRIC-01.txt     -e slurm_err/Fluo-N3DL-TRIC-01.txt      --wrap '/bin/time -v ./Fluo-N3DL-TRIC-01.sh    '
sbatch -J Fluo-N3DL-TRIC-02     -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/Fluo-N3DL-TRIC-02.txt     -e slurm_err/Fluo-N3DL-TRIC-02.txt      --wrap '/bin/time -v ./Fluo-N3DL-TRIC-02.sh    '
sbatch -J Fluo-N3DL-TRIF-01     -p gpu --gres gpu:1 -n 1 -c 1 -t 36:00:00 --mem 128000 -o slurm_out/Fluo-N3DL-TRIF-01.txt    -e slurm_err/Fluo-N3DL-TRIF-01.txt      --wrap '/bin/time -v ./Fluo-N3DL-TRIF-01.sh    '
sbatch -J Fluo-N3DL-TRIF-02     -p gpu --gres gpu:1 -n 1 -c 1 -t 36:00:00 --mem 128000 -o slurm_out/Fluo-N3DL-TRIF-02.txt    -e slurm_err/Fluo-N3DL-TRIF-02.txt      --wrap '/bin/time -v ./Fluo-N3DL-TRIF-02.sh    '
sbatch -J PhC-C2DH-U373-01      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/PhC-C2DH-U373-01.txt      -e slurm_err/PhC-C2DH-U373-01.txt       --wrap '/bin/time -v ./PhC-C2DH-U373-01.sh     '
sbatch -J PhC-C2DH-U373-02      -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/PhC-C2DH-U373-02.txt      -e slurm_err/PhC-C2DH-U373-02.txt       --wrap '/bin/time -v ./PhC-C2DH-U373-02.sh     '
sbatch -J PhC-C2DL-PSC-01       -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/PhC-C2DL-PSC-01.txt       -e slurm_err/PhC-C2DL-PSC-01.txt        --wrap '/bin/time -v ./PhC-C2DL-PSC-01.sh      '
sbatch -J PhC-C2DL-PSC-02       -p gpu --gres gpu:1 -n 1 -c 1 -t 3:30:00 --mem 128000 -o slurm_out/PhC-C2DL-PSC-02.txt       -e slurm_err/PhC-C2DL-PSC-02.txt        --wrap '/bin/time -v ./PhC-C2DL-PSC-02.sh      '



## ISBI Challenge data prediction timings

# 7:07 	Fluo-N3DL-TRIF-01
# 4:19 	Fluo-N3DL-TRIF-02
# 1:02 	Fluo-N3DL-TRIC-02
# 0:47 	Fluo-C3DH-H157-02
# 0:46 	Fluo-C3DH-H157-01
# 0:37 	Fluo-N3DL-TRIC-01
# 0:32 	Fluo-N3DL-DRO-01
# 0:32 	Fluo-N3DL-DRO-02
# 0:13 	Fluo-N3DH-SIM+-01
# 0:11 	Fluo-N3DH-SIM+-02
# 0:10 	Fluo-N3DH-CHO-01
# 0:08 	Fluo-N3DH-CE-01
# 0:07 	BF-C2DL-HSC-02
# 0:07 	BF-C2DL-HSC-01
# 0:07 	Fluo-N3DH-CE-02
# 0:06 	BF-C2DL-MuSC-02
# 0:06 	BF-C2DL-MuSC-01
# 0:03 	Fluo-N3DH-CHO-02
# 0:03 	PhC-C2DL-PSC-02
# 0:03 	PhC-C2DL-PSC-01
# 0:02 	Fluo-C3DH-A549-SIM-02
# 0:02 	Fluo-C3DH-A549-02
# 0:02 	Fluo-C3DH-A549-SIM-01
# 0:02 	Fluo-C3DL-MDA231-02
# 0:02 	PhC-C2DH-U373-02
# 0:02 	PhC-C2DH-U373-01
# 0:02 	Fluo-C3DH-A549-01
# 0:02 	Fluo-C3DL-MDA231-01
# 0:02 	Fluo-N2DH-GOWT1-02
# 0:02 	Fluo-N2DH-SIM+-02
# 0:02 	Fluo-N2DL-HeLa-02
# 0:02 	Fluo-N2DH-SIM+-01
# 0:02 	Fluo-N2DL-HeLa-01
# 0:02 	Fluo-N2DH-GOWT1-01
# 0:02 	DIC-C2DH-HeLa-01
# 0:02 	DIC-C2DH-HeLa-02
# 0:02 	Fluo-C2DL-MSC-02
# 0:01 	Fluo-C2DL-MSC-01