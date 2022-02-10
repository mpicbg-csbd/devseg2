!!! WARNING
This repo is WIP and the README is usually outdated.

This projects hosts scientific python code for detection, segmentation and tracking of cells and nuclei in 2-D and 3-D microscopy images, denoising using StructN2V method, and detection of mitotic nuclei.


# Basic usage

```bash
cd devseg_2/src
module load gcc/6.2.0
module load cuda/9.2.148
source myenv/bin/activate

## generate training data, train and predict on test data.
python e23_mauricio2.py 
```

Data is usually stored in `../expr/{file name}/{function name}/*`.
E.g. `e23_mauricio2.train()` stores data in `../expr/e23_mauricio2/train/*`.

Run via SLURM with

```bash
sbatch -J e23-mau -p gpu --gres gpu:1 -n 1 -t  6:00:00 -c 1 --mem 128000 -o slurm_out/e23-mau.out -e slurm_err/e23-mau.out --wrap '/bin/time -v python e23_mauricio2.py'
```