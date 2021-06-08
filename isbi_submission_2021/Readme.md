
## Overview

Python scripts to perform cell detection/segmentation with CP-Net.

The bash scripts in this folder follow the ISBI CTC calling convention.
They are runnable without args, i.e. as `./Fluo-N2DH-GOWT1-01.sh` in the cwd.
Associated folders of TIF files are expected to be located in `../Fluo-N2DH-GOWT1/01/` with names in glob `t*.tif`.

These bash scripts run `predict_stacks_new.py` with data-specific params.

## Install

We require the following dependencies:
```txt
- python=3.6
- ipython
- tifffile
- numpy
- scipy
- scikit-image
- pytorch
- cudatoolkit=9.2
- pykdtree
```

Use conda to install: `conda env create -f environment.yml`.


## Example usage


```bash
cd isbi_submission_2021
./Fluo-N2DH-GOWT1-01.sh
```

Requires
```
../Fluo-N2DH-GOWT1/01/
├── t003.tif
├── t008.tif
├── t013.tif
├── t018.tif
├── t023.tif
├── t028.tif
```

Produces
```
../Fluo-N2DH-GOWT1/01_RES/
├── mask003.tif
├── mask008.tif
├── mask013.tif
├── mask018.tif
├── mask023.tif
├── mask028.tif
```
