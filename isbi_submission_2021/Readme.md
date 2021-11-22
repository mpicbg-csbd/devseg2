
## Overview

Python scripts to perform cell detection with CP-Net and tracking with nearest parent connections.

The bash scripts in this folder follow the ISBI CTC naming conventions.
They are runnable without args, i.e. as `./Fluo-N2DH-GOWT1-01.sh` in the cwd.
Associated folders of raw data (TIFF) files are expected to be located in `../Fluo-N2DH-GOWT1/01/` with names in glob `t*.tif`.

These bash scripts run `predict.py` with data-specific params.

## Install

tested with:
- python 3.6
- pip 21
- cudatoolkit 9.2 and 9.0

Python deps are listed in `requirements.txt`

We require the following python dependencies:

```

# -- prediction only (CTC and CSC)

pip install numpy
pip install torch
pip install tifffile
pip install scipy
pip install scikit-image
pip install imagecodecs
pip install pykdtree
pip install numpy-indexed

# -- for interactive 

pip install ipython
pip install jedi==0.17.2
pip install ipdb

# -- for training torch models

pip install pandas
pip install -e ../../segtools/
pip install zarr
pip install git+https://github.com/stardist/augmend.git
pip install psutil
pip install tqdm
pip install gputools
```

## Example usage


```bash
cd isbi_submission_2021
./Fluo-N2DH-GOWT1-01.sh
```

Requires
```
../Fluo-N2DH-GOWT1/01/
├── t000.tif
├── t001.tif
├── t002.tif
├── t003.tif
├── t004.tif
├── t005.tif
```

Produces
```
../Fluo-N2DH-GOWT1/01_RES/
├── mask000.tif
├── mask001.tif
├── mask002.tif
├── mask003.tif
├── mask004.tif
├── mask005.tif
├── res_track.txt
```
