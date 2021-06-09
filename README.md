!!! WARNING
This repo is WIP and the README is usually outdated.

This projects hosts scientific python code for detection, segmentation and tracking of cells and nuclei in 2-D and 3-D microscopy images.

# Interactive usage

```bash
cd devseg_2/src
ipython ## python>=3.5
```

then in ipython on a machine with a GPU + CUDA.
```python
import e24_isbidet_AOT as A
import e24_trainer as B

## looks for Fluo-N2DH-GOWT1/01/t*.tif in /projects/project-broaddus/rawdata/GOWT1/

A.build_patchFrame(pid=16) # generate training data
B.train(pid=16) # train a CP-Net model
B.evaluate(pid=16) ## scores on train/vali/test patches
B.evaluate_imgFrame(pid=16) ## scores on full images

```


---


old, deprecated ipython workflow...
```python
## blocking cuda enables straightforward time profiling
export CUDA_LAUNCH_BLOCKING=1
ipython

import denoiser, detector, tracking
import networkx as nx
import numpy_indexed as ndi

import numpy as np
from segtools.ns2dir import load,save,toarray
import experiments2 as ex
import analysis2, ipy
%load_ext line_profiler
```
