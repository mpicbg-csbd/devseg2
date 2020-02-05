This is not a library.
If you want to use the code herein, you're gonna have to rename all the hard-coded paths and manually find and adjust the appropriate hyperparameters...
But it's not that complex, and I welcome you, oh courageous explorer, to try.

## usage and names

- `detect*.py`  data-specific modules for training centerpoint detection
- `denoise*.py` data-specific modules for applying (Structured) Noise2Void
- `projection.py` data-specific 3D centerpoint detection models with 2D training data on max projections
- `Snakemake`, `files.py`, `cluster.yaml` are for making the whole project run in parallel on the cluster
- `evaluation.py`, `point_matcher.py`, `predict.py` for predictions and evaluations
- `ipy.py`, `ns2dir.py` utils
- `torch_models.py` implements U-net and associated helper funcs

To train and use a centerpoint detection model, e.g. `detect_adapt_fly.py` open ipython (on machine GPU) and do :

```
import detect_adapt_fly
m,d,td,ta = detect_adapt_fly.init("my_local_dir/experiment01/")
detect_adapt_fly.train(m,d,td,ta)

## names
## m  :: models (update in place)
## vd :: validation data
## td :: training data
## ta :: training artifacts (update in place)

import predict
from skimage.feature  import peak_local_max

result_image = predict.apply_net_tiled_3d(m.net,input_image)
centerpoints = peak_local_max(result_image,threshold_abs=0.2,exclude_border=False,footprint=np.ones((3,3,3)))
```

if you want to stop training just use Ctrl-C, it will restart at the same iteration where you left off because of state in ta and m.

--- 

Repos are supposed to have helpful README's right?
Does this count?