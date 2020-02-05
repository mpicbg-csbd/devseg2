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

--- 

Repos are supposed to have helpful README's right?
Does this count?