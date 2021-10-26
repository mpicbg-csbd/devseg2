# #!/bin/bash

echo "Current dir: $(pwd)"
# echo "Current Python: $(which python)"

./my_env3/bin/python3 predict_stacks_new.py \
	-i "<indir>" \
	-o "<outdir>" \
	--cpnet_weights "models/<weightname>" \
	--zoom <zoom> \
	--nms_footprint <nms_footprint> \
