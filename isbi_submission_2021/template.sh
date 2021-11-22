# #!/bin/bash

# echo "Current dir: $(pwd)"
# echo "Current Python: $(which python)"

python3 predict.py \
	-i "<indir>" \
	-o "<outdir>" \
	--cpnet_weights "models/<weightname>" \
	--zoom <zoom> \
	--nms_footprint <nms_footprint> \
	--scale <scale> \
	--radius <radius> \
	--mantrack_t0 "<mantrack_t0>" \
	--evalBorder <evalBorder> \

