
# #!/bin/bash

echo $(pwd)
echo $(ls)
echo "Current Python: $(which python3)"

python3 predict_stacks_new.py \
	-i "/projects/project-broaddus/rawdata/GOWT1/Fluo-N2DH-GOWT1/01/" \
	-o "/projects/project-broaddus/rawdata/GOWT1/Fluo-N2DH-GOWT1/01_RES/" \
	--cpnet_weights "models/Fluo-N2DH-GOWT1-01_weights.pt" \
	--zoom 1. 1. \
	--nms_footprint 5 5 \
