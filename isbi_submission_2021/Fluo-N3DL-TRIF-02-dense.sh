# #!/bin/bash

# echo "Current dir: $(pwd)"
# echo "Current Python: $(which python)"

python3 predict.py \
	-i "/projects/project-broaddus/rawdata/isbi_challenge/Fluo-N3DL-TRIF/02" \
	-o "/projects/project-broaddus/rawdata/isbi_challenge_out_dense/Fluo-N3DL-TRIF/02_RES" \
	--cpnet_weights "models/Fluo-N3DL-TRIF-01+02_weights.pt" \
	--zoom 0.500 0.500 0.500  \
	--nms_footprint   3   5   5  \
	--scale 1.000 1.000 1.000  \
	--radius 10 \
	--mantrack_t0 "None" \
	--evalBorder   0   0   0  \

