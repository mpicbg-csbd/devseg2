#!/bin/bash

echo $(pwd)

python3 predict_stacks_new.py \
	-i "../Fluo-N2DH-GOWT1/01/" \
	-o "../Fluo-N2DH-GOWT1/01_RES/" \
	--cpnet "models/Fluo-N2DH-GOWT1-01_weights.pt" \
	--zoom 1. 1. \
	--nms_footprint 5 5 \
