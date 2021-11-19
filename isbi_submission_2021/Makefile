
zipit:
	python make_templates.py --target-remote
	bash compress_isbi.sh
	python make_templates.py --target-local

weights:
	cp /projects/project-broaddus/devseg_2/expr/e26_isbidet/train/DIC-C2DH-HeLa/both/m/best_weights_loss.pt  models/DIC-C2DH-HeLa-01+02_weights.pt
	cp /projects/project-broaddus/devseg_2/expr/e26_isbidet/train/Fluo-C2DL-MSC/both/m/best_weights_loss.pt  models/Fluo-C2DL-MSC-01+02_weights.pt
	cp /projects/project-broaddus/devseg_2/expr/e26_isbidet/train/Fluo-N3DL-TRIC/both/m/best_weights_loss.pt models/Fluo-N3DL-TRIC-01+02_weights.pt
