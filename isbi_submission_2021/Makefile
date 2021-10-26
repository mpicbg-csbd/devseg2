
bash:
	## run locally to recreate bash scripts.
	# 	./envs/bin/python3 prepare_bash_scripts.py
	python prepare_bash_scripts.py

test:
	## to run with full timing and max resident memory output
	# 	/bin/time -v ./test_Fluo-N2DH-SIM+-01.sh 1> out 2> err &
	/bin/time -v ./test_Fluo-N2DH-GOWT1-02.sh 1> out 2> err &

install:
	conda env create -f environment.yml --prefix ./envs

update:
	# will update and prune dependencies
	conda env update --prefix ./envs --file environment.yml  --prune

clean:
	echo "Clean err and out files..."
	rm err out

wipe: clean
	echo "A full wipe!"
	rm -rf __pycache__
	# 	rm -rf envs/


