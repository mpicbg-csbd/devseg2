
bash:
	## run locally to recreate bash scripts.
	python3 prepare_bash_scripts.py

test:
	## to run with full timing and max resident memory output
	/bin/time -v ./test_Fluo-N2DH-SIM+-01.sh 1> out 2> err &

clean:
	# 	rm err out
	echo "The clean step"

wipe: clean
	# 	rm -rf __pycache__
	# 	rm -rf envs/
	echo "hello my friend"

