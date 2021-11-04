
mkdir temp
cp Readme.md               temp/
cp expand_labels_scikit.py temp/
cp models                  temp/
cp predict_stacks_new.py   temp/
cp requirements.txt        temp/
cp torch_models.py         temp/
cp tracking.py             temp/
zip -r isbi_submission_2021.zip temp/
rm -rf temp