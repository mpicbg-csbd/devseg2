
mkdir temp

cp Readme.md                     temp/
cp -r models/                    temp/
cp predict_stacks_new_local.py   temp/
cp utils.py											 temp/
cp requirements.txt              temp/
cp torch_models.py               temp/
cp tracking.py                   temp/
cp Fluo-* DIC-* BF-* PhC-*       temp/

cd temp
sed -i "s/\/projects\/project-broaddus\/rawdata\/isbi_challenge_out/../g" *.sh
sed -i "s/\/projects\/project-broaddus\/rawdata\/isbi_challenge/../g" *.sh
cd ..

zip -r isbi_submission_2021.zip temp/
# rm -rf temp