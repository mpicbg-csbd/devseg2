
rm -rf temp
mkdir temp

cp Readme.md                     temp/
cp -r models/                    temp/
cp predict.py   temp/
cp utils.py											 temp/
cp requirements.txt              temp/
cp torch_models.py               temp/
cp tracking.py                   temp/
cp Fluo-* DIC-* BF-* PhC-*       temp/

cd temp
## this awful command requires \ as regex prefix EXCEPT FOR . (dot) character !
# sed -i "s/\/projects\/project-broaddus\/rawdata\/isbi_challenge_out_dense\/.\+\//\.\//g" *dense.sh
sed -i "s/\/projects\/project-broaddus\/rawdata\/isbi_challenge_out_dense/../g" *dense.sh
sed -r -i "s/(0[12]_RES)/CSB\/\1\//g" *dense.sh
sed -i "s/\/projects\/project-broaddus\/rawdata\/isbi_challenge_out/../g" *.sh
sed -i "s/\/projects\/project-broaddus\/rawdata\/isbi_challenge/../g" *.sh
cd ..

