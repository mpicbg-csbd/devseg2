
tdir="/projects/project-broaddus/devseg_2/expr/e26_isbidet/train/"
mdir="/projects/project-broaddus/devseg_2/isbi_submission_2021/models/"
mkdir -p $mdir
rm $mdir/*.pt
pdir="/projects/project-broaddus/devseg_2/isbi_submission_2021/trainparams/"
mkdir -p $pdir
rm $pdir/*.pkl

cp "$tdir/pid000/m/best_weights_loss.pt"              "$mdir/BF-C2DL-HSC-01+02_weights.pt"
cp "$tdir/pid001/m/best_weights_loss.pt"              "$mdir/BF-C2DL-MuSC-01+02_weights.pt"
cp "$tdir/DIC-C2DH-HeLa/both/m/best_weights_loss.pt"  "$mdir/DIC-C2DH-HeLa-01+02_weights.pt"
cp "$tdir/Fluo-C2DL-MSC/both/m/best_weights_loss.pt"  "$mdir/Fluo-C2DL-MSC-01+02_weights.pt"
cp "$tdir/pid004/m/best_weights_loss.pt"              "$mdir/Fluo-C3DH-A549-01+02_weights.pt"
cp "$tdir/pid005/m/best_weights_loss.pt"              "$mdir/Fluo-C3DH-A549-SIM-01+02_weights.pt"
cp "$tdir/pid006/m/best_weights_loss.pt"              "$mdir/Fluo-C3DH-H157-01+02_weights.pt"
cp "$tdir/pid007/m/best_weights_loss.pt"              "$mdir/Fluo-C3DL-MDA231-01+02_weights.pt"
cp "$tdir/pid008/m/best_weights_loss.pt"              "$mdir/Fluo-N2DH-GOWT1-01+02_weights.pt"
cp "$tdir/pid009/m/best_weights_loss.pt"              "$mdir/Fluo-N2DH-SIM+-01+02_weights.pt"
cp "$tdir/pid010/m/best_weights_loss.pt"              "$mdir/Fluo-N2DL-HeLa-01+02_weights.pt"
cp "$tdir/pid011/m/best_weights_loss.pt"              "$mdir/Fluo-N3DH-CE-01+02_weights.pt"
cp "$tdir/pid012/m/best_weights_loss.pt"              "$mdir/Fluo-N3DH-CHO-01+02_weights.pt"
cp "$tdir/pid013/m/best_weights_loss.pt"              "$mdir/Fluo-N3DH-SIM+-01+02_weights.pt"
cp "$tdir/pid014/m/best_weights_loss.pt"              "$mdir/Fluo-N3DL-DRO-01+02_weights.pt"
cp "$tdir/Fluo-N3DL-TRIC/both/m/best_weights_loss.pt" "$mdir/Fluo-N3DL-TRIC-01+02_weights.pt"
cp "$tdir/pid016/m/best_weights_loss.pt"              "$mdir/PhC-C2DH-U373-01+02_weights.pt"
cp "$tdir/pid017/m/best_weights_loss.pt"              "$mdir/PhC-C2DL-PSC-01+02_weights.pt"
cp "$tdir/pid018/m/best_weights_loss.pt"              "$mdir/Fluo-N3DL-TRIF-01+02_weights.pt"


cp "$tdir/pid000/params.pkl"              "$pdir/BF-C2DL-HSC-01+02_params.pkl"
cp "$tdir/pid001/params.pkl"              "$pdir/BF-C2DL-MuSC-01+02_params.pkl"
cp "$tdir/DIC-C2DH-HeLa/both/params.pkl"  "$pdir/DIC-C2DH-HeLa-01+02_params.pkl"
cp "$tdir/Fluo-C2DL-MSC/both/params.pkl"  "$pdir/Fluo-C2DL-MSC-01+02_params.pkl"
cp "$tdir/pid004/params.pkl"              "$pdir/Fluo-C3DH-A549-01+02_params.pkl"
cp "$tdir/pid005/params.pkl"              "$pdir/Fluo-C3DH-A549-SIM-01+02_params.pkl"
cp "$tdir/pid006/params.pkl"              "$pdir/Fluo-C3DH-H157-01+02_params.pkl"
cp "$tdir/pid007/params.pkl"              "$pdir/Fluo-C3DL-MDA231-01+02_params.pkl"
cp "$tdir/pid008/params.pkl"              "$pdir/Fluo-N2DH-GOWT1-01+02_params.pkl"
cp "$tdir/pid009/params.pkl"              "$pdir/Fluo-N2DH-SIM+-01+02_params.pkl"
cp "$tdir/pid010/params.pkl"              "$pdir/Fluo-N2DL-HeLa-01+02_params.pkl"
cp "$tdir/pid011/params.pkl"              "$pdir/Fluo-N3DH-CE-01+02_params.pkl"
cp "$tdir/pid012/params.pkl"              "$pdir/Fluo-N3DH-CHO-01+02_params.pkl"
cp "$tdir/pid013/params.pkl"              "$pdir/Fluo-N3DH-SIM+-01+02_params.pkl"
cp "$tdir/pid014/params.pkl"              "$pdir/Fluo-N3DL-DRO-01+02_params.pkl"
cp "$tdir/Fluo-N3DL-TRIC/both/params.pkl" "$pdir/Fluo-N3DL-TRIC-01+02_params.pkl"
cp "$tdir/pid016/params.pkl"              "$pdir/PhC-C2DH-U373-01+02_params.pkl"
cp "$tdir/pid017/params.pkl"              "$pdir/PhC-C2DL-PSC-01+02_params.pkl"
cp "$tdir/pid018/params.pkl"              "$pdir/Fluo-N3DL-TRIF-01+02_params.pkl"