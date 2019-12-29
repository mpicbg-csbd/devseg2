from pathlib import Path
import itertools

home = Path("/lustre/projects/project-broaddus/devseg_2/")
rawdata = Path("/lustre/projects/project-broaddus/rawdata/")

## always given: all the raw data and annotations

raw_ce_chal_01   = [rawdata / f"celegans_isbi/Fluo-N3DH-CE_challenge/01/t{n:03d}.tif" for n in range(190)]
raw_ce_chal_02   = [rawdata / f"celegans_isbi/Fluo-N3DH-CE_challenge/02/t{n:03d}.tif" for n in range(190)]

raw_ce_train_01  = [rawdata / f"celegans_isbi/Fluo-N3DH-CE/01/t{n:03d}.tif" for n in range(250)]
raw_ce_train_02  = [rawdata / f"celegans_isbi/Fluo-N3DH-CE/02/t{n:03d}.tif" for n in range(250)]
anno_ce_train_01 = [rawdata / f"celegans_isbi/Fluo-N3DH-CE/01/t{n:03d}.tif" for n in range(250)]
anno_ce_train_02 = [rawdata / f"celegans_isbi/Fluo-N3DH-CE/02/t{n:03d}.tif" for n in range(250)]

## wildcards

wc_raw_ce   = rawdata / "celegans_isbi/{dset}/{ab}/t{time}.tif"
wc_train_ce = home / "{ed}/{kd}/m/net10.pt"
wc_pred_ce  = home / "{ed}/{kd}/pred/{dset}/{ab}/p{time}.tif"


wc_pts_input_250 = [home / "{ed}/{kd}/pred/{dset}/{ab}/" / f"p{time:03d}.tif" for time in range(250)]
wc_pts_ce   = home / "{ed}/{kd}/pts/{dset}/{ab}/traj.pkl"

# wc_train_ce_view = home / "e02/t{k,[0-9]}/ta/res{m}view.tif"
# mk_wc_train_ce_view = [home / f"e02/t{k,[0-9]}/ta/res{m}view.tif" for k in [1,2,3,4] for m in range(10)]

## parameter sets

s_k      = [] #range(12,16)
# s_dset = ['Fluo-N3DH-CE', 'Fluo-N3DH-CE_challenge']
# s_ab   = ['01','02']
s_dset   = ['Fluo-N3DH-CE']
s_ab     = ['01','02']
s_times  = {'Fluo-N3DH-CE':range(250), 'Fluo-N3DH-CE_challenge':range(190)}

## results

train_ce   = [home / f"e02/t{k}/m/net40.pt" for k in s_k]
train_ce  += [home / f"e03/test/m/net10.pt"]

pts_ce   = [k.parent.parent / f"pts/{dset}/{ab}/traj.pkl" for k,dset,ab in itertools.product(train_ce,s_dset,s_ab)]

pred_ce  = [[k.parent.parent / f"pred/{dset}/{ab}/p{time:03d}.tif" 
              for time in s_times[dset]]
              for k,dset,ab in itertools.product(train_ce,s_dset,s_ab)]


# pred_ce  = [[home / f"e02/t{k}/pred/{dset}/{ab}/p{time:03d}.tif" 
#               for time in s_times[dset]]
#               for k,dset,ab in itertools.product(s_k,s_dset,s_ab)]


ls2 = [2,
     2,
     2,
     3,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     4,
     5,
     6,
     7,
     7,
     8,
     8,
     8,
     8,
     8,
     8,
     8,
     8,
     8,
     10,
     12,
     13,
     14,
     14,
     15,
     15,
     15,
     15,
     15,
     15,
     16,
     16,
     16,
     17,
     21,
     24,
     24,
     24,
     24,
     26,
     26,
     26,
     27,
     28,
     28,
     28,
     28,
     28,
     28,
     28,
     28,
     28,
     28,
     28,
     31,
     37,
     39,
     41,
     43,
     45,
     48,
     49,
     50,
     51,
     51,
     51,
     51,
     51,
     52,
     52,
     54,
     54,
     54,
     55,
     55,
     55,
     55,
     55,
     55,
     56,
     60,
     65,
     74,
     83,
     86,
     90,
     91,
     91,
     95,
     96,
     96,
     96,
     97,
     99,
     99,
     99,
     99,
     99,
     100,
     100,
     101,
     101,
     103,
     104,
     104,
     105,
     105,
     105,
     108,
     110,
     113,
     118,
     126,
     134,
     141,
     146,
     151,
     158,
     164,
     171,
     174,
     175,
     176,
     177,
     179,
     180,
     182,
     182,
     183,
     185,
     185,
     185,
     185,
     184,
     184,
     185,
     186,
     186,
     187,
     187,
     188,
     189,
     190,
     193,
     196,
     201,
     205,
     211,
     218,
     223,
     230,
     238,
     246,
     253,
     269,
     274,
     284,
     293,
     301,
     309,
     314,
     317,
     321,
     324,
     331,
     338,
     340,
     347,
     349,
     351,
     351,
     351,
     351,
     353,
     353,
     354,
     355,
     357,
     358,
     359,
     359,
     360]


## eval dataset

# wc_ce_scores = home / "e02/eval/Fluo-N3DH-CE/{ab}/scores.npy"
# ce_scores_01 = home / "e02/eval/Fluo-N3DH-CE/01/scores.npy"
# ce_scores_02 = home / "e02/eval/Fluo-N3DH-CE/02/scores.npy"
