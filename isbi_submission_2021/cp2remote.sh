#!/bin/bash


lftp -u ctc148,Kq6Z9N ftp.celltrackingchallenge.net << EOF

## TRANSFER CODE

ls

cd /SW
put /projects/project-broaddus/devseg_2/isbi_submission_2021/isbi_submission_2021.zip

# NOW TRANSFER THE DATA

# cd /BF-C2DL-HSC/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/BF-C2DL-HSC/01_RES/*
# cd /BF-C2DL-HSC/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/BF-C2DL-HSC/02_RES/*
# cd /BF-C2DL-MuSC/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/BF-C2DL-MuSC/01_RES/*
# cd /BF-C2DL-MuSC/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/BF-C2DL-MuSC/02_RES/*

# cd /DIC-C2DH-HeLa/01_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/DIC-C2DH-HeLa/01_RES/*
# cd /DIC-C2DH-HeLa/02_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/DIC-C2DH-HeLa/02_RES/*
# cd /Fluo-C2DL-Huh7/01_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C2DL-Huh7/01_RES/*
# cd /Fluo-C2DL-Huh7/02_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C2DL-Huh7/02_RES/*
# cd /Fluo-C2DL-MSC/01_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C2DL-MSC/01_RES/*
# cd /Fluo-C2DL-MSC/02_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C2DL-MSC/02_RES/*
cd /Fluo-N2DH-GOWT1/01_RES/
mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N2DH-GOWT1/01_RES/*
cd /Fluo-N2DH-GOWT1/02_RES/
mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N2DH-GOWT1/02_RES/*
# cd /Fluo-N2DL-HeLa/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N2DL-HeLa/01_RES/*
# cd /Fluo-N2DL-HeLa/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N2DL-HeLa/02_RES/*
# cd /PhC-C2DH-U373/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/PhC-C2DH-U373/01_RES/*
# cd /PhC-C2DH-U373/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/PhC-C2DH-U373/02_RES/*
# cd /PhC-C2DL-PSC/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/PhC-C2DL-PSC/01_RES/*
# cd /PhC-C2DL-PSC/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/PhC-C2DL-PSC/02_RES/*
# cd /Fluo-N2DH-SIM+/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N2DH-SIM+/01_RES/*
# cd /Fluo-N2DH-SIM+/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N2DH-SIM+/02_RES/*
# cd /Fluo-C3DH-A549/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C3DH-A549/01_RES/*
# cd /Fluo-C3DH-A549/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C3DH-A549/02_RES/*
# cd /Fluo-C3DH-H157/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C3DH-H157/01_RES/*
# cd /Fluo-C3DH-H157/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C3DH-H157/02_RES/*
# cd /Fluo-C3DL-MDA231/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C3DL-MDA231/01_RES/*
# cd /Fluo-C3DL-MDA231/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C3DL-MDA231/02_RES/*
# cd /Fluo-N3DH-CE/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DH-CE/01_RES/*
# cd /Fluo-N3DH-CE/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DH-CE/02_RES/*
# cd /Fluo-N3DH-CHO/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DH-CHO/01_RES/*
# cd /Fluo-N3DH-CHO/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DH-CHO/02_RES/*
# cd /Fluo-N3DL-DRO/01_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DL-DRO/01_RES/*
# cd /Fluo-N3DL-DRO/02_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DL-DRO/02_RES/*
# cd /Fluo-N3DL-TRIC/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DL-TRIC/01_RES/*
# cd /Fluo-N3DL-TRIC/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DL-TRIC/02_RES/*
# cd /Fluo-N3DL-TRIF/01_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DL-TRIF/01_RES/*
# cd /Fluo-N3DL-TRIF/02_RES/
# # mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DL-TRIF/02_RES/*
# cd /Fluo-C3DH-A549-SIM/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C3DH-A549-SIM/01_RES/*
# cd /Fluo-C3DH-A549-SIM/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-C3DH-A549-SIM/02_RES/*
# cd /Fluo-N3DH-SIM+/01_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DH-SIM+/01_RES/*
# cd /Fluo-N3DH-SIM+/02_RES/
# mput /projects/project-broaddus/rawdata/isbi_challenge_out/Fluo-N3DH-SIM+/02_RES/*


bye
EOF