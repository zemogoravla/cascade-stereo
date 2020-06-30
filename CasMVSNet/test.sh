#!/usr/bin/env bash
#TESTPATH="data/DTU/dtu_test_all"
#TESTLIST="lists/dtu/test.txt"
#CKPT_FILE=$1
#python test.py --dataset=general_eval --batch_size=1 --testpath=$TESTPATH  --testlist=$TESTLIST --loadckpt $CKPT_FILE ${@:2}

#!/usr/bin/env bash
TESTPATH="/home/agomez/Software/MultiStereo/YAO/DATA" #"data/DTU/dtu_test_all"
TESTLIST="lists/dtu/testAG.txt"
CKPT_FILE="/home/agomez/Software/MultiStereo/CASMVSNet/pretrained/casmvsnet.ckpt"
#CKPT_FILE=$1
python test.py --dataset=general_eval --batch_size=1 --testpath=$TESTPATH  --testlist=$TESTLIST --loadckpt $CKPT_FILE ${@:1}