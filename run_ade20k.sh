# 随时输出log: cityscapes
date_now=$(date +"%Y%m%d_%H%M%S")
#python=../envs/torch11/bin/python

#ROOT_DIR='../data/cityscapes'
#ROOT_DIR='../data/cityscapes_split_erase19'
#ROOT_DIR='../data/cityscapes_split_erase19C2'
#ROOT_DIR='../voc'
ROOT_DIR='../ade20k_split_class_2'

LABEL_ONLINE_GEN=False
LOSSTYPE=back #ce,ohem,back,'dice','focal','lovasz','bce'

WARMSTEP=20
FRAC_NUM=5
LOCAL_EP=2
MIXLABLE=True
FEDPROX_MU=0
##################
DISTILL=False
TEMP_DIST=0.1
LAMB_PI=0.1
LAMB_PA=0
##################
RAND_INIT=False

##################PROTO_NEW
IS_PROTO=True
MOM_UPDATE=False

GLOBALEMA=False
PROTO_START_EPOCH=1
CON_LAMB=1
MOMENTUM=0.99
TEMP=0.07
EPOCH_NUM=800
MAX_ANCHOR=4096
KMEAN_NUM=2
PSEUDO_LABLE=True
PSEUDO_LABEL_START_EPOCH=1
LOCALMEM=True
CON_LAMB_LOCAL=1

##################
DATASET=ade20k #cityscapes #ade20k  #camvid
NUM_CLS=150
NUM_USERS=450

python -u segmentation/federated_main.py \
--gpu="0" \
--dataset=$DATASET \
--root_dir=$ROOT_DIR \
--USE_ERASE_DATA=True \
--num_classes=$NUM_CLS \
--data="train" \
--num_workers=4 \
--model="bisenetv2" \
--checkpoint="" \
--lr=0.05 \
--lr_scheduler="step" \
--iid=False \
--num_users=$NUM_USERS \
--frac_num=$FRAC_NUM \
--epochs=$EPOCH_NUM \
--local_ep=$LOCAL_EP \
--local_bs=8 \
--is_proto=$IS_PROTO \
--losstype=$LOSSTYPE \
--fedprox_mu=$FEDPROX_MU \
--label_online_gen=$LABEL_ONLINE_GEN \
--distill=$DISTILL \
--distill_lamb_pi=$LAMB_PI \
--distill_lamb_pa=$LAMB_PA \
--rand_init=$RAND_INIT \
--warmstep=$WARMSTEP \
--globalema=$GLOBALEMA \
--temp_dist=$TEMP_DIST \
--mixlabel=$MIXLABLE \
--proto_start_epoch=$PROTO_START_EPOCH \
--con_lamb=$CON_LAMB \
--con_lamb_local=$CON_LAMB_LOCAL \
--momentum=$MOMENTUM \
--temperature=$TEMP \
--max_anchor=$MAX_ANCHOR \
--kmean_num=$KMEAN_NUM \
--pseudo_label=$PSEUDO_LABLE \
--pseudo_label_start_epoch=$PSEUDO_LABEL_START_EPOCH \
--localmem=$LOCALMEM \
--mom_update=$MOM_UPDATE \
--save_frequency=20 \
--local_test_frequency=9999 \
--global_test_frequency=20 \
--USE_WANDB=0 \
--date_now=${date_now} \
| tee -a "save/logs/log-${date_now}.txt"


