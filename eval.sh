# 随时输出log: cityscapes
date_now=$(date +"%Y%m%d_%H%M%S")

python=../envs/torch11/bin/python

ROOT_DIR='../data/cityscapes_split_erase19'


$python -u segmentation/eval.py \
--gpu="0" \
--dataset="cityscapes" \
--root_dir=$ROOT_DIR \
--num_classes=19 \
--data="val" \
--num_workers=4 \
--model="bisenetv2" \
--checkpoint="saved.pth" \
--USE_ERASE_DATA=True \
| tee -a "save/logs/eval_log-${date_now}.txt"


# --root_dir="/disk1/fll_data/cityscapes_split_erase" \ # Non-IID-19class
# 在运行自己划分的数据集时要把dataloader.py中的label_remap函数注释掉
# 并且切换self.new_classes的映射值

