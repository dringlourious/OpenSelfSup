#!/usr/bin/env bash

export PYTHONPATH=$(pwd):$PYTHONPATH

CFG=configs/selfsup/moco/cifar10_r50v2.py
GPUS=1
PORT=12348

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CFG --work_dir $WORK_DIR --seed 0 --launcher pytorch --deterministic
