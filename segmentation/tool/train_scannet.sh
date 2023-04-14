#!/bin/sh
export PYTHONPATH=./

PYTHON=python
dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp tool/train_scannet.sh tool/train_scannet.py ${config} ${exp_dir}
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

$PYTHON tool/train_scannet.py --config=${config} 2>&1 | tee ${model_dir}/train-$now.log

