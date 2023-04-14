#!/bin/sh
export PYTHONPATH=./

PYTHON=python
dataset=$1
exp_dir=exp/${dataset}/pretrain
model_dir=${exp_dir}/model
config=config/${dataset}/${dataset}_pretrain.yaml

mkdir -p ${model_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp tool/pretrain.sh tool/pretrain.py ${config} ${exp_dir}
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

CUDA_LAUNCH_BLOCKING=1 $PYTHON tool/pretrain.py --config=${config} 2>&1 | tee ${model_dir}/train-$now.log

