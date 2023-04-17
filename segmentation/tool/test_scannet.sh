#!/bin/sh

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
PYTHON=python

TEST_CODE=test_scannet.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}_scannetpre/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
cp ${config} tool/test_scannet.sh tool/${TEST_CODE} ${exp_dir}
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=7 $PYTHON tool/test_scannet.py \
  --config=${config} \
  save_folder ${result_dir}/best \
  model_path ${model_dir}/model_best_72.8.pth \
  2>&1 | tee ${exp_dir}/test_best-$now.log

# $PYTHON tool/test.py \
#   --config=${config} \
#   save_folder ${result_dir}/last \
#   model_path ${model_dir}/model_last.pth \
#   2>&1 | tee ${exp_dir}/test_last-$now.log

#: '
# $PYTHON -u ${exp_dir}/${TEST_CODE} \
#   --config=${config} \
#   save_folder ${result_dir}/best \
#   model_path ${model_dir}/model_best.pth \
#   2>&1 | tee ${exp_dir}/test_best-$now.log
#'

#: '
# $PYTHON -u ${exp_dir}/${TEST_CODE} \
#   --config=${config} \
#   save_folder ${result_dir}/last \
#   model_path ${model_dir}/model_last.pth \
#   2>&1 | tee ${exp_dir}/test_last-$now.log
#'
