
export PYTHONPATH=./
eval "$(conda shell.bash hook)"
PYTHON=python

TEST_CODE=test_s3dis.py

dataset=$1
exp_name=$2

exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
cp ${config} tool/test_s3dis.sh tool/${TEST_CODE} ${exp_dir}
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'


$PYTHON tool/test_s3dis.py \
  --config=${config} \
  save_folder ${result_dir}/best \
  model_path ${model_dir}/model_best.pth \
  2>&1 | tee ${exp_dir}/test_best-$now.log


