# Semantic Segmentation

## Requirements
- Ubuntu: 18.04 or higher
- CUDA: 11.1
- pytorch: 1.9.0
- Hardware: 4 x 24G memory GPUs or better

## Installation
First you can create an anaconda environment called `mm3dscene`:
```bash
conda create -n mm3dscene python=3.7 -y
conda activate mm3dscene
```

Then install Dependecies

```bash
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx -y
```


Next install cuda operations

```bash
cd segmentation/lib/pointops
python3 setup.py install
cd ../../..
```


## Data preparation
For S3DIS, download the [dateset](https://drive.google.com/uc?export=download&id=1KUxWagmEWnvMhEb4FRwq2Mj0aa3U3xUf) and change the 'data_root' in ./config/s3dis/s3dis_pretrain.yaml

For ScanNet, follow the [README](/detection/pretrain/scannet/README.md) under the `scannet` folder, download ScanNet dataset and extract point clouds and  semantic segmentation annotations.



## Usage

### Pretraining on s3dis

- cd segmentation
- sh tool/pretrain.sh s3dis


### finetuning on s3dis
- sh tool/train_s3dis.sh s3dis pointtransformer_repro


## test
The best model on s3dis can be found from [here]()
- sh tool/test_s3dis.sh s3dis pointtransformer_repro


