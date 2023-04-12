
### Requirements


## pretrain the backbone

# for s3dis
- change the data_root in ./config/s3dis/s3dis_pretrain.yaml
- sh tool/pretrain.sh s3dis

# for scannet
- change the data_root in ./config/s3dis/scannet_pretrain.yaml
- sh tool/pretrain.sh scannet


## finetuning

- sh tool/train_s3dis.sh s3dis pointtransformer_repro
- sh tool/train_scannet.sh scannet pointtransformer_repro


## test
- the best model on s3dis can be found in
- sh too/test_s3dis.sh s3dis pointtransformer_repro


