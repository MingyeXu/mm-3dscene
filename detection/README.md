# 3D Object Detection

## Installation

### Requirements

It is required that you have access to GPUs. Matlab is required to prepare data for SUN RGB-D. The code is tested with Python v3.8, Pytorch v1.10 and CUDA 11.7.

Compile the CUDA layers [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network::

    cd pretrain/pointnet2
    python setup.py install

Compile another CUDA layers for pretraining MM-3DScene:

    cd pretrain/lib/pointops
    python setup.py install

Install the following Python dependencies (with `pip install`):

    tensorflow
    scipy
    matplotlib
    opencv-python
    plyfile
    'trimesh>=2.35.39,<2.35.40'
    'networkx>=2.2,<2.3'

### Data preparation

For ScanNet, follow the [README](./pretrain/scannet/README.md) under the `scannet` folder.

For SUN RGB-D, follow the [README](./pretrain/sunrgbd/README.md) under the `sunrgbd` folder.

## Usage

### Pre-training on ScanNet

To pretrain a [VoteNet](https://github.com/facebookresearch/votenet) model using our MM-3DScene on ScanNet data (fused scan), edit the [DATA_DIR](https://github.com/MingyeXu/mm-3dscene/tree/main/detection/pretrain/scannet/scannet_detection_dataset.py#L31) into the actual location of data, then run:

    cd pretrain

    CUDA_VISIBLE_DEVICES=x python pretrain.py --dataset scannet --log_dir log_scannet --num_point 40000 --use_color

### Fine-tuning on ScanNet

Get our pretrained checkpoint from [here](https://drive.google.com/file/d/1wmk7dfbOlp8-29Uo9lUBTshQL6-iw6Qu/view?usp=share_link).

To finetune a [VoteNet](https://github.com/facebookresearch/votenet) model pretrained by our MM-3DScene on ScanNet data (fused scan), edit the [DATA_DIR](https://github.com/MingyeXu/mm-3dscene/tree/main/detection/downstream/scannet/scannet_detection_dataset.py#L31) into the actual location of data, then run:

    cd downstream

    CUDA_VISIBLE_DEVICES=x python train.py --dataset scannet --log_dir log_scannet --pretrained_path detection_scannet_pretrained.tar --num_point 40000 --use_color --no_height

You may need to set `--pretrained_path` according to the actual location of the pretrained checkpoint.

### Evaluation on ScanNet

Get our finetuned checkpoint from [here](https://drive.google.com/file/d/1RAG-USh6WuUNfKxhA-Dpojx-5-C41X62/view?usp=share_link).

To evaluate the finetuned model with its checkpoint:

    CUDA_VISIBLE_DEVICES=x python eval.py --dataset scannet --checkpoint_path detection_scannet_finetuned.tar --dump_dir eval_scannet --num_point 40000 --use_color --no_height --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

You may need to set `--checkpoint_path` according to the actual location of the finetuned checkpoint.

Example results will be dumped in the `eval_scannet` folder (or any other folder you specify). In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on axis aligned boxes. A properly finetuned model should have around 63 mAP@0.25 and 41 mAP@0.5.

<!-- ### Train and test on SUN RGB-D

To train a new VoteNet model on SUN RGB-D data (depth images):

    CUDA_VISIBLE_DEVICES=0 python train.py --dataset sunrgbd --log_dir log_sunrgbd

You can use `CUDA_VISIBLE_DEVICES=0,1,2` to specify which GPU(s) to use. Without specifying CUDA devices, the training will use all the available GPUs and train with data parallel (Note that due to I/O load, training speedup is not linear to the nubmer of GPUs used). Run `python train.py -h` to see more training options (e.g. you can also set `--model boxnet` to train with the baseline BoxNet model).
While training you can check the `log_sunrgbd/log_train.txt` file on its progress, or use the TensorBoard to see loss curves.

To test the trained model with its checkpoint:

    python eval.py --dataset sunrgbd --checkpoint_path log_sunrgbd/checkpoint.tar --dump_dir eval_sunrgbd --cluster_sampling seed_fps --use_3d_nms --use_cls_nms --per_class_proposal

Example results will be dumped in the `eval_sunrgbd` folder (or any other folder you specify). You can run `python eval.py -h` to see the full options for evaluation. After the evaluation, you can use MeshLab to visualize the predicted votes and 3D bounding boxes (select wireframe mode to view the boxes).
Final evaluation results will be printed on screen and also written in the `log_eval.txt` file under the dump directory. In default we evaluate with both AP@0.25 and AP@0.5 with 3D IoU on oriented boxes. A properly trained VoteNet should have around 57 mAP@0.25 and 32 mAP@0.5. -->

## Acknowledgements
This code is based on [VoteNet](https://github.com/facebookresearch/votenet).