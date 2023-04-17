# MM-3DScene: 3D Scene Understanding by Customizing Masked Modeling with Informative-Preserved Reconstruction and Self-Distilled Consistency

## Installation

Follow the [installation.md](installation.md) to install all required packages and environments.

## Data Preparation

We provide the scene data of ScanNet and S3DIS. The download link is XXX

## Training and evaluating

For detection, follow the [README](https://github.com/MingyeXu/mm-3dscene/detection/README.md) under the `detection` folder.

For segmentation, follow the [README](https://github.com/MingyeXu/mm-3dscene/blob/main/segmentation/README.md) under the `segmentation` folder.


## References

If you use this code, please cite [MM-3DScene](https://arxiv.org/pdf/2212.099484):
```
@article{xu2022mm,
  title={MM-3DScene: 3D Scene Understanding by Customizing Masked Modeling with Informative-Preserved Reconstruction and Self-Distilled Consistency},
  author={Xu, Mingye and Xu, Mutian and He, Tong and Ouyang, Wanli and Wang, Yali and Han, Xiaoguang and Qiao, Yu},
  journal={arXiv preprint arXiv:2212.09948},
  year={2022}
}
```

## [Acknowledgement]

We include the following libraries and algorithms:  
[1] [CD](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)  
[2] [PointTransformer](https://github.com/POSTECH-CVLab/point-transformer)   
[3] [VoteNet](https://github.com/facebookresearch/votenet)
