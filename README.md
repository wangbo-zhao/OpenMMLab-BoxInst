


## Introduction
This repository is the code that needs to be submitted for [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021), the paper is [BoxInst: High-Performance Instance Segmentation with Box Annotations](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html)




## License

This project is released under the [Apache 2.0 license](LICENSE).



## Benchmark and model zoo

- [x] [BoxInst (CVPR'2021)](configs/boxinst)
- [ ] ConInst (ECCV'2020)

### BoxInst


| Name                                                                             | box AP | mask AP |                                  log                                 | download                                                               |
|----------------------------------------------------------------------------------|:------:|:-------:|:--------------------------------------------------------------------:|------------------------------------------------------------------------|
| [BoxInst_MS_R_50_1x](configs/boxinst/boxinst_r50_caffe_fpn_coco_mstrain_1x.py)   |  0.390 |  0.304  | [log](https://moxkl67q65.feishu.cn/file/boxcnhbdZiFdUtUbURyCILX94xf) | [model](https://moxkl67q65.feishu.cn/file/boxcnay178uhZwiYBmzRfV20TEb) |
| [BoxInst_MS_R_50_90k](configs/boxinst/boxinst_r50_caffe_fpn_coco_mstrain_90k.py) |  0.388 |  0.302  | [log](https://moxkl67q65.feishu.cn/file/boxcnmyWDlC0n1HVXadUMoMOj6d) | [model](https://moxkl67q65.feishu.cn/file/boxcnvRGKQCCvjjZAH5udD0gA9b) |
| [BoxInst_MS_R_101_90k](boxinst_r101_caffe_fpn_coco_mstrain_90k.py)               |  0.410 |  0.318  |                                   -                                  | [model](https://moxkl67q65.feishu.cn/file/boxcnNoGdGIQnwuQFzoWWXppcuh) |

Some other methods in [MMDetection](https://github.com/open-mmlab/mmdetection) are also supported.

## Getting Started

Our project is totally based on MMCV and MMDetection. Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.


### Train
Please see [doc](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#train-predefined-models-on-standard-datasets) to start training. Example,
```sheel
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 sh ./tools/dist_train.sh configs/boxinst/boxinst_r50_caffe_fpn_coco_mstrain_1x.py 4
```
please following  linear [linear scaling rule](https://arxiv.org/abs/1706.02677) to adjust batch size, learning rate and iterations.
### Inference and Eval
```sheel
python tools/test.py configs/boxinst/boxinst_r50_caffe_fpn_coco_mstrain_1x.py work_dirs/boxinst_r50_caffe_fpn_coco_mstrain_1x.py/latest.pth --eval bbox segm
```



### 🔥🔥🔥New 2021.08.21🔥🔥🔥 
[OBBDetection](https://github.com/jbwang1997/OBBDetection) is an oriented object detection toolbox based on MMdetection. What an awesome work from [jbwang1997](https://github.com/jbwang1997)!!!

## Acknowledgement

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
