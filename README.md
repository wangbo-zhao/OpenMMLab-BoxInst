


## Introduction
This repository is the code that needs to be submitted for [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021), the paper is [BoxInst: High-Performance Instance Segmentation with Box Annotations](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html)




## License

This project is released under the [Apache 2.0 license](LICENSE).



## Benchmark and model zoo

- [x] [BoxInst (CVPR'2021)](configs/boxinst)
- [ ] ConInst (ECCV'2020)

###BozInst
Name | box AP | mask AP | mask AP (test-dev)| download
--- |:---:|:---:|:---:|:---:
[BoxInst_MS_R_50_1x](configs/boxinst/boxinst_r50_caffe_fpn_coco_mstrain_1x.py) | | | - | []()
[BoxInst_MS_R_50_90k](configs/boxinst/boxinst_r50_caffe_fpn_coco_mstrain_90k.py) | | | - | []()
[BoxInst_MS_R_101_90k](boxinst_r101_caffe_fpn_coco_mstrain_90k.py) | | | - | []()

Some other methods in [MMDetection](https://github.com/open-mmlab/mmdetection) are also supported.


## Getting Started

Our project is totally based on MMCV and MMDetection. Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.






## Acknowledgement

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
