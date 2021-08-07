


## Introduction
This repository is the code that needs to be submitted for [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021), the paper is [BoxInst: High-Performance Instance Segmentation with Box Annotations](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html)




## License

This project is released under the [Apache 2.0 license](LICENSE).



## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).



Supported methods:

- [x] [RPN (NeurIPS'2015)](configs/rpn)
- [x] [Fast R-CNN (ICCV'2015)](configs/fast_rcnn)



Some other methods are also supported in [projects using MMDetection](./docs/projects.md).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/customize_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.




## Acknowledgement

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
