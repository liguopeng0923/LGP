## Prerequisites

- Linux & Windows
- TorchVision: 0.13.1
- OpenCv: 4.8.0
- Python 3.7+
- PyTorch 1.6+
- CUDA 9.2+
- GCC 5+
- MMCV: 1.7.1
- MMCV Compiler: GCC 9.3
- MMCV CUDA Compiler: 11.3
- MMDetection: 2.26.0+
- MMRotate: 0.3.2+


## Installation

### Prepare environment

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n LGP python=3.7 -y
    conda activate LGP
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.7, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch
    ```

### Install MMdet and MMRotate

It is recommended to install MMRotate with [MIM](https://github.com/open-mmlab/mim),
which automatically handle the dependencies of OpenMMLab projects, including mmcv and other python packages.

```shell
pip install openmim
mim install mmcv-full==1.6.0
mim install mmdet==2.26.0
mim install mmrotate==0.3.2

pip install piq==0.7.0
```

You can find details in these docs:

- [mmdet](https://github.com/liguopeng0923/LGP/blob/main/mmdet/docs/en/install.md) 2.26.0+

- [mmrotate](https://github.com/liguopeng0923/LGP/blob/main/mmrotate/docs/en/install.md)0.3.2+


