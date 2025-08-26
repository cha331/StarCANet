## Overview

This is the PyTorch implementation of the paper <StarCANet: A Compact and Expandable Neural Network for Massive MIMO CSI Feedback>, which has been submitted to IEEE for possible publication.

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.10
- PyTorch >= 2.6
- SciPy >= 1.15
- TensorBoard
- [thop](https://github.com/Lyken17/pytorch-OpCounter)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Dropbox](https://www.dropbox.com/scl/fo/tqhriijik2p76j7kfp9jl/h?rlkey=4r1zvjpv4lh5h4fpt7lbpus8c&e=1&st=wqinniyn&dl=0), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── StarCANet  # The cloned StarCANet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # The data folder
│   ├── DATA_Htestin.mat
│   ├── ...
├── run.sh  # The bash script
...
```

## Train StarCANet from Scratch

An example of run.sh is listed below. Simply use it with `sh run.sh`. It starts to train StarCANet from scratch. The model size can be specified as S, M, or L using the `--size` argument. Change scenario using `--scenario` and change compression ratio with `--cr`.

``` bash
python /home/StarCANet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --epochs 1000 \
  --batch-size 200 \
  --workers 8 \
  --cr 4 \
  --size 'L'
  --scheduler cosine \
  --gpu 0 \
  2>&1 | tee log.out
```

## Results and Reproduction


**To reproduce the results, simple add `--evaluate` to `run.sh` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` bash
python /home/StarCANet/main.py \
  --data-dir '/home/COST2100' \
  --scenario 'in' \
  --pretrained './checkpoints/in_4.pth' \
  --evaluate \
  --batch-size 200 \
  --workers 0 \
  --cr 4 \
  --cpu \
  2>&1 | tee test_log.out

```

## Acknowledgment

This repository is modified from the [CRNet open source code](https://github.com/Kylin9511/CRNet). Thanks Zhilin for his amazing work.
Thanks Chao-Kai Wen and Shi Jin group for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet) 

