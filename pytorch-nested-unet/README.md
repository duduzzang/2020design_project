# PyTorch implementation of UNet++ (Nested U-Net)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

This repository contains code for a image segmentation model based on [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165) implemented in PyTorch.

## Requirements
- PyTorch 1.x or 0.41

## Installation
1. Create an anaconda environment.
```sh
conda create -n=<env_name> python=3.6 anaconda
conda activate <env_name>
```
2. Install PyTorch.
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
3. Install pip packages.
```sh
pip install -r requirements.txt
```

## Training 
1. Preprocess.
```sh
python data_preprocessing.py
```
2. After data_preprocessing
```
inputs
└── <dataset name>
    ├── images
    |   ├── 0a7e06.png
    │   ├── 0aab0a.png
    │   ├── 0b1761.png
    │   ├── ...
    |
    └── masks
        └── 0
            ├── 0a7e06.png
            ├── 0aab0a.png
            ├── 0b1761.png
            ├── ...
        
```
3. Train the model.
```sh
python train.py --dataset dsb2018_96 --arch NestedUNet
```
4. Evaluate.
```sh
python test.py --name dsb2018_96_NestedUNet_woDS
```
