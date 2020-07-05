# 自定义数据集的深度学习

## step1: Load data

### 1. inherit from torch.utils.data.Dataset

### 2. __len__ ==> 数据集总体的样本数量

### 3. __getitem__ ==> 返回一个指定的样本

1. image resize
    - 224 * 224 for ResNet18
2. data argumentation
    - rotate
    - crop
3. normalize
    - mean, std
4. to tensor

## step2: build model

### 1. inherit from base class

### 2. define forward graph

## step3: train and test

## step4: transfer learning 

