# EVC-Net: A Hybrid Deep Learning Network for Breast Cancer Classification

This repository contains the PyTorch implementation of the EVC-Net architecture as presented in the paper "EVC-Net: A Hybrid Deep Learning Network for Breast Cancer Classification from Histopathological Images".

## Overview
EVC-Net is a hybrid deep learning framework designed to automate the classification of breast cancer from histopathological images. The model integrates three distinct components:
1. **EfficientNetV2S**: Extracts fine-grained local texture features.
2. **Vision Transformer (ViT)**: Captures global context and long-range dependencies using multi-head self-attention.
3. **Capsule Network**: Preserves spatial hierarchies within tissue structures using dynamic routing.

## Requirements
* Python 3.8+
* PyTorch >= 2.0.0
* torchvision >= 0.15.0
* timm >= 0.9.0

Install the dependencies using:
```bash

pip install -r requirements.txt


## Usage

The primary model definition is located in `model.py`. The `EVCNet` class requires the definition of the number of target classes (e.g., `num_classes=2` for binary classification and `num_classes=8` for multi-class).

### Training

Execute `train.py` to initiate the training loop. The script includes automated mixed-precision training, early stopping, and a learning rate scheduler (`ReduceLROnPlateau`). Ensure your dataset is formatted as a standard PyTorch `DataLoader` before execution.

```bash
python train.py

```

## Citation

If you utilize this code or model in your research, please cite our paper accordingly.

```
