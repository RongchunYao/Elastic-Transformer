# I REWRITE SOME CODE SO USAGE IS INVALID. I WILL UPDATE TOMORROW
# Adaptive Inference for Transformer
This repository is a simple demo for CPU Adaptive Inference for both pytorch and onnx runtime web. We currently use pretrained vit_base_patch16_224 from timm as the demo and haven't extend support to other models.  

## Usage

### Requirements

- torch 
- torchvision
- timm

**Data preparation**: 

Download and extract ImageNet 2012 validation dataset images from http://image-net.org/. 

You could download the pre-process script from https://github.com/soumith/imagenetloader.torch/blob/master/valprep.sh or somewhere else you like. Put the pre-process script in the directory where validation dataset is and run.

### Pytorch Demo
First, to generate profiling file run: 
```
python path/to/profiling.py
```
Run dynamic inference:
```
python path/to/run.py
```
To see option meanings:
```
python path/to/profiling.py --help
python path/to/run.py --help
```
You don't have to cd into directory where layer_timing.py or run.py is.


### Onnx Web Demo
In-progress


