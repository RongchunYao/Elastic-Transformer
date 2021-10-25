# Adaptive Inference for Transformer
This repository is a simple demo for CPU Adaptive Inference for both pytorch and onnx runtime web. We currently use pretrained vit_base_patch16_224 from timm as the demo and haven't extend support to other models.  

## Usage

### Requirements

- torch 
- torchvision
- timm

**Data preparation**: 

Download and extract ImageNet 2012 validation dataset images from http://image-net.org/. 

You could download the pre-process script from https://github.com/soumith/imagenetloader.torch/blob/master/valprep.sh or somewhere else you like. Put the pre-process script in the directory where validation dataset is and run. You could put the dataset folder any where in the project directory, if the folder name is not 'ILSVRC2012', you need to specify the path to the folder or folder name, for example
```
python -m ViT.run --dataset=DIR_NAME_OR_PATH
```
You do not need to download dataset if you just want to do the profiling to see the potential of gaining the time reduction. Just run command below for cpu
```
python -m ViT.profiling.profiling --device=cpu 
```
and command below for gpu 
```
python -m ViT.profiling.profiling --device=cuda
```
### Pytorch Demo
First, to generate profiling file run: 
```
python -m ViT.profiling.profiling
```
Run dynamic inference:
```
python -m ViT.run
```
To see option meanings:
```
python -m profiling.profiling --help
python -m ViT.run --help
```



### Onnx Web Demo
In-progress


