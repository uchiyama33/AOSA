# Adaptive occlusion sensitivity analysis

This repository contains the implementation of the paper "Visually explaining 3D-CNN predictions for video classification with an adaptive occlusion sensitivity analysis"

## Requirements

- docker > 20.
- docker-compose

## Instalation

1. `$cd envs`
1. `$docker build . -t "image-name"`
1. change the image name in the docker-compose.yml to "image-name"
1. `$docker-compose up -d`
1. `$docker attach aosa`

## Example

Please refer to occlusion_sensitivity_analysis.ipynb.

If there is no enough GPU memory, please try to small "batchsize" in the example codes.

## Models and dataset utils

We use the code from the following repository for 3D-CNN models and dataset utilities. To download datasets and other resources, please refer to this repository.

[kenshohara/3D-ResNets-PyTorch: 3D ResNets for Action Recognition (CVPR 2018)](https://github.com/kenshohara/3D-ResNets-PyTorch)


## Citation

```
@article{
}
```