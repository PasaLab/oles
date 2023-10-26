# Operation-Level Early Stopping for Robustifying Differentiable NAS 

## About

Code accompanying the paper: Operation-Level Early Stopping for Robustifying Differentiable NAS (NeurIPS 2023)

Differentiable NAS (DARTS) is a simple and efficient neural architecture search method that has been extensively adopted in various machine learning tasks. Never theless, DARTS still encounters several robustness issues, mainly the domination of skip connections. The resulting architectures are full of parametric-free operations, leading to performance collapse. Existing methods suggest that the skip connection has additional advantages in optimization compared to other parametric operations and propose to alleviate the domination of skip connections by eliminating these additional advantages. In this paper, we analyze this issue from a simple and

straightforward perspective and propose that the domination of skip connections results from parametric operations overfitting the training data while architecture parameters are trained on the validation data, leading to undesired behaviors. Based on this observation, we propose the operation-level early stopping (OLES) method to overcome this issue and robustify DARTS without introducing any computation overhead. Extensive experimental results can verify our hypothesis and the effectiveness of OLES.

This code is based on the implementation of [DARTS](https://github.com/quark0/darts) , [DrNAS](https://github.com/xiangning-chen/DrNAS) , [AGNAS](https://github.com/Sunzh1996/AGNAS) considering different search spaces。

## Architecture search

#### CIFAR-10

```
python train_search.py 
```
#### CIFAR-100

```
python train_search.py --cifar100  
```

## Architecture Evaluation

The searched architecture is in the genotype.py

```
python train.py --auxiliary --cutout --arch oles_cifar10	# cifar10
python train.py --auxiliary --cutout --arch oles_cifar100	# cifar100
python train_imagenet.py --auxiliary            # ImageNet
```

