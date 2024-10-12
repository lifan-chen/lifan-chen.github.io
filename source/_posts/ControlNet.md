---
title: ControlNet
date: 2024-10-12 17:58:24
categories:
  - Note
  - AI
---

ControlNet, a neural network architecture to add spatial conditioning controls to large, pretrained text-to-image diffusion models.

## ControlNet



## ControlNet for Text-to-Image Diffusion

* Inject additional conditions into the blocks of a neural network.
  $$
  y = \cal{F}(x;\Theta),
  $$
  Suppose $\cal{F}(\cdot;\Theta)$ is a trained neural block, with parameters $\Theta$, that transforms an input feature map $x$, into another feature map $y$.

  Use two instances of zero convolutions with parameters $\Theta_{z1}$ and $\Theta{z2}$ respectively. The complete ControlNet then computes
  $$
  y_c = \cal{F}(x;\Theta)+\cal{Z}(x+Z(c;\Theta_{z1});\Theta_{z2}),
  $$
  where $y_c$ is the output of the ControlNet block.

  In the first training step, since both the weight and bias parameters of a zero convolution layer are initialized to zero, both of the $\cal{Z}(\cdot;\cdot)$ terms in the above equation to zero, and
  $$
  y_c = y.
  $$
  

  <img src="D:\blog\source\_posts\ControlNet\image-20241012190140674.png" alt="image-20241012190140674" style="zoom:80%;" />

  * *network block*: refer to a set of neural layers that are commonly put together to form a single unit of a neural network (e.g., resnet block, conv-bn-relu block, multi-head attention block, transformer block, etc)
  * *zero convolution*: 1x1 convolution with both weight and bias initialized to zero. It can protect the backbone by eliminating random noise as gradients in the initial training steps.
  * The locked parameters preserve the production-ready model trained with billion of images, while the trainable copy reuses such large-scale pretrained model to establish a deep, robust and strong backbone for handling diverse input conditions.

* 

## Training



## Inference

