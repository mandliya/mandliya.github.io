---
layout: post
title: 'Deeply Learning 1: Dropout Implementation from scratch'
date: '2026-02-17 22:37:39 '
categories: [Deeply Learning, Neural Networks]
tags: [Deep Learning, Dropout, PyTorch, Regularization, Neural Networks]
description: Deeply learning one concept at a time. In this post, we implement simple dropout from scratch.
image: /assets/img/deeply-learning-1-dropout-implementation-from-scratch/cover.png
image_alt: 'Deeply Learning 1: Dropout Implementation from scratch'
math: true
mermaid: true
pin: false
toc: true
comments: true
---

I am starting this series of blog posts to just jot down my learnings as I revisit the concepts of Deep Learning. This is just a fun way to keep learning, and committing a few posts per week to my blog. Just to start with, I'll implement a simple dropout layer from scratch and compare it with the PyTorch implementation. 

# Deeply Learning 1: Dropout Implementation from scratch

Dropout is one of the most effective regularization technniques used in Deep Learning with very simple implementation. Let's understand how it works and then implement it from scratch.

## The Core Idea

During training, we randomly drop (zero out) neurons of a layer with probability $p$ (dropout probability). The remaining neurons are scaled by the inverse of the dropout probability $1-p$ to keep the expected output the same as if the layer had not been dropped out.

$$
\begin{align}
\mathbb{E}[\text{output}] &= \mathbb{E}\left[\frac{x \cdot \text{mask}}{1-p}\right] \\
&= \frac{x \cdot \mathbb{E}[\text{mask}]}{1-p} \\
&= \frac{x \cdot (1-p)}{1-p} \\
&= x
\end{align}
$$



### Why does it work?

#### Co-Adaptation of Features

It forces the network to learn more robust and generalizable representations. Dropping units at random forces the network to avoid relying on any small set of neurons; representations become more distributed and redundant.

#### Ensemble View
With independent dropout masks, the network becomes an ensemble of all the sub-networks that can be formed by dropping out different neurons and averaging their outputs. This leads to a more robust and generalizable model.

## Implementation

During training, we scale the activations of the remaining units by the inverse of the dropout probability $1-p$, while during inference, we do nothing.

```python
import torch
import torch.nn as nn

class DropoutFromScratch(nn.Module):
  def __init__(self, p: float = 0.5, inplace: bool = False):
    super().__init__()
    self.p = p
    self.inplace = inplace

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    if not self.training or self.p == 0:
      return input

    keep_prob = 1 - self.p    
    mask = (torch.rand_like(input) < keep_prob).float().to(input.device)


    if self.inplace:
      input.mul_(mask).div_(keep_prob)
      return input
    else:
      return input * mask / keep_prob
```

```python
input = torch.ones(1000, 100)
dropout = DropoutFromScratch(p=0.5)
dropout.train()
output = dropout(input)
torch_dropout = nn.Dropout(p=0.5)
torch_dropout.train()
output_torch = torch_dropout(input)

print('Stats of output from DropoutFromScratch:')
print(f'Mean: {output.mean()}, Std: {output.std()}')
print('Stats of output from PyTorch Dropout:')
print(f'Mean: {output_torch.mean()}, Std: {output_torch.std()}')
```

    Stats of output from DropoutFromScratch:
    Mean: 0.9994800090789795, Std: 1.0000048875808716
    Stats of output from PyTorch Dropout:
    Mean: 0.9958199858665466, Std: 0.9999962449073792


## Key Takeaways
- Dropout is a simple and effective regularization technique that helps prevent overfitting by randomly dropping out neurons during training.
- It only affects the training process and does not change the inference process.
- New random masks are generated for each forward pass.
