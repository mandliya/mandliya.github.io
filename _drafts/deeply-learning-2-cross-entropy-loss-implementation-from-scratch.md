---
layout: post
title: 'Deeply Learning 2: Cross Entropy Loss Implementation from scratch'
date: '2026-02-17 23:15:35 '
categories:
- Technology
tags:
- Jupyter
- Notebook
description: Deeply learning one concept at a time. In this post, we implement cross
  entropy loss from scratch.
image: /assets/img/deeply-learning-2-cross-entropy-loss-implementation-from-scratch/cover.png
image_alt: 'Deeply Learning 2: Cross Entropy Loss Implementation from scratch'
math: true
mermaid: true
pin: false
toc: true
comments: true
---

# Deeply Learning 2: Cross Entropy Loss Implementation from scratch
Deeply learning one concept at a time. In this post, we implement cross entropy loss from scratch.

## Introduction
Cross entropy loss is one of the most commonly used loss functions in classification problems. It measures the difference between two probability distributions: the true distribution (from the ground truth labels) and the predicted distribution (from the model's output).

## The Core Idea
The cross entropy loss for a single sample can be defined as:
$$
L = -\sum_{i=1}^{C} y_i \log(p_i)
$$
Where:
- $C$ is the number of classes.
- $y_i$ is the true label for class $i$ (1 if the sample belongs to class $i$, otherwise 0).
- $p_i$ is the predicted probability for class $i$.

Why log(p)? Idea is to capture how surprised we are when the model makes a wrong prediction. The logarithm here is used to penalize incorrect predictions more heavily. If the model predicts 99% probability for the correct class, resulting in low surprise, the loss will be low. However, if the model predicts a low probability for the correct class, the loss will be high, indicating a high level of surprise. This encourages that gradient flow will be stronger when the model is making incorrect predictions, which helps in faster learning.
$$
\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}
$$
This gradient is used during backpropagation to update the model's parameters. 

## Implementation from Scratch
Let's implement the cross entropy loss from scratch using just basic Pytorch operations.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

class CrossEntropyLossFromScratch(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(
    self,
    logits: Int[Tensor, "batch_size num_classes"],
    targets: Int[Tensor, "batch_size"]
    ) -> Float[Tensor, ""]:
    batch_size, num_classes = logits.shape
    assert targets.shape == (batch_size,)

    max_logits, _ = torch.max(logits, dim=1, keepdim=True) #[batch_size, 1]
    logits = logits - max_logits # stabize the logits #[batch_size, num_classes]
    sum_exp_logits = torch.sum(torch.exp(logits), dim=1, keepdim=True) #[batch_size, 1]
    log_probs = logits - torch.log(sum_exp_logits) #[batch_size, num_classes]
    loss = -log_probs[torch.arange(batch_size), targets] #[batch_size]
    return loss.mean()

```

```python
# a random logit tensor and target tensor for testing
torch.manual_seed(0)
logits = torch.randn(4, 3) # [batch_size, num_classes]
targets = torch.tensor([0, 1, 2, 1]) # [batch_size]

criterion = CrossEntropyLossFromScratch()
loss = criterion(logits, targets)
print(f"Cross Entropy Loss: {loss.item():.4f}")

torch_loss = F.cross_entropy(logits, targets)
print(f"PyTorch Cross Entropy Loss: {torch_loss.item():.4f}")

print(f'Losses are close: {torch.isclose(loss, torch_loss)}')
```

    Cross Entropy Loss: 1.4412
    PyTorch Cross Entropy Loss: 1.4412
    Losses are close: True


## Key Takeaways
- Cross entropy loss captures the difference between true and predicted distributions.
- It penalizes incorrect predictions more heavily, encouraging the model to learn faster.
- The gradient of the loss with respect to the predicted probabilities is used for backpropagation
