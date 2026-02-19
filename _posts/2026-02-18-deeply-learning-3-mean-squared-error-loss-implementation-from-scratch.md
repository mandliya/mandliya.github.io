---
layout: post
title: 'Deeply Learning 3: Mean Squared Error Loss Implementation from Scratch'
date: '2026-02-18 21:31:19 '
categories: [Deeply Learning, Neural Networks]
tags: [Deep Learning, Mean Squared Error, PyTorch, Loss Functions, Neural Networks]
description: Deeply learning one concept at a time. In this post, we will implement mean squared error loss from scratch.
image: /assets/img/deeply-learning-3-mean-squared-error-loss-implementation-from-scratch/cover.png
image_alt: 'Deeply Learning 3: Mean Squared Error Loss Implementation from Scratch'
math: true
mermaid: true
pin: false
toc: true
comments: true
---

# Deeply Learning 3: Mean Squared Error Loss Implementation from Scratch

Deeply learning one concept at a time. In this post, we will implement mean squared error loss from scratch.

## Introduction
Mean squared error (MSE) loss is a commonly used loss function for regression problems. It measures the average squared difference between the predicted values and the ground truth values. The formula for MSE loss is:

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Where:
- $N$ is the number of samples
- $y_i$ is the true value for the $i$-th sample
- $\hat{y}_i$ is the predicted value for the $i$-th sample
- $\sum$ denotes the summation over all samples

## Core Concept
The core concept behind the mean squared error loss is to penalize the model for making predictions that are far from the true values. By squaring the differences, we ensure that larger errors are penalized more heavily than smaller ones. The MSE loss is smooth and differentiable, which makes it suitable for optimization using gradient descent.

## Implementation from Scratch
Let's implement the mean squared error loss from scratch as a Pytorch module. We then compare our implementation with the built-in `torch.nn.MSELoss` to ensure correctness. PyTorch also supports mean squared error loss with an optional `reduction` parameter that can be set to `'mean'`, `'sum'`, or `'none'` to specify how the loss should be aggregated across the batch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

class MSELossFromScratch(nn.Module):
  def __init__(self, reduction: str = "mean"):
    super().__init__()
    self.reduction = reduction

  def forward(
    self,
    y_pred: Float[torch.Tensor, "batch_size output_dim"],
    y_true: Float[torch.Tensor, "batch_size output_dim"]
    ) -> Float[torch.Tensor, ""]:
    """Computes the Mean Squared Error (MSE) loss between the predicted and true values."""
    # Ensure that the input tensors have the same shape
    assert y_pred.shape == y_true.shape, "Shape of y_pred and y_true must be the same"
    squared_diff = (y_pred - y_true) ** 2
    if self.reduction == "mean":
      mse_loss = squared_diff.mean()
    elif self.reduction == "sum":
      mse_loss = squared_diff.sum()
    else:
      mse_loss = squared_diff

    return mse_loss

# A simple test case to verify the implementation
torch.manual_seed(123)
y_pred = torch.randn(5, 3) # Predicted values (batch_size=5, output_dim=3)
y_true = torch.randn(5, 3) # True values (batch_size=5, output_dim=3)
mse_loss_fn = MSELossFromScratch()
loss = mse_loss_fn(y_pred, y_true)
print(f"MSE Loss: {loss.item():.4f}")

torch_loss_fn = nn.MSELoss()
torch_loss = torch_loss_fn(y_pred, y_true)
print(f"PyTorch MSE Loss: {torch_loss.item():.4f}")

mse_loss_fu_sum = MSELossFromScratch(reduction="sum")
loss_sum = mse_loss_fu_sum(y_pred, y_true)
print(f"MSE Loss (sum reduction): {loss_sum.item():.4f}")

torch_loss_sum = nn.MSELoss(reduction="sum")
torch_loss_sum_value = torch_loss_sum(y_pred, y_true)
print(f"PyTorch MSE Loss (sum reduction): {torch_loss_sum_value.item():.4f}")

```

    MSE Loss: 1.0232
    PyTorch MSE Loss: 1.0232
    MSE Loss (sum reduction): 15.3479
    PyTorch MSE Loss (sum reduction): 15.3479


## Key Takeaways
- Mean squared error loss measures the average squared difference between predicted and true values.
- It penalizes larger errors more heavily, encouraging the model to make more accurate predictions.
- The gradient of the loss with respect to the predicted values is used for backpropagation during training.
