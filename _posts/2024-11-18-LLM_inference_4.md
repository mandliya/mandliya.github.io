---
title: "Primer on Large Language Model (LLM) Inference Optimizations: 4. System Optimization: FlashAttention"
date: 2024-11-18 1:30:48 -0700
categories: [Large Language Model, Inference Optimization]
tags: [LLM, Inference Optimization, Transformer, Attention Mechanism, Multi-Head Attention, K-V Caching, Memory Calculation, Optimization Metrics, Optimization Techniques, Model Architecture Optimizations, FlashAttention, System Optimization]
description: "Exploring System architecture optimizations for Large Language Model (LLM) inference, focusing on FlashAttention."
image: "assets/img/inference_4/ai_gen.png"
math: true
mermaid: true
pin: false
---

**Posts in this series**:
1. [Primer on Large Language Model (LLM) Inference Optimizations: 1. Background and Problem Formulation](https://mandliya.github.io/posts/LLM_inference_1/)
2. [Primer on Large Language Model (LLM) Inference Optimizations: 2. Introduction to Artificial Intelligence (AI) Accelerators](https://mandliya.github.io/posts/LLM_inference_2/)
3. [Primer on Large Language Model (LLM) Inference Optimizations: 3. Model Architecture Optimizations](https://mandliya.github.io/posts/model_architecture_optimizations/)
4. **Primer on Large Language Model (LLM) Inference Optimizations: 4. System-level Optimization: FlashAttention** (This Post)

## Introduction

In our previous posts on LLM inference optimization, we explored the fundamentals of transformer architecture, AI accelerators, and model-level optimizations such as GQA (Grouped Query Attention) and MoE (Mixture of Experts). While these model-specific techniques are effective, they come with certain limitations. For instance, incorporating these optimizations requires retraining the model, and their applicability is restricted to specific architectural designs.

In contrast, system-level optimizations offer a more general approach. These techniques can be applied across various models without the need for retraining and are equally beneficial during training. Furthermore, system-level optimizations can complement model-level techniques to achieve even greater performance gains.

In this post, we will dive into FlashAttention, a groundbreaking system-level optimization that redefines how attention mechanisms are computed. By focusing on memory efficiency, FlashAttention demonstrates how system-level innovations can unlock significant performance improvements.

## LLM Inference: A Quick Recap

Let's revisit the inference process in a Large Language Model (LLM) to understand the challenges that FlashAttention aims to address. The inference process in an LLM can be broadly divided into two stages:

### 1. Prefill Stage
In this stage, the model processes the entire input prompt in parallel:
- Takes the complete prompt as input (e.g., "What is the capital of")
- Computes attention scores for all tokens simultaneously
- Generates and stores Key (K) and Value (V) vectors for each token
- Complexity: $O(L \cdot n \cdot d^2 + L^2 \cdot n \cdot h \cdot d)$ where L is sequence length
- The latency of this stage is usually measured by metric *TTFT (Time To First Token)*.
- The prefill stage is the most computationally intensive part of the inference process.

![Prefill Stage](/assets/img/inference_4/prefill_stage.png)

### 2. Decode Stage
This is where the model generates one token at a time:
- Uses cached K,V vectors from prefill stage
- Computes attention only for the new token
- Adds new K,V vectors to cache
- Complexity: $O(n \cdot d^2 + L \cdot n \cdot d \cdot h)$ per token
- The latency of this stage is usually measured by metric *ITL (Inter-Token Latency)*.
- The decode stage is the most memory-intensive part of the inference process.
![Decode Stage](/assets/img/inference_4/decode_stage.png)

Now that we have a clear understanding of the inference process, let's explore how FlashAttention optimizes the memory usage in the decode stage.

## Memory Access: The Hidden Bottleneck

Let's understand memory hierarchy in typical AI accelerators like GPU and their host. The memory hierarchy consists of multiple levels, each with different access times and sizes:

1. Main Memory (CPU DRAM): This the largest memory in the system but has the slowest access time. It is used to store the model parameters and intermediate results.

2. GPU Memory (HBM): This is the high-bandwidth memory on the GPU. It is faster than the CPU DRAM but has limited capacity. It is used to store the input data and intermediate results during computation. Typically, 40 GB to 80 GB in capacity and 1 TB/s to 1.5 TB/s in bandwidth.

3. GPU SRAM: This is the fastest memory on the GPU, but it has the smallest capacity. It is used to store the model parameters and intermediate results during computation. Typically, 20 MB in capacity, and 19 TB/s in bandwidth.

![Memory Hierarchy](/assets/img/inference_4/memory_hierarchy.png)

The memory hierarchy plays a crucial role in determining the performance of model inference. The key challenge is to minimize the data movement between different memory levels, as this movement incurs significant latency and energy costs.

Traditional attention mechanisms, such as the multi-head attention in transformers has the following memory access pattern:

0. Matrix Q, K and V are stored in GPU memory (HBM).
1. Load Q, K by blocks from HBM to GPU SRAM. Compute $S = Q \cdot K^T$ and write S to HBM.
2. Read S from HBM to GPU SRAM and compute $P = softmax(S)$ and write P to HBM.
3. Load P and V by blocks from HBM to GPU SRAM. Compute $O = P \cdot V$ and write O to HBM.
4. Return output O.


 FlashAttention addresses this challenge by introducing a novel memory-efficient attention mechanism.




