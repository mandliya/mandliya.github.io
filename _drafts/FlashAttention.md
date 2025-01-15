## Introduction

In our previous posts on LLM inference optimization, we covered the fundamentals of transformer architecture, AI accelerators, and model architecture optimizations like GQA and MoE. We learned how attention mechanism is central to transformer architecture but comes with significant computational costs, especially for long sequences. Today, we'll dive deep into FlashAttention, a system-level optimization that revolutionizes how attention is computed on modern hardware accelerators.

## Memory Access: The Hidden Bottleneck

To understand why FlashAttention is revolutionary, we need to first understand the memory hierarchy in modern AI accelerators. In a typical GPU:

- HBM (High Bandwidth Memory): Large but relatively slow (~ TB/s)
- SRAM Cache: Small but very fast (~ PB/s)

Traditional attention implementation requires multiple HBM memory accesses:
1. Load Q, K, V matrices from HBM
2. Compute attention scores (QK^T)
3. Store attention scores to HBM
4. Load attention scores back from HBM
5. Compute softmax
6. Store softmax outputs to HBM
7. Load softmax outputs back from HBM
8. Compute final attention outputs
9. Store final outputs to HBM

This constant movement of data between HBM and SRAM creates a significant bottleneck. For example, for a sequence length of 1024 and hidden dimension of 1024, standard attention requires:
- Computing FLOPs: ~1 billion
- Memory Access: ~4 GB

The key insight of FlashAttention is that memory access, not computation, is the primary bottleneck in attention mechanisms. While previous optimization efforts focused on reducing computational complexity, FlashAttention addresses the more critical problem of memory I/O efficiency.