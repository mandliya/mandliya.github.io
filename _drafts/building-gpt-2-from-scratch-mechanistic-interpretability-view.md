---
layout: post
title: 'Building GPT-2 From Scratch:  Mechanistic Interpretability View'
date: '2026-01-11 06:56:44 '
categories:
- Technology
tags:
- Jupyter
- Notebook
description: In this post, we're going to build GPT-2 from the ground up fun and learning,
  implementing every component ourselves and understanding exactly how thi...
image: /assets/img/building-gpt-2-from-scratch-mechanistic-interpretability-view/cover.png
image_alt: 'Building GPT-2 From Scratch:  Mechanistic Interpretability View'
math: true
mermaid: true
pin: false
toc: true
comments: true
---

# Building GPT-2 From Scratch:  Mechanistic Interpretability View

In this post, we're going to build GPT-2 from the ground up fun and learning, implementing every component ourselves and understanding exactly how this remarkable architecture works. This is the first part of a two-part series. Here, we'll focus on understanding and implementing the architecture, loading pre-trained weights, and running inference. **In the next post, we'll train our implementation from scratch** and explore the training dynamics.

We’ll build the model as a stack of simple, testable building blocks (LayerNorm → Embeddings → Attention → MLP → Transformer blocks → Unembedding). We’ll use the **“Anthropic / mechanistic interpretability” [view](https://transformer-circuits.pub/2021/framework/index.html)** of a transformer: keep attention heads explicit (separate $W_Q$, $W_K$, $W_V$, $W_O$ per head) because it makes later analysis much easier (patching, head attribution, interventions, etc.). Finally we’ll validate correctness by **loading GPT‑2 Small weights** from Neel Nanda’s reference [implementation](https://www.youtube.com/watch?v=bOYE6E8JrtU&list=PL7m7hLIqA0hoIUPhC26ASCVs_VrqcDpAz) (via [EasyTransformer](https://github.com/redwoodresearch/Easy-Transformer)) and checking that our model predicts the same next tokens. 


Here’s a clean mental model for what we’re building (residual stream + attention + MLP as “writers” into the residual stream):

![transformer architecture from Anthropic - IMAGE NOT FOUND](../assets/img/building-gpt-2-from-scratch-mechanistic-interpretability-view/transformer_arch.png)


In the a future post, we will actually train this model from scratch and explore various training optimizations.


# Setup

We’ll install a couple of dependencies, import everything we need, and pick a device (`cuda` if available).  
If you’re running this locally, make sure you have a recent PyTorch build and enough VRAM for GPT‑2 Small.


```python
# install what is missing
%pip install -q git+https://github.com/neelnanda-io/Easy-Transformer.git@clean-transformer-demo
# Install another version of node that makes PySvelte work way faster
!curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash - >/dev/null;
!sudo apt-get install -y nodejs -qq >/dev/null
%pip install -q git+https://github.com/neelnanda-io/PySvelte.git
%pip install -q fancy_einsum
%pip install -q einops
```

      Preparing metadata (setup.py) ... [?25l[?25hdone
      Building wheel for easy_transformer (setup.py) ... [?25l[?25hdone
    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    
    W: Skipping acquire of configured file 'main/source/Sources' as repository 'https://r2u.stat.illinois.edu/ubuntu jammy InRelease' does not seem to provide it (sources.list entry misspelt?)
    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    
    
    WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
    
    W: Skipping acquire of configured file 'main/source/Sources' as repository 'https://r2u.stat.illinois.edu/ubuntu jammy InRelease' does not seem to provide it (sources.list entry misspelt?)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 78, <> line 1.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
      Preparing metadata (setup.py) ... [?25l[?25hdone
      Building wheel for PySvelte (setup.py) ... [?25l[?25hdone
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    inflect 7.5.0 requires typeguard>=4.0.1, but you have typeguard 2.13.3 which is incompatible.[0m[31m
    [0m

```python
# imports
import einops
from fancy_einsum import einsum
import torch
import torch.nn as nn
import math
from tqdm.auto import tqdm
from pprint import pprint
from dataclasses import dataclass
```

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Reference model (for weight loading + sanity checks)

As we build each component, it’s easy to accidentally get one small detail wrong (a transpose, a broadcast, a norm, a bias).

To keep ourselves honest, we’ll use **Neel Nanda’s `EasyTransformer` GPT‑2 Small** as a reference:

- We’ll **load its weights into our implementation** (same parameter names / shapes).
- We’ll run a few prompts and verify that **the next-token predictions match**.

```python
from easy_transformer import EasyTransformer #to compare our implementation with GPT-2
from easy_transformer.utils import get_corner, gelu_new, tokenize_and_concatenate

reference_gpt2 = EasyTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False)
```

    Moving model to device:  cuda
    Finished loading pretrained model gpt2-small into EasyTransformer!


```python
reference_text = "I live in France, and I speak"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens)
log_probs = logits.log_softmax(dim=-1)
next_token = logits[0, -1].argmax(dim=-1)
print(reference_gpt2.tokenizer.decode(next_token))

```

     French


## Config

GPT‑2 has a very specific set of hyperparameters (embedding size, number of heads, MLP expansion, context length, etc.).

We’ll define a `Config` dataclass that mirrors the GPT‑2 Small configuration and then reuse `cfg` everywhere.  
Keeping these names consistent is also what makes weight loading painless later.

We're implementing GPT-2 Small, which has:
- **12 layers** (transformer blocks)
- **12 attention heads** per layer
- **768-dimensional** embeddings
- **50,257 token** vocabulary
- **1024 token** context window



```python
@dataclass
class Config:
  d_model: int = 768 #embedding size or residual stream size
  n_layer: int = 12
  n_head: int = 12 # number of attention head
  d_mlp: int = 3072 # standard d_model x 4
  n_ctx: int = 1024 # max sequence length or block size
  layer_norm_eps: int = 1e-5
  init_range: float = 0.02
  debug: bool = True
  d_vocab: int = 50257
  d_head: int = 64
```

```python
# some utils to print and test stuff

def rand_float_test(cls, shape):
  cfg = Config(debug=True)
  layer = cls(cfg).to(device)
  random_input = torch.randn(shape).to(device)
  print(f"{random_input.shape=}")
  output = layer(random_input)
  print(f"{output.shape=}")

def rand_int_test(cls, shape):
  cfg = Config(debug=True)
  layer = cls(cfg).to(device)
  random_input = torch.randint(100, 1000, shape).to(device)
  print(f"{random_input.shape=}")
  output = layer(random_input)
  print(f"{output.shape=}")

def load_gpt2_test(cls, gpt2_layer, input_name, cache_dict=cache.cache_dict):
  cfg = Config(debug=True)
  layer = cls(cfg).to(device)
  layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
  if isinstance(input_name, str):
    reference_input = cache_dict[input_name]
  else:
    reference_input = input_name

  reference_output = gpt2_layer(reference_input)
  output = layer(reference_input)
  print(f"{reference_input.shape=}")
  print(f"{reference_output.shape=}")

  comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
  print(f"{comparison.sum() / comparison.numel():.2%} of the values match")
```

```python
for activation_name, activation in cache.cache_dict.items():
  print(f"{activation_name}: {activation.shape=}")
```

    hook_embed: activation.shape=torch.Size([1, 9, 768])
    hook_pos_embed: activation.shape=torch.Size([1, 9, 768])
    blocks.0.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.0.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.0.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.0.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.0.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.0.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.0.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.0.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.0.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.0.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.0.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.0.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.0.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.0.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.0.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.0.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.0.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.1.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.1.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.1.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.1.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.1.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.1.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.1.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.1.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.1.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.1.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.1.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.1.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.1.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.1.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.1.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.1.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.1.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.2.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.2.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.2.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.2.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.2.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.2.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.2.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.2.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.2.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.2.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.2.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.2.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.2.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.2.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.2.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.2.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.2.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.3.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.3.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.3.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.3.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.3.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.3.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.3.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.3.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.3.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.3.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.3.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.3.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.3.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.3.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.3.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.3.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.3.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.4.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.4.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.4.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.4.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.4.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.4.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.4.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.4.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.4.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.4.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.4.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.4.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.4.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.4.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.4.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.4.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.4.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.5.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.5.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.5.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.5.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.5.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.5.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.5.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.5.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.5.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.5.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.5.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.5.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.5.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.5.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.5.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.5.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.5.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.6.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.6.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.6.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.6.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.6.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.6.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.6.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.6.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.6.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.6.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.6.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.6.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.6.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.6.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.6.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.6.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.6.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.7.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.7.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.7.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.7.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.7.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.7.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.7.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.7.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.7.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.7.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.7.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.7.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.7.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.7.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.7.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.7.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.7.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.8.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.8.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.8.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.8.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.8.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.8.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.8.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.8.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.8.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.8.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.8.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.8.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.8.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.8.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.8.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.8.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.8.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.9.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.9.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.9.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.9.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.9.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.9.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.9.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.9.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.9.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.9.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.9.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.9.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.9.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.9.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.9.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.9.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.9.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.10.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.10.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.10.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.10.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.10.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.10.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.10.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.10.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.10.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.10.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.10.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.10.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.10.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.10.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.10.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.10.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.10.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    blocks.11.hook_resid_pre: activation.shape=torch.Size([1, 9, 768])
    blocks.11.ln1.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.11.ln1.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.11.attn.hook_q: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.11.attn.hook_k: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.11.attn.hook_v: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.11.attn.hook_attn_scores: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.11.attn.hook_attn: activation.shape=torch.Size([1, 12, 9, 9])
    blocks.11.attn.hook_z: activation.shape=torch.Size([1, 9, 12, 64])
    blocks.11.hook_attn_out: activation.shape=torch.Size([1, 9, 768])
    blocks.11.hook_resid_mid: activation.shape=torch.Size([1, 9, 768])
    blocks.11.ln2.hook_scale: activation.shape=torch.Size([1, 9, 1])
    blocks.11.ln2.hook_normalized: activation.shape=torch.Size([1, 9, 768])
    blocks.11.mlp.hook_pre: activation.shape=torch.Size([1, 9, 3072])
    blocks.11.mlp.hook_post: activation.shape=torch.Size([1, 9, 3072])
    blocks.11.hook_mlp_out: activation.shape=torch.Size([1, 9, 768])
    blocks.11.hook_resid_post: activation.shape=torch.Size([1, 9, 768])
    ln_final.hook_scale: activation.shape=torch.Size([1, 9, 1])
    ln_final.hook_normalized: activation.shape=torch.Size([1, 9, 768])


## Building Block 1: LayerNorm

LayerNorm is our first component. GPT-2 uses **pre-layer norm**, meaning we normalize *before* each attention and MLP sublayer (not after, as in the original Transformer paper).

### Why LayerNorm Matters

LayerNorm stabilizes training by normalizing activations to have mean 0 and variance 1 across the feature dimension. Unlike BatchNorm (which normalizes across examples), LayerNorm normalizes each example independently:

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu$ and $\sigma^2$ are computed across the feature dimension (d_model)
- $\gamma$ and $\beta$ are learned scale and shift parameters
- $\epsilon$ prevents division by zero

This is crucial for deep networks — without it, activations can explode or vanish as they flow through layers.


```python
class LayerNorm(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.w = nn.Parameter(torch.ones(cfg.d_model))
    self.b = nn.Parameter(torch.zeros(cfg.d_model))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x.size = [batch, seq_len, d_model]
    mean = einops.reduce(x, "batch seq_len d_model -> batch seq_len 1", "mean")
    x_mean_diff = x - mean
    var = einops.reduce(x_mean_diff.pow(2), "batch seq_len d_model -> batch seq_len 1", "mean")
    std = torch.sqrt(var + self.cfg.layer_norm_eps)
    normalized_x = x_mean_diff / std
    normalized_x = normalized_x * self.w + self.b
    return normalized_x

```

```python
_ = rand_float_test(LayerNorm, [2, 4, 768])
_ = load_gpt2_test(LayerNorm, reference_gpt2.ln_final, "blocks.11.hook_resid_post")
```

    random_input.shape=torch.Size([2, 4, 768])
    output.shape=torch.Size([2, 4, 768])
    reference_input.shape=torch.Size([1, 9, 768])
    reference_output.shape=torch.Size([1, 9, 768])
    100.00% of the values match


## Building Block 2: Token Embedding

A language model starts with token IDs like:

$$
[15496,\ 318,\ 257,\ 1332,\ \dots]
$$

We need to convert discrete tokens (integers from 0 to 50,256) into continuous vectors that the model can process.
We map each token ID to a learned vector in $\mathbb{R}^{d_{model}}$ using an embedding matrix:

$$
W_E \in \mathbb{R}^{d_{vocab} \times d_{model}}
$$

Indexing into this table gives you token embeddings:

$$
x_{tok} = W_E[\text{token\_id}]
$$

Yes, PyTorch has `nn.Embedding`  but implementing it ourselves keeps the logic transparent and the parameter naming consistent with the reference model.


```python
class Embedding(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.W_E = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
    nn.init.normal_(self.W_E, std=cfg.init_range)
    # works like nn.Linear(cfg.d_vocab, cfg.d_model)

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    #[batch, seq_len] -> [batch, seq_len, d_model]
    embed = self.W_E[tokens, :]
    return embed

```

```python
rand_int_test(Embedding, [2, 4])
load_gpt2_test(Embedding, reference_gpt2.embed, tokens)
```

    random_input.shape=torch.Size([2, 4])
    output.shape=torch.Size([2, 4, 768])
    reference_input.shape=torch.Size([1, 9])
    reference_output.shape=torch.Size([1, 9, 768])
    100.00% of the values match


## Building Block 3: Positional Embedding

Self-attention is *content-based* and *position agnostic*. It doesn’t inherently know whether a token is the 3rd token or the 300th token.

GPT‑2 solves this with **learned positional embeddings**. For each position $p \in [0, \dots, n_{ctx}-1]$, we learn:

$$
W_{pos}[p] \in \mathbb{R}^{d_{model}}
$$

The model input at each position becomes:

$$
x = x_{tok} + x_{pos}
$$

This is the simplest approach (and it works surprisingly well). Later architectures add rotary embeddings (RoPE), ALiBi, etc., but GPT‑2 Small is learned absolute positions.


```python
class PosEmbedding(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model))
    nn.init.normal_(self.W_pos, std=cfg.init_range)

  def forward(self, tokens: torch.Tensor):
    pos_embeds = self.W_pos[:tokens.size(1), :]
    pos_embeds = einops.repeat(
        pos_embeds,
        "seq_len d_model -> batch seq_len d_model",
        batch=tokens.size(0)
    )
    return pos_embeds
```

```python
_ = rand_int_test(PosEmbedding, [3, 4])
_ = load_gpt2_test(PosEmbedding, reference_gpt2.pos_embed, tokens)
```

    random_input.shape=torch.Size([3, 4])
    output.shape=torch.Size([3, 4, 768])
    reference_input.shape=torch.Size([1, 9])
    reference_output.shape=torch.Size([1, 9, 768])
    100.00% of the values match


## Building Block 4: Causal Self‑Attention

This is where things get interesting. Attention is the mechanism that allows information to flow between token positions. It's what makes transformers so powerful.

### The Attention Intuition

Imagine you're reading the sentence: "The trophy doesn't fit in the suitcase because **it** is too large." To understand what "it" refers to, you need to look back at previous words. Attention does exactly this — it lets each token "attend to" other tokens in the sequence. If you are less familiar with attention, I definitely recommend this Andrej Karpathy's [video](https://www.youtube.com/watch?v=kCc8FmEb1nY), I have also another post on visualizing attention [here](https://mandliya.github.io/posts/visualizing-attention-see-what-an-llm-sees/).

### Multi-Head Attention Mechanics

For each token, we:
1. **Create queries, keys, and values**: Linear projections of the input
2. **Compute attention scores**: Query·Key^T tells us "how much to attend"
3. **Apply causal mask**: Prevent looking at future tokens (this makes it autoregressive)
4. **Softmax and weighted sum**: Attention weights × Values gives us the output

GPT-2 uses **multi-head attention** — 12 heads in parallel, each learning different patterns. Some heads might learn syntax, others semantics, others positional relationships.

### Implementation Choice: Separate Q, K, V

Our key implementation difference from standard libraries: we use separate weight matrices `W_Q`, `W_K`, `W_V` instead of a combined `c_attn`. This makes it trivial to:
- Analyze what each head attends to
- Intervene on specific attention patterns
- Understand information routing

The formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$



```python
class CausalAttention(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    assert cfg.d_model % cfg.n_head == 0
    self.d_head = cfg.d_model // cfg.n_head
    self.n_head = cfg.n_head
    self.d_model = cfg.d_model
    self.cfg = cfg
    self.W_Q = nn.Parameter(torch.randn(self.n_head, self.d_model, self.d_head))
    self.W_K = nn.Parameter(torch.randn(self.n_head, self.d_model, self.d_head))
    self.W_V = nn.Parameter(torch.randn(self.n_head, self.d_model, self.d_head))
    self.W_O = nn.Parameter(torch.randn(self.n_head, self.d_head, self.d_model))
    self.b_Q = nn.Parameter(torch.zeros(self.n_head, self.d_head))
    self.b_K = nn.Parameter(torch.zeros(self.n_head, self.d_head))
    self.b_V = nn.Parameter(torch.zeros(self.n_head, self.d_head))
    self.b_O = nn.Parameter(torch.zeros(self.d_model))
    self.register_buffer('IGNORE', torch.tensor(-1e10, dtype=torch.float32, device=device))

  def apply_causal_mask(self, attn: torch.Tensor) -> torch.Tensor:
    #[b, n_head, query_pos, key_pos]
    seq_len = attn.size(-2)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1
    ).bool()
    attn.masked_fill_(mask, self.IGNORE)
    return attn

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    #[batch, seq_len d_model] -> [batch seq_len d_model]
    q = einsum("batch query_pos d_model, n_head d_model d_head -> batch query_pos n_head d_head",
               input, self.W_Q) + self.b_Q
    k = einsum("batch key_pos d_model, n_head d_model d_head -> batch key_pos n_head d_head",
               input, self.W_K) + self.b_K
    v = einsum("batch key_pos d_model, n_head d_model d_head -> batch key_pos n_head d_head",
               input, self.W_V) + self.b_V

    attn = einsum(
        "batch query_pos n_head d_head, batch key_pos n_head d_head -> batch n_head query_pos key_pos",
        q, k)
    attn = attn / math.sqrt(self.d_head)
    attn = self.apply_causal_mask(attn)
    attn = attn.softmax(dim=-1)
    z = einsum(
        "batch n_head query_pos key_pos, batch key_pos n_head d_head -> batch query_pos n_head d_head",
        attn, v
    )

    out = einsum(
        "batch query_pos n_head d_head, n_head d_head d_model -> batch query_pos d_model",
        z, self.W_O
    ) + self.b_O

    return out
```

```python
_ = rand_float_test(CausalAttention, [3, 4, 768])
_ = load_gpt2_test(CausalAttention, reference_gpt2.blocks[0].attn, cache["blocks.0.ln1.hook_normalized"])
```

    random_input.shape=torch.Size([3, 4, 768])
    output.shape=torch.Size([3, 4, 768])
    reference_input.shape=torch.Size([1, 9, 768])
    reference_output.shape=torch.Size([1, 9, 768])
    100.00% of the values match


## Building Block 5: MLP

While attention *routes* information between positions, the MLP (Multi-Layer Perceptron) is where the actual *computation* happens. Think of it as: attention moves things around, MLP processes them.

GPT‑2’s MLP is a 2-layer feed-forward network applied independently at each token position:

$$
\text{MLP}(x) = W_{out}\,\text{GELU}(W_{in}x + b_{in}) + b_{out}
$$

### Architecture

The MLP is refreshingly simple:
1. **Expand**: Linear layer projects from d_model (768) to 4×d_model (3072)
2. **Activate**: GELU activation adds non-linearity
3. **Contract**: Linear layer projects back down to d_model (768)

### Why 4×?

The 4× expansion is empirically determined. The intermediate layer needs to be larger than the input/output to increase model capacity. The current architecture runs:

`768 → 3072 → 768`

This MLP is applied identically and independently to each position, i.e. it's a position-wise feed-forward network.

### GELU vs ReLU

GPT-2 uses GELU (Gaussian Error Linear Unit) instead of ReLU. GELU is smoother and provides better gradients, especially important for the large-scale pre-training regime.

Key details:
- It expands dimensionality from $d_{model}$ to $d_{mlp}$ (typically 4×).
- Uses **GELU** activation (smoother than ReLU; works well in transformers).
- Runs per position (no mixing across tokens inside the MLP — that’s attention’s job).

This is where a lot of the model’s capacity lives.


```python
class MLP(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.W_in = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp))
    self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
    self.gelu = nn.GELU()
    self.W_out = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model))
    self.b_out = nn.Parameter(torch.randn(cfg.d_model))

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    z = einsum(
        "batch seq_len d_model, d_model d_mlp -> batch seq_len d_mlp",
        input, self.W_in
    ) + self.b_in
    z = self.gelu(z)
    out = einsum(
        "batch seq_len d_mlp, d_mlp d_model -> batch seq_len d_model",
        z, self.W_out
    ) + self.b_out
    return out

```

```python
_ = rand_float_test(CausalAttention, [3, 4, 768])
```

    random_input.shape=torch.Size([3, 4, 768])
    output.shape=torch.Size([3, 4, 768])


## Building Block 6: Transformer Block

Now we assemble the pieces into a complete transformer block. This is the fundamental repeating unit of GPT-2 - we'll stack 12 of these.

### The Architecture Pattern

Schematically (for residual stream $r$):

$$
r \leftarrow r + \text{Attn}(\text{LN}(r))
$$
$$
r \leftarrow r + \text{MLP}(\text{LN}(r))
$$

### Why Residual Connections?

The $r + \dots$ is crucial. Residual connections (skip connections) allow gradients to flow directly through the network during training. Without them, deep networks would be nearly impossible to train. They also provide a conceptual benefit: each block can be viewed as applying a *refinement* or *correction* to the representation, rather than transforming it completely.

### Pre-Norm vs Post-Norm

We normalize *before* each sublayer (pre-norm). The original Transformer paper normalized after, but pre-norm has become standard because it stabilizes training for deeper models.
The residual stream is the main state that flows through the network. Attention and MLP are “writers” that propose updates; the residual connection keeps information moving forward without being overwritten.

Once we have a correct block, building the full model is just stacking.


```python
class TransformerBlock(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.ln1 = LayerNorm(cfg)
    self.attn = CausalAttention(cfg)
    self.ln2 = LayerNorm(cfg)
    self.mlp = MLP(cfg)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x
```

```python
_ = rand_float_test(TransformerBlock, [3, 4, 768])
```

    random_input.shape=torch.Size([3, 4, 768])
    output.shape=torch.Size([3, 4, 768])


## Building Block 7: Unembedding

After processing through 12 transformer blocks, we have rich representations for each token position. But we need to convert these back into predictions over the vocabulary.

### The Unembedding Matrix

The unembedding layer is a linear projection from $d_{model} (768)$ to $d_{vocab} (50,257)$. For each position, this gives us logits, a raw scores for each possible next token.
$$
W_U \in \mathbb{R}^{d_{model} \times d_{vocab}}
$$


Logits for the next token are:

$$
\text{logits} = r_{final} W_U
$$


### Weight Tying (Not used in this implementation)

Some models tie the embedding and unembedding weights (they're transposes of each other). Original GPT-2 uses it but Neel's implementation doesn't. This gives more flexibility but costs more parameters. We don't do it in this implementation.

At the end, we need to map the final residual stream back into vocabulary logits. 

```python
class Unembedding(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.W_U = nn.Parameter(torch.randn(cfg.d_model, cfg.d_vocab))
    self.b_U = nn.Parameter(torch.zeros(cfg.d_vocab), requires_grad=False)

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    out = einsum(
        "batch seq_len d_model, d_model d_vocab -> batch seq_len d_vocab",
        input, self.W_U
    ) + self.b_U

    return out
```

## Putting It All Together: The Complete GPT-2

Now for the exciting part — we assemble all our building blocks into the complete transformer!

### The Full Forward Pass

Here's what happens when we feed in a sequence of tokens:

1. **Embed**: Convert tokens to vectors and add positional encodings
2. **Process**: Pass through 12 transformer blocks sequentially
3. **Normalize**: Apply final LayerNorm
4. **Unembed**: Project to vocabulary logits

The beauty is that each block refines the representation, building up increasingly abstract and context-aware features.

### Autoregressive Generation

At inference time, GPT-2 generates one token at a time:
- Feed in "The cat"
- Get logits, sample the next token (say "sat")
- Feed in "The cat sat"
- Get next token ("on")
- And so on...

This is why the causal mask is so important, we can't peek at future tokens during generation because they don't exist yet!

```python
class GPT2Transformer(nn.Module):
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
    self.embed = Embedding(cfg)
    self.pos_embed = PosEmbedding(cfg)
    self.blocks = nn.ModuleList([
        TransformerBlock(cfg)
        for _ in range(cfg.n_layer)
    ])
    self.ln_final = LayerNorm(cfg)
    self.unembed = Unembedding(cfg)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    emd = self.embed(x)
    pos_emd = self.pos_embed(x)
    residual = emd + pos_emd
    for block in self.blocks:
      residual = block(residual)

    normalized_residual = self.ln_final(residual)
    logits = self.unembed(normalized_residual)
    return logits

```

## The Moment of Truth: Loading Pre-trained Weights

Time to test if our implementation is correct! We'll load the pre-trained GPT-2 weights from our reference model into our custom implementation.

If we've built everything correctly, our model should:
1. Accept the weights without shape mismatches
2. Generate coherent text
3. Produce identical outputs to the reference model

Let's instantiate our model and load the weights:

```python
cfg = Config()
my_model = GPT2Transformer(cfg)
```

```python
my_model.load_state_dict(reference_gpt2.state_dict(), strict=False)
my_model.to(device)
```




    GPT2Transformer(
      (embed): Embedding()
      (pos_embed): PosEmbedding()
      (blocks): ModuleList(
        (0-11): 12 x TransformerBlock(
          (ln1): LayerNorm()
          (attn): CausalAttention()
          (ln2): LayerNorm()
          (mlp): MLP(
            (gelu): GELU(approximate='none')
          )
        )
      )
      (ln_final): LayerNorm()
      (unembed): Unembedding()
    )



```python
text = "The 2021 Masters (officially the 2021 Betfred Masters) was a professional non-ranking snooker tournament that took place from 10 to 17 January 2021 at the Marshall Arena in Milton Keynes, England. It was the 47th staging of the Masters, which was first held in 1975, and the second of three Triple Crown events in the 2020–21 season. The top sixteen players from the snooker world rankings were invited to compete in a knockout tournament, organised by the World Professional Billiards and Snooker Association. It was played behind closed doors because of COVID-19 restrictions in the United Kingdom. The defending champion, Stuart Bingham, had defeated Ali Carter 10–8 in the 2020 Masters final. Bingham lost 6–5 to Yan Bingtao (pictured) in the semi-finals. Yan (one of three debutants at the event, alongside Thepchaiya Un-Nooh and Gary Wilson) met John Higgins in the final. Yan completed a 10–8 victory to win his "
```

```python
for i in tqdm(range(100)):
  test_tokens = reference_gpt2.to_tokens(text).to(device)
  logits = my_model(test_tokens)
  text += reference_gpt2.tokenizer.decode(logits[-1, -1].argmax())
print(text)
```


      0%|          | 0/100 [00:00<?, ?it/s]


    The 2021 Masters (officially the 2021 Betfred Masters) was a professional non-ranking snooker tournament that took place from 10 to 17 January 2021 at the Marshall Arena in Milton Keynes, England. It was the 47th staging of the Masters, which was first held in 1975, and the second of three Triple Crown events in the 2020–21 season. The top sixteen players from the snooker world rankings were invited to compete in a knockout tournament, organised by the World Professional Billiards and Snooker Association. It was played behind closed doors because of COVID-19 restrictions in the United Kingdom. The defending champion, Stuart Bingham, had defeated Ali Carter 10–8 in the 2020 Masters final. Bingham lost 6–5 to Yan Bingtao (pictured) in the semi-finals. Yan (one of three debutants at the event, alongside Thepchaiya Un-Nooh and Gary Wilson) met John Higgins in the final. Yan completed a 10–8 victory to win his vernacular title.
    
    
    The tournament was held in the Marshall Arena, Milton Keynes, England. The tournament was played in a closed-door, non-competitive manner. The tournament was held in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played


```python
for i in tqdm(range(100)):
  test_tokens = reference_gpt2.to_tokens(text).to(device)
  logits = reference_gpt2(test_tokens)
  text += reference_gpt2.tokenizer.decode(logits[-1, -1].argmax())
print(text)
```


      0%|          | 0/100 [00:00<?, ?it/s]


    The 2021 Masters (officially the 2021 Betfred Masters) was a professional non-ranking snooker tournament that took place from 10 to 17 January 2021 at the Marshall Arena in Milton Keynes, England. It was the 47th staging of the Masters, which was first held in 1975, and the second of three Triple Crown events in the 2020–21 season. The top sixteen players from the snooker world rankings were invited to compete in a knockout tournament, organised by the World Professional Billiards and Snooker Association. It was played behind closed doors because of COVID-19 restrictions in the United Kingdom. The defending champion, Stuart Bingham, had defeated Ali Carter 10–8 in the 2020 Masters final. Bingham lost 6–5 to Yan Bingtao (pictured) in the semi-finals. Yan (one of three debutants at the event, alongside Thepchaiya Un-Nooh and Gary Wilson) met John Higgins in the final. Yan completed a 10–8 victory to win his vernacular title.
    
    
    The tournament was held in the Marshall Arena, Milton Keynes, England. The tournament was played in a closed-door, non-competitive manner. The tournament was held in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner. The tournament was played in a closed-door, non-competitive manner


## What’s next: training from scratch

In the next post, we’ll keep this exact model code (same module structure + parameter names) and add the missing pieces to **train GPT‑2 Small from scratch**:

- dataset + tokenizer pipeline (and how to batch sequences efficiently)
- causal language modeling loss (next-token prediction)
- optimizer + learning rate schedule
- training loop with logging, checkpointing, and evaluation
- quick overfit tests and a small-scale run you can reproduce on a single GPU

The nice part is: because we’ve already validated the forward pass against a known-good reference, any training issues will be about *optimization/data*, not mysterious architecture bugs.

