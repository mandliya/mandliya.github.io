---
layout: post
title: Gpt 2 Complete Loop
date: '2026-02-20 05:19:43 '
categories:
- Technology
tags:
- Jupyter
- Notebook
description: <a href="https://colab.research.google.com/github/mandliya/mandliya.github.io/blob/main/parked/gpt2completeloop.ipynb"
  target="parent"><img src="https...
image: /assets/img/gpt-2-complete-loop/cover.png
image_alt: Gpt 2 Complete Loop
math: true
mermaid: true
pin: false
toc: true
comments: true
---

<a href="https://colab.research.google.com/github/mandliya/mandliya.github.io/blob/main/parked/gpt_2_complete_loop.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```python
!pip install -q datasets jaxtyping tiktoken
```

```python
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from jaxtyping import Float, Int
import math
from typing import Optional, Tuple
import tiktoken
from datasets import load_dataset
from google.colab import userdata
import pathlib
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
```

```python
@dataclass
class GPT2Config:
  n_layers: int = 12
  d_model: int = 768
  n_heads: int = 12
  vocab_size: int = 50257
  layer_norm_eps: float = 0.02
  init_range: float = 0.02
  dropout: float = 0.1
  n_ctx: int = 1024
  d_mlp: int = 4 * 768
  weight_tying: bool = True
```

```python
class GPT2Attention(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super().__init__()
    assert cfg.d_model % cfg.n_heads == 0, (
        f"{cfg.d_model} should be divisible by {cfg.n_heads}"
    )
    self.cfg = cfg
    self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
    self.attn_dropout = nn.Dropout(cfg.dropout)
    self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)
    self.resid_dropout = nn.Dropout(cfg.dropout)
    self.register_buffer(
        'mask',
        torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx))
        .view(1, 1, cfg.n_ctx, cfg.n_ctx)
    )

  def forward(
      self,
      x: Float[Tensor, "B T d_model"]) -> Float[Tensor, "B T d_model"]:
      B, T, d_model = x.shape
      n_heads = self.cfg.n_heads
      d_head = d_model // n_heads
      qkv = self.c_attn(x) #[B, T, d_model * 3]
      q, k, v = qkv.split(d_model, dim=2)
      q = q.view(B, T, n_heads, d_head).transpose(1, 2) #[B nh T dh]
      k = k.view(B, T, n_heads, d_head).transpose(1, 2)
      v = v.view(B, T, n_heads, d_head).transpose(1, 2)

      attn = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(d_head))
      attn = attn.masked_fill(
          self.mask[:, :, :T, :T] == 0,
          float('-inf')
      )
      attn = attn.softmax(dim=-1)
      attn = self.attn_dropout(attn)
      out = attn @ v #[B, nh, T, dh]
      out = out.transpose(1, 2).contiguous().view(B, T, d_model)
      return self.resid_dropout(out)


```

```python
class GPT2MLP(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super().__init__()
    self.cfg = cfg
    self.c_fc = nn.Linear(cfg.d_model, cfg.d_mlp)
    self.gelu = nn.GELU()
    self.c_proj = nn.Linear(cfg.d_mlp, cfg.d_model)
    self.dropout = nn.Dropout(cfg.dropout)

  def forward(self, x: Float[Tensor, "B T d_model"]) -> Float[Tensor, "B T d_model"]:
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return self.dropout(x)
```

```python
class GPT2Block(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super().__init__()
    self.cfg = cfg
    self.ln_1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
    self.attn = GPT2Attention(cfg)
    self.ln_2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
    self.mlp = GPT2MLP(cfg)

  def forward(self, x: Float[Tensor, "B T d_model"]) -> Float[Tensor, "B T d_model"]:
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x
```

```python
class GPT2Model(nn.Module):
  def __init__(self, cfg: GPT2Config):
    super().__init__()
    self.cfg = cfg
    self.transformer = nn.ModuleDict(dict(
        wte = nn.Embedding(cfg.vocab_size, cfg.d_model),
        wpe = nn.Embedding(cfg.n_ctx, cfg.d_model),
        embd_dropout = nn.Dropout(cfg.dropout),
        h = nn.ModuleList([
            GPT2Block(cfg) for _ in range(self.cfg.n_layers)
        ]),
        ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
    ))
    self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)
    if cfg.weight_tying:
      self.lm_head.weight = self.transformer.wte.weight

    self.apply(self._init_weights)
    for np, p in self.named_parameters():
      if np.endswith('c_proj.weight'):
        nn.init.normal_(p, mean=0.0, std=(cfg.init_range/math.sqrt(2 * cfg.n_layers)))

  def _init_weights(self, module: nn.Module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_range)
      if module.bias is None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_range)

  def forward(
      self,
      tokens: Int[Tensor, "B T"],
      targets: Optional[Int[Tensor, "B T"]]
    ) -> Tuple[Int[Tensor, "B T vocab_size"], Float[Tensor, ""]]:
    B, T = tokens.shape
    assert T <= self.cfg.n_ctx, (
        f"Sequence length {T} is longer than max sequence length: {self.cfg.n_ctx}"
    )
    tok_emb = self.transformer.wte(tokens)
    pos = torch.arange(0, T, dtype=torch.long, device=tokens.device)
    pos_emb = self.transformer.wpe(pos)
    residual = pos_emb + tok_emb
    residual = self.transformer.embd_dropout(residual)
    for block in self.transformer.h:
      residual = block(residual)

    residual = self.transformer.ln_f(residual)
    if targets is not None:
      logits = self.lm_head(residual) #[B T vocab_size]
      loss = F.cross_entropy(
          logits.view(-1, logits.size(-1)), #[B*T vocab_size]
          targets.view(-1), #[B * T]
          ignore_index=-1
      )
    else:
      logits = self.lm_head(residual[:, [-1], :]) #[B 1 vocab_size]
      loss = None

    return logits, loss

  @torch.no_grad()
  def generate(
      self,
      tokens: Int[Tensor, "B T"],
      temperature: float = 1.0,
      max_num_tokens: int = 256,
      top_k: Optional[int] = None
    ) -> Int[Tensor, "B T+max_num_tokens"]:
    for _ in range(max_num_tokens):
      tok_cond = (
          tokens
          if tokens.size(-1) <= self.cfg.n_ctx
          else tokens[:, -self.cfg.n_ctx:]
      )
      logits, _ = self(tok_cond, targets=None) #[B T vocab_size]
      logits = logits[:, -1, :] #[B vocab_size]
      logits = logits / temperature
      if top_k is not None:
        k = min(k, self.cfg.vocab_size)
        v, _ = torch.topk(logits, k) #[B k]
        threshold = v[:, [-1]] #[B, 1]
        logits.masked_fill_(logits < threshold, float('-inf'))
      probs = logits.softmax(dim=-1) #[B vocab_size]
      next_token = torch.multinomial(probs, num_samples=1) #[B, 1]
      tokens = torch.cat((tokens, next_token), dim=1)
    return tokens



```

```python
text = "Hello, I am a large language model." * 500
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)
print(f'Total tokens: {len(tokens)}')
```

    Total tokens: 4500


```python
def get_batch(tokens: Tensor, block_size: int, batch_size: int, device: str):
  idx = torch.randint(0, len(tokens)-block_size, (batch_size,))
  x = torch.stack([tokens[i: i+block_size] for i in idx])
  y = torch.stack([tokens[i+1: i+block_size+1] for i in idx])
  return x.to(device), y.to(device)
```

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = GPT2Config(dropout=0)
model = GPT2Model(cfg)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-5)
steps = 200
for step in range(steps):
  x, y = get_batch(tokens, block_size=cfg.n_ctx, batch_size=2, device=device)
  optimizer.zero_grad()
  _, loss = model(x, targets=y)
  loss.backward()
  optimizer.step()
  if step % 10 == 0:
    print(f'{step=:4} | Loss: {loss.item():.6f}')
```

    step=   0 | Loss: 11.091410
    step=  10 | Loss: 5.922181
    step=  20 | Loss: 5.396425
    step=  30 | Loss: 5.022036
    step=  40 | Loss: 4.661715
    step=  50 | Loss: 4.316027
    step=  60 | Loss: 3.979648
    step=  70 | Loss: 3.658325
    step=  80 | Loss: 3.350863
    step=  90 | Loss: 3.031948
    step= 100 | Loss: 2.147627
    step= 110 | Loss: 1.343678
    step= 120 | Loss: 0.310863
    step= 130 | Loss: 0.093907
    step= 140 | Loss: 0.061922
    step= 150 | Loss: 0.048684
    step= 160 | Loss: 0.039142
    step= 170 | Loss: 0.033269
    step= 180 | Loss: 0.029778
    step= 190 | Loss: 0.025852


```python
HF_TOKEN = userdata.get('HF_TOKEN')

```

```python
@dataclass
class DatasetConfig:
  out_dir: str = './data'
  write_batch_size: int = 128
  hf_dataset:str = 'roneneldan/TinyStories'
  max_examples: int = 600000
  n_ctx: int = 1024
```

```python
def pretokenize_and_save(config: DatasetConfig, split='train'):
  dataset = load_dataset(
      config.hf_dataset,
      split=split,
      streaming=True,
  )
  enc = tiktoken.get_encoding('gpt2')
  os.makedirs(config.out_dir, exist_ok=True)
  data_path = pathlib.Path(config.out_dir) / f'{split}.bin'
  total_tokens = 0
  with open(data_path, mode='wb') as f:
    batch_tokens = []
    for i, example in enumerate(tqdm(dataset)):
      if i >= config.max_examples:
        break
      tokens = enc.encode_ordinary(example['text'])
      batch_tokens.extend(tokens)
      total_tokens += len(tokens)
      if (i + 1) % config.write_batch_size == 0:
        chunk = np.array(batch_tokens, dtype=np.uint16)
        f.write(chunk.tobytes())
        batch_tokens = []

    if batch_tokens:
      chunk = np.array(chunk, dtype=np.uint16)
      f.write(chunk.tobytes())

  print(f'Wrote {total_tokens=:,} to path: {str(data_path)}')

```

```python
config = DatasetConfig()
pretokenize_and_save(config)
```


    README.md: 0.00B [00:00, ?B/s]


    600000it [02:20, 4269.56it/s]

    Wrote total_tokens=134,054,957 to path: data/train.bin


    


```python
pretokenize_and_save(config, split='validation')
```

    21990it [00:06, 3316.04it/s]

    Wrote total_tokens=4,743,928 to path: data/validation.bin


    


```python
class TokenDataset(Dataset):
  def __init__(self, config: DatasetConfig, split='train'):
    super().__init__()
    self.block_size = config.n_ctx
    path = pathlib.Path(config.out_dir) / f'{split}.bin'
    self.tokens = np.memmap(path, dtype=np.uint16, mode='r')

  def __len__(self):
    return len(self.tokens) // self.block_size

  def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
    start = index * self.block_size
    chunk = torch.from_numpy(
        self.tokens[start: start+1+self.block_size].astype(np.int64)
    )
    x = chunk[:-1]
    y = chunk[1:]
    return x, y
```

```python
@dataclass
class TrainingConfig:
  lr: float = 3e-5
  log_steps: int = 1000
  max_iters: int = 100
  train_batch_size: int = 8
  val_batch_size: int = 8
  out_path: str = './out'

```

```python
def evaluate(
    model: GPT2Model,
    loader: DataLoader,
    max_batches: int = 20) -> float:
    model.eval()
    device = model.device
    losses = []
    enc = tiktoken.get_encoding('gpt2')
    with torch.no_grad():
      for i, (x, y) in enumerate(tqdm(loader)):
        if i >= max_batches:
          break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        losses.append(loss.item())
        print(f'Input: {enc.decode(x[0])}\nOutput: {enc.decode(logits[len(x[0]):])}')
    return sum(losses) / len(losses)
```

```python

torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPT2Config()
model = GPT2Model(config).to(device)
data_config = DatasetConfig()
train_dataset = TokenDataset(data_config, split='train')
val_dataset = TokenDataset(data_config, split='validation')




def train(
    model: GPT2Model,
    train_dataset: TokenDataset,
    valid_dataset: TokenDataset,
    train_config: TrainingConfig):

  train_dataloader = DataLoader(
      train_dataset,
      batch_size=train_config.train_batch_size,
      shuffle=True,
      num_workers=4,
      pin_memory=True
  )

  val_dataloader = DataLoader(
      val_dataset,
      batch_size=train_config.val_batch_size,
      shuffle=False,
      pin_memory=False
  )

  best_val_loss = float('inf')
  best_model_path = pathlib.Path(train_config.out_path) / f'best_model.pt'

  for epoch in range(train_config.max_iters):
    for step, (x, y) in enumerate(train_dataloader):
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      _, loss = model(x, targets=y)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      if step % train_config.log_steps == 0:
        print(f'epoch {epoch:4} | step: {step:5d} | train_loss {loss.item():.4f}')


    val_loss = evaluate(model, val_dataloader, max_batches=2000)
    print(f'epoch: {epoch} | val loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save({
          'epoch' : epoch,
          'model' : model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'val_loss' : val_loss,
          'model_config': config,
          'data_config': data_config,
          'train_config': train_config
      }, best_model_path)



```

```python
train_config = TrainingConfig()
train(model, train_dataset=train_dataset, valid_dataset=val_dataset, train_config=train_config)
```

    epoch    0 | step:     0 | train_loss 10.9824
    epoch    0 | step:  1000 | train_loss 10.9613
    epoch    0 | step:  2000 | train_loss 10.9674
    epoch    0 | step:  3000 | train_loss 10.9631
    epoch    0 | step:  4000 | train_loss 10.9626
    epoch    0 | step:  5000 | train_loss 10.9816
    epoch    0 | step:  6000 | train_loss 10.9872
    epoch    0 | step:  7000 | train_loss 11.0034
    epoch    0 | step:  8000 | train_loss 10.9915
    epoch    0 | step:  9000 | train_loss 10.9861
    epoch    0 | step: 10000 | train_loss 10.9900


```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
@dataclass
class DatasetConfig:
  out_dir: str = './out'
  hf_dataset: str = 'roneneldan/TinyStories'
  write_batch_size: int = 100
  max_examples: int = 10000
  n_ctx: int = 1024

```

```python
def pretokenize_and_save_dataset(
    d_config: DatasetConfig,
    split: str = 'train'
  ) -> None:
  dataset = load_dataset(
      d_config.hf_dataset,
      split=split,
      streaming=True
  )
  os.makedirs(d_config.out_dir, exist_ok=True)
  data_path = pathlib.Path(d_config.out_dir) / f'{split}.bin'
  with open(data_path, mode='wb') as f:
    batch_tokens = []
    total_tokens = 0

    for i, example in enumerate(tqdm(dataset)):
      if i >= d_config.max_examples:
        break
      tokens = enc.encode_ordinary(example['text'])
      tokens.append(enc.eot_token)
      batch_tokens.extend(tokens)
      total_tokens += len(tokens)
      if (i+1) % d_config.write_batch_size:
        chunk = np.array(batch_tokens, dtype=np.uint16)
        f.write(chunk.tobytes())
        batch_tokens = []

    if batch_tokens:
      chunk = np.array(batch_tokens, dtype=np.uint16)
      f.write(chunk.tobytes())
  print(f'Wrote {total_tokens=} to path: {str(data_path)}')
```

```python
d_config = DatasetConfig()
pretokenize_and_save_dataset(d_config)
```

    10000it [00:22, 446.71it/s]

    Wrote total_tokens=2162078 to path: out/train.bin


    


```python
pretokenize_and_save_dataset(d_config, split='validation')
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipython-input-1748778094.py in <cell line: 0>()
    ----> 1 pretokenize_and_save_dataset(d_config, split='validation')
    

    NameError: name 'pretokenize_and_save_dataset' is not defined


```python
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
  def __init__(self, config: DatasetConfig, split:str = 'train'):
    super().__init__()
    self.block_size = config.n_ctx
    path = pathlib.Path(config.out_dir) / f'{split}.bin'
    self.tokens = np.memmap(path, dtype=np.uint16, mode='r')
    print(f'Total tokens: {len(self.tokens):,}')
    print(f'Total chunks: {len(self):,}')

  def __len__(self):
    return len(self.tokens) // self.block_size

  def __getitem__(self, index):
    start = index * self.block_size
    chunk = torch.from_numpy(
        self.tokens[start: start + self.block_size + 1].astype(np.int64)
    )
    x = chunk[:-1]
    y = chunk[1:]
    return x,y
```

```python
dataset = TokenDataset(d_config)
```

    Total tokens: 2,162,078
    Total chunks: 2,111


```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = GPT2Config(dropout=0)
model = GPT2Model(cfg)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-5)
steps = 20000
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, )
for step in range(steps):
  x, y = next
  optimizer.zero_grad()
  _, loss = model(x, targets=y)
  loss.backward()
  optimizer.step()
  if step % 10 == 0:
    print(f'{step=:4} | Loss: {loss.item():.6f}')
```
