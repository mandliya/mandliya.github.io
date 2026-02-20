---
layout: post
title: Gpt 2 Complete Loop
date: '2026-02-20 19:50:40 '
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
  layer_norm_eps: float = 1e-5
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
      out = self.c_proj(out)
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
  log_steps: int = 200
  max_iters: int = 100
  train_batch_size: int = 8
  val_batch_size: int = 4
  out_path: str = './out'

```

```python
def evaluate(
    model: GPT2Model,
    loader: DataLoader,
    max_batches: int = 20,
    device:str='cuda') -> float:
    model.eval()
    losses = []
    enc = tiktoken.get_encoding('gpt2')
    with torch.no_grad():
      for i, (x, y) in enumerate(tqdm(loader)):
        if i >= max_batches:
          break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        losses.append(loss.item())
        if i == 0:
          preds = logits.argmax(dim=-1)
          print(f'Input: {enc.decode(x[0].tolist())}\nOutput: {enc.decode(preds[0].tolist())}')
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

  optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)

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
  os.makedirs(train_config.out_path, exist_ok=True)

  for epoch in range(train_config.max_iters):
    model.train()
    for step, (x, y) in enumerate(train_dataloader):
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad()
      _, loss = model(x, targets=y)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      if step % train_config.log_steps == 0:
        print(f'epoch {epoch:4} | step: {step:5d} | train_loss {loss.item():.4f}')


    val_loss = evaluate(model, val_dataloader, max_batches=2000, device=device)
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

    epoch    0 | step:     0 | train_loss 10.8672
    epoch    0 | step:   200 | train_loss 4.7543
    epoch    0 | step:   400 | train_loss 4.2919
    epoch    0 | step:   600 | train_loss 4.1205
    epoch    0 | step:   800 | train_loss 3.8463
    epoch    0 | step:  1000 | train_loss 3.8636
    epoch    0 | step:  1200 | train_loss 3.4059
    epoch    0 | step:  1400 | train_loss 3.4324
    epoch    0 | step:  1600 | train_loss 3.4059
    epoch    0 | step:  1800 | train_loss 3.4817
    epoch    0 | step:  2000 | train_loss 3.1964
    epoch    0 | step:  2200 | train_loss 3.0764
    epoch    0 | step:  2400 | train_loss 3.0702
    epoch    0 | step:  2600 | train_loss 2.9180
    epoch    0 | step:  2800 | train_loss 3.0865
    epoch    0 | step:  3000 | train_loss 2.6199
    epoch    0 | step:  3200 | train_loss 2.7768
    epoch    0 | step:  3400 | train_loss 2.7093
    epoch    0 | step:  3600 | train_loss 2.7877
    epoch    0 | step:  3800 | train_loss 2.8743
    epoch    0 | step:  4000 | train_loss 2.7639
    epoch    0 | step:  4200 | train_loss 2.7605
    epoch    0 | step:  4400 | train_loss 2.5423
    epoch    0 | step:  4600 | train_loss 2.6480
    epoch    0 | step:  4800 | train_loss 2.6782
    epoch    0 | step:  5000 | train_loss 2.4164
    epoch    0 | step:  5200 | train_loss 2.5817
    epoch    0 | step:  5400 | train_loss 2.3830
    epoch    0 | step:  5600 | train_loss 2.4596
    epoch    0 | step:  5800 | train_loss 2.4962
    epoch    0 | step:  6000 | train_loss 2.3183
    epoch    0 | step:  6200 | train_loss 2.6343
    epoch    0 | step:  6400 | train_loss 2.3217
    epoch    0 | step:  6600 | train_loss 2.3415
    epoch    0 | step:  6800 | train_loss 2.3861
    epoch    0 | step:  7000 | train_loss 2.3711
    epoch    0 | step:  7200 | train_loss 2.2240
    epoch    0 | step:  7400 | train_loss 2.3557
    epoch    0 | step:  7600 | train_loss 2.2241
    epoch    0 | step:  7800 | train_loss 1.9983
    epoch    0 | step:  8000 | train_loss 2.1771
    epoch    0 | step:  8200 | train_loss 2.1780
    epoch    0 | step:  8400 | train_loss 2.0895
    epoch    0 | step:  8600 | train_loss 2.3186
    epoch    0 | step:  8800 | train_loss 2.2652
    epoch    0 | step:  9000 | train_loss 2.0469
    epoch    0 | step:  9200 | train_loss 2.2080
    epoch    0 | step:  9400 | train_loss 2.2749
    epoch    0 | step:  9600 | train_loss 2.0624
    epoch    0 | step:  9800 | train_loss 2.2771
    epoch    0 | step: 10000 | train_loss 2.1773
    epoch    0 | step: 10200 | train_loss 2.0818
    epoch    0 | step: 10400 | train_loss 1.9239
    epoch    0 | step: 10600 | train_loss 2.1989
    epoch    0 | step: 10800 | train_loss 1.9658
    epoch    0 | step: 11000 | train_loss 2.0957
    epoch    0 | step: 11200 | train_loss 2.0179
    epoch    0 | step: 11400 | train_loss 1.8112
    epoch    0 | step: 11600 | train_loss 1.8858
    epoch    0 | step: 11800 | train_loss 2.0028
    epoch    0 | step: 12000 | train_loss 2.1308
    epoch    0 | step: 12200 | train_loss 1.9914
    epoch    0 | step: 12400 | train_loss 2.0677
    epoch    0 | step: 12600 | train_loss 1.8563
    epoch    0 | step: 12800 | train_loss 1.7798
    epoch    0 | step: 13000 | train_loss 1.8533
    epoch    0 | step: 13200 | train_loss 2.0353
    epoch    0 | step: 13400 | train_loss 2.0131
    epoch    0 | step: 13600 | train_loss 1.6725
    epoch    0 | step: 13800 | train_loss 1.9525
    epoch    0 | step: 14000 | train_loss 1.9424
    epoch    0 | step: 14200 | train_loss 1.7583
    epoch    0 | step: 14400 | train_loss 1.8926
    epoch    0 | step: 14600 | train_loss 1.8120
    epoch    0 | step: 14800 | train_loss 2.0186
    epoch    0 | step: 15000 | train_loss 1.9006
    epoch    0 | step: 15200 | train_loss 1.9359
    epoch    0 | step: 15400 | train_loss 1.6987
    epoch    0 | step: 15600 | train_loss 2.0470
    epoch    0 | step: 15800 | train_loss 1.9062
    epoch    0 | step: 16000 | train_loss 1.8837
    epoch    0 | step: 16200 | train_loss 1.8130


      0%|          | 3/1160 [00:00<01:54, 10.12it/s]

    Input: Spot. Spot saw the shiny car and said, "Wow, Kitty, your car is so bright and clean!" Kitty smiled and replied, "Thank you, Spot. I polish it every day."
    
    After playing with the car, Kitty and Spot felt thirsty. They found a small pond with clear water. They drank the water and felt very happy. They played together all day and became best friends.Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and hills. One day, Roxy found an icy hill. She had never seen anything like it before. It was shiny and cold, and she wanted to climb it.
    
    Roxy tried to climb the icy hill, but it was very slippery. She tried again and again, but she kept falling down. Roxy was sad. She wanted to climb the icy hill so much. Then, she saw a little bird named Billy. Billy saw that Roxy was sad and asked, "Why are you sad, Roxy?"
    
    Roxy told Billy about the icy hill and how she couldn't climb it. Billy said, "I have an idea! Let's find some big leaves to put under your feet. They will help you climb the icy hill." Roxy and Billy looked for big leaves and found some. Roxy put the leaves under her feet and tried to climb the icy hill again.
    
    This time, Roxy didn't slip. She climbed and climbed until she reached the top of the icy hill. Roxy was so happy! She and Billy played on the icy hill all day. From that day on, Roxy and Billy were the best of friends, and they climbed and played together all the time. And Roxy learned that with a little help from a friend, she could climb anything.Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was Daisy. Daisy was very small, but she was also very happy.
    
    One day, Daisy saw a dog. The dog was big and had a name too. His name was Max. Max liked to play in the yard. Daisy liked to watch Max play. Max and Daisy became friends.
    
    Every day, Max would come to the yard to play. Daisy would watch and smile. They were very happy together. And even though Daisy was small, she knew that she had a big friend in Max.Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around the house. One day, her mom asked her to wipe the table after they ate their lunch. Sue was happy to help.
    
    As Sue was wiping the table, she saw a pretty candle on the window sill. The candle was her mom's favorite. Sue wanted to do something nice for her mom, so she said, "Mom, can I light the candle for you?" Her mom said, "Yes, but be very careful."
    
    Sue carefully lit the candle and put it on the table. Her mom was so happy to see the pretty candle. They both sat and watched the candle burn. Sue's mom said, "Thank you, Sue, for being so thoughtful and careful." Sue felt proud that she could help her mom.
    
    The moral of the story is to always be thoughtful and careful when helping others.Once upon a time, there was a kind farmer. He had a big cow. The cow was sad. The farmer did not know why.
    
    One day, a little boy came to the farm. He saw the sad cow. The boy kneeled down to talk to the cow. "Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a friend."
    
    The kind farmer heard the cow. He wanted to help. So, he got another cow to be friends with the sad cow. The sad cow was happy now. They played together every day. And the kind farmer, the little boy, and the two cows all lived happily ever after.Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom. They loved to play together in the big green park near their house. One sunny day, they went to the park to play.
    
    While playing, Tom saw a big sour lemon on the ground. He wanted to play with it, but when he touched it, it started to roll away. Tom ran after the lemon, trying to catch it. But as he ran, Tom got lost in the park. Lucy looked around, but she could not find Tom. She was very sad.
    
    Lucy did not give up. She searched the park for her friend. At last, she found him near a big tree. Tom was trying to catch the lemon, but it vanished into a hole in the ground. Tom was happy to see Lucy again
    Output:  was He was a cat thing and wanted, "Wow, that! that car is so pretty and pretty.
     was and said, "Yes you, Spot! I love my so day."
    
    Spot a, the car, Kitty and Spot went happy. They went a big bottle to ducks water. Kitty drank some water and drank happy happy. They played with in day long had good friends.Once upon a time, in a small forest, there was a littleinoceros named Bobuf. Roxy was to play trees He would trees and and, and even. One day, Roxy saw a old tree. She wanted never seen it like it before.
     was very and pretty. and she wanted to go it.
    
    Roxy climbed to climb the hill hill, but she was too hard. She tried to and again, but she couldn going.. Sheoxy was sad, She wanted to climb the hill hill, badly. She, she saw a big bird. Lily. Billy was the heoxy was sad. asked, "Why are you sad?" littleoxy?"
    
    Roxy said Billy about the icy hill. how he could't climb it. Billy said, "Don can an idea! Let's climb a water rocks to climb on the feet and Maybe will be you." the hill hill." Sooxy was Billy worked around the leaves to found a leaves Theyoxy was the leaves on his feet. they to climb the hill hill..
    
    R time, Roxy and't give and She was the climbed until she reached the top. the hill hill. Sheoxy was so happy to She thanked Billy were in the icy hill all day long They that day on, Roxy and Billy were the best of friends. and they always the played together every day time.Once theyoxy was that sometimes a little help, her cold, he could do the sheOnce upon a time, there a small house, there was a little dogisy. The daisy was many big on The name was Lily. Daisy liked a pretty. but she liked very very small.
    
    One day, Daisy saw a big. The dog was big and brown a long.. Daisy name was Max. Daisy was to play with the yard. He wanted to play the play with
     liked Daisy played best.
    
    One day, Daisy and go to the yard and play with He would play Max play. She would very happy.. They they though Max was small, she could Max Max could a friend friend. the. Max upon a time, there was a little dog named Lily. Sue loved to play her mom in the house. One day, her mom asked her to help the dishes. dinner were. food. Sue did very to help her
    
    S they was cleaning the table, her saw a big flower on the table.. She candle was very mom's birthday toy Sue wanted to help something with for her mom. so she went, "Mom, can I help the candle?" you?" Her mom smiled, "Yes, Sue be careful careful.
    
    Sue went opened the candle and put it on her window. She mom was very proud to see the candle candle. She both smiled down watched the candle together. Sue was mom said, "Good you, Sue. for helping so kind and helping with Sue smiled proud of she could help her mom andOnce
    From moral of the story is that always be kind and help when you others,Once upon a time, there was a little girl. He had a big farm named The cow was very because The farmer wanted not like what the He
    One day, the little girl came to the farm. The saw the cow cow. The farmer wantedeled down and the to the cow. TheHi are you sad?" cow?" the asked. The cow said, "I am sad. I am to friend."
    
    The farmer farmer smiled the boy's He said to help the He, he said a cow. play his.. cow cow. The cow cow was not.. The all together all day.Once they farmer farmer was the cow boy, the the cow became became lived happily ever after.Once upon a time, there was a little girl named Lily. She loved a big cat named Mitt. Tom were to play together. the park, grass. their house.
     day day, they saw to the park to play.
    
    At they, they saw a big tree bug on the ground. He wanted to eat with it, but he he tried it, it made to shrink away. Lucy was after the lemon, but to get it.
     he he was, he tri very. the mud.
     was for and but she could not find Tom.
     was very sad.
    
    Thency saw not know up. She went high park, Tom mom, She the, she found Tom. the big tree. She was so to get the lemon, but he was. the big. the ground. Lucy was very and see her again.


    100%|██████████| 1160/1160 [01:47<00:00, 10.79it/s]


    epoch: 0 | val loss: 1.8179
    epoch    1 | step:     0 | train_loss 1.6890
    epoch    1 | step:   200 | train_loss 1.8509
    epoch    1 | step:   400 | train_loss 2.0580
    epoch    1 | step:   600 | train_loss 1.8764
    epoch    1 | step:   800 | train_loss 1.7376
    epoch    1 | step:  1000 | train_loss 1.9256
    epoch    1 | step:  1200 | train_loss 1.9663
    epoch    1 | step:  1400 | train_loss 1.8613
    epoch    1 | step:  1600 | train_loss 1.7877
    epoch    1 | step:  1800 | train_loss 1.8273
    epoch    1 | step:  2000 | train_loss 1.7959
    epoch    1 | step:  2200 | train_loss 1.7887
    epoch    1 | step:  2400 | train_loss 1.9185
    epoch    1 | step:  2600 | train_loss 1.7561
    epoch    1 | step:  2800 | train_loss 1.7235
    epoch    1 | step:  3000 | train_loss 1.8207
    epoch    1 | step:  3200 | train_loss 1.6815
    epoch    1 | step:  3400 | train_loss 1.8177
    epoch    1 | step:  3600 | train_loss 1.8451
    epoch    1 | step:  3800 | train_loss 1.8381
    epoch    1 | step:  4000 | train_loss 1.8918
    epoch    1 | step:  4200 | train_loss 1.7610
    epoch    1 | step:  4400 | train_loss 1.8250
    epoch    1 | step:  4600 | train_loss 1.7966
    epoch    1 | step:  4800 | train_loss 1.9053
    epoch    1 | step:  5000 | train_loss 1.7515
    epoch    1 | step:  5200 | train_loss 1.8978
    epoch    1 | step:  5400 | train_loss 1.8902
    epoch    1 | step:  5600 | train_loss 1.6428
    epoch    1 | step:  5800 | train_loss 1.6934
    epoch    1 | step:  6000 | train_loss 1.7835
    epoch    1 | step:  6200 | train_loss 1.7451
    epoch    1 | step:  6400 | train_loss 1.8574
    epoch    1 | step:  6600 | train_loss 1.7564
    epoch    1 | step:  6800 | train_loss 1.9046
    epoch    1 | step:  7000 | train_loss 1.6890
    epoch    1 | step:  7200 | train_loss 1.7639
    epoch    1 | step:  7400 | train_loss 1.6103
    epoch    1 | step:  7600 | train_loss 1.7496
    epoch    1 | step:  7800 | train_loss 1.6882
    epoch    1 | step:  8000 | train_loss 1.6848
    epoch    1 | step:  8200 | train_loss 1.6687
    epoch    1 | step:  8400 | train_loss 1.7695
    epoch    1 | step:  8600 | train_loss 1.9208
    epoch    1 | step:  8800 | train_loss 1.8674
    epoch    1 | step:  9000 | train_loss 1.6581
    epoch    1 | step:  9200 | train_loss 1.7932
    epoch    1 | step:  9400 | train_loss 1.7711
    epoch    1 | step:  9600 | train_loss 1.7594
    epoch    1 | step:  9800 | train_loss 1.6592
    epoch    1 | step: 10000 | train_loss 1.6440
    epoch    1 | step: 10200 | train_loss 1.5631
    epoch    1 | step: 10400 | train_loss 1.8042
    epoch    1 | step: 10600 | train_loss 1.6981
    epoch    1 | step: 10800 | train_loss 1.5989
    epoch    1 | step: 11000 | train_loss 1.8172
    epoch    1 | step: 11200 | train_loss 1.7644
    epoch    1 | step: 11400 | train_loss 1.7860
    epoch    1 | step: 11600 | train_loss 1.5987
    epoch    1 | step: 11800 | train_loss 1.6080
    epoch    1 | step: 12000 | train_loss 1.5227
    epoch    1 | step: 12200 | train_loss 1.6634
    epoch    1 | step: 12400 | train_loss 1.7539
    epoch    1 | step: 12600 | train_loss 1.8504
    epoch    1 | step: 12800 | train_loss 1.5832
    epoch    1 | step: 13000 | train_loss 1.8126
    epoch    1 | step: 13200 | train_loss 1.5939
    epoch    1 | step: 13400 | train_loss 1.5528
    epoch    1 | step: 13600 | train_loss 1.7451
    epoch    1 | step: 13800 | train_loss 1.5520
    epoch    1 | step: 14000 | train_loss 1.5412
    epoch    1 | step: 14200 | train_loss 1.8012
    epoch    1 | step: 14400 | train_loss 1.5283
    epoch    1 | step: 14600 | train_loss 1.6017
    epoch    1 | step: 14800 | train_loss 1.6395
    epoch    1 | step: 15000 | train_loss 1.5024
    epoch    1 | step: 15200 | train_loss 1.7027
    epoch    1 | step: 15400 | train_loss 1.6614
    epoch    1 | step: 15600 | train_loss 1.7703
    epoch    1 | step: 15800 | train_loss 1.5871
    epoch    1 | step: 16000 | train_loss 1.5299
    epoch    1 | step: 16200 | train_loss 1.4290


      0%|          | 2/1160 [00:00<01:49, 10.59it/s]

    Input: Spot. Spot saw the shiny car and said, "Wow, Kitty, your car is so bright and clean!" Kitty smiled and replied, "Thank you, Spot. I polish it every day."
    
    After playing with the car, Kitty and Spot felt thirsty. They found a small pond with clear water. They drank the water and felt very happy. They played together all day and became best friends.Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and hills. One day, Roxy found an icy hill. She had never seen anything like it before. It was shiny and cold, and she wanted to climb it.
    
    Roxy tried to climb the icy hill, but it was very slippery. She tried again and again, but she kept falling down. Roxy was sad. She wanted to climb the icy hill so much. Then, she saw a little bird named Billy. Billy saw that Roxy was sad and asked, "Why are you sad, Roxy?"
    
    Roxy told Billy about the icy hill and how she couldn't climb it. Billy said, "I have an idea! Let's find some big leaves to put under your feet. They will help you climb the icy hill." Roxy and Billy looked for big leaves and found some. Roxy put the leaves under her feet and tried to climb the icy hill again.
    
    This time, Roxy didn't slip. She climbed and climbed until she reached the top of the icy hill. Roxy was so happy! She and Billy played on the icy hill all day. From that day on, Roxy and Billy were the best of friends, and they climbed and played together all the time. And Roxy learned that with a little help from a friend, she could climb anything.Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was Daisy. Daisy was very small, but she was also very happy.
    
    One day, Daisy saw a dog. The dog was big and had a name too. His name was Max. Max liked to play in the yard. Daisy liked to watch Max play. Max and Daisy became friends.
    
    Every day, Max would come to the yard to play. Daisy would watch and smile. They were very happy together. And even though Daisy was small, she knew that she had a big friend in Max.Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around the house. One day, her mom asked her to wipe the table after they ate their lunch. Sue was happy to help.
    
    As Sue was wiping the table, she saw a pretty candle on the window sill. The candle was her mom's favorite. Sue wanted to do something nice for her mom, so she said, "Mom, can I light the candle for you?" Her mom said, "Yes, but be very careful."
    
    Sue carefully lit the candle and put it on the table. Her mom was so happy to see the pretty candle. They both sat and watched the candle burn. Sue's mom said, "Thank you, Sue, for being so thoughtful and careful." Sue felt proud that she could help her mom.
    
    The moral of the story is to always be thoughtful and careful when helping others.Once upon a time, there was a kind farmer. He had a big cow. The cow was sad. The farmer did not know why.
    
    One day, a little boy came to the farm. He saw the sad cow. The boy kneeled down to talk to the cow. "Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a friend."
    
    The kind farmer heard the cow. He wanted to help. So, he got another cow to be friends with the sad cow. The sad cow was happy now. They played together every day. And the kind farmer, the little boy, and the two cows all lived happily ever after.Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom. They loved to play together in the big green park near their house. One sunny day, they went to the park to play.
    
    While playing, Tom saw a big sour lemon on the ground. He wanted to play with it, but when he touched it, it started to roll away. Tom ran after the lemon, trying to catch it. But as he ran, Tom got lost in the park. Lucy looked around, but she could not find Tom. She was very sad.
    
    Lucy did not give up. She searched the park for her friend. At last, she found him near a big tree. Tom was trying to catch the lemon, but it vanished into a hole in the ground. Tom was happy to see Lucy again
    Output:  was He was Spot cat thing and wanted, "Wow, that! that car is so pretty and pretty. Kitty smiled and said, "Thank you, Spot!" I like my so day."
    
    From they, Spot car, Spot and Spot went good. They went a big bottle to water water. Kitty drank some water and felt better happy. Kitty played with and day long became best friends.Once upon a time, there a small forest, there was a littleinoceros. Remyino. Roxy was to play trees He would trees and and, and even. R day, Roxy saw a old tree. She wanted never seen snow like it before.
     was very and pretty. and R wanted to climb it.
    
    Roxy climbed to climb the hill hill, but she was too hard. She slipped to and again, but she could falling.. Sheoxy was sad and She wanted to climb the icy hill, badly. She, she had a big bird named Tweet. Billy was R Roxy was sad and asked, "Why are you sad, Roxy?" R
    Roxy said Billy about the icy hill. how she wanted't climb it. Billy said, "Don can an idea! Let's climb a shade rocks and make on the back." We will be you climb the icy hill."
    oxy and Billy worked around the leaves and found some big Theyoxy put the leaves on her feet and they to climb the icy hill.. She
    R time, Roxy and't give and She climbed the climbed until she reached the top. the hill hill. Sheoxy was so happy and She thanked Billy climbed in the icy hill all day long They that day on, Roxy and Billy always the best of friends. and they always the climbed together every day time.Once theyoxy was that sometimes a little help, her friend, she could do the.Once upon a time, there a small house, there lived a little boyisy. The daisy was many pretty. The name was Daisy. Daisy loved a happy. but she was very very small. She
    One day, Daisy saw a big. The dog was very and had a long.. Daisy name was Max. Daisy was to play with the yard. He wanted to play Max play.
     liked Daisy liked good.
    
    One day, Daisy would visit to the yard to play with He would watch Max Max. She would very happy together. They they though Max was small, she was Max Max could a friend friend like Max.Once upon a time, there was a little little named Lily. Sue loved to help her mom in the house. One day, Sue mom asked her to help the table. dinner finished the food. Sue did very to help.
    
    S they was wiping the table, she saw a big flower on the table.. She candle was very favorite's birthday color Sue wanted to help the special for her mom. so she went, "Mom, can I help the candle?" me?" Her mom smiled, "Yes, you be careful careful."
    
    Sue took wiped the candle and put it on the table. She mom was so proud and see the pretty candle. Sue both smiled down watched the candle together. Sue was mom said, "Good you, Sue, for helping so thoughtful and helping." Sue smiled proud that she helped help her mom.Once
    From moral of the story is to always be kind and help when you others,Once upon a time, there was a little girl named He had a big farm named The cow was very because The farmer wanted not want how. He
    One day, the little girl came to the farm. He saw the cow cow. The boy wantedeled down and talk to the cow. TheWhy are you sad?" cow?" the asked. The cow said, "I am sad because I want to friend."
    
    The boy farmer wanted the boy's He said to help the He, he said a cow. talk his.. cow cow. The cow cow was happy.. The played together and day.Once they cow farmer was the cow boy, and the cow were lived lived happily ever after.Once upon a time, there was a little girl named Lily. She loved a big dog named Fl. Tom loved to play together. the park yard yard. their house. One day day, Lucy went to the park to play.
    
    While they, Lucy saw a big, lemon on the ground. He wanted to eat with it, but Lucy he tried it, it made to shrink away. Lucy was after the lemon, but to get it. But the he ran, he tri caught. the park.
     was everywhere and but she could not find Tom.
     was very sad.
    
    Lucy's not give up. She went and park, Tom cat, She the, she found Tom under a big tree. She was so to get the lemon, but he was. the big. the ground. Lucy was very to see her again.


    100%|██████████| 1160/1160 [01:47<00:00, 10.81it/s]


    epoch: 1 | val loss: 1.5978
    epoch    2 | step:     0 | train_loss 1.4615
    epoch    2 | step:   200 | train_loss 1.6890
    epoch    2 | step:   400 | train_loss 1.5846
    epoch    2 | step:   600 | train_loss 1.6913
    epoch    2 | step:   800 | train_loss 1.6966
    epoch    2 | step:  1000 | train_loss 1.5864
    epoch    2 | step:  1200 | train_loss 1.5298
    epoch    2 | step:  1400 | train_loss 1.5734
    epoch    2 | step:  1600 | train_loss 1.5904
    epoch    2 | step:  1800 | train_loss 1.7103
    epoch    2 | step:  2000 | train_loss 1.5467
    epoch    2 | step:  2200 | train_loss 1.4878
    epoch    2 | step:  2400 | train_loss 1.4842
    epoch    2 | step:  2600 | train_loss 1.5420
    epoch    2 | step:  2800 | train_loss 1.6234
    epoch    2 | step:  3000 | train_loss 1.6033
    epoch    2 | step:  3200 | train_loss 1.7546
    epoch    2 | step:  3400 | train_loss 1.4086
    epoch    2 | step:  3600 | train_loss 1.6263
    epoch    2 | step:  3800 | train_loss 1.6853
    epoch    2 | step:  4000 | train_loss 1.5591
    epoch    2 | step:  4200 | train_loss 1.6005
    epoch    2 | step:  4400 | train_loss 1.8926
    epoch    2 | step:  4600 | train_loss 1.4185
    epoch    2 | step:  4800 | train_loss 1.6349
    epoch    2 | step:  5000 | train_loss 1.3380
    epoch    2 | step:  5200 | train_loss 1.6876
    epoch    2 | step:  5400 | train_loss 1.7718
    epoch    2 | step:  5600 | train_loss 1.4173
    epoch    2 | step:  5800 | train_loss 1.6560
    epoch    2 | step:  6000 | train_loss 1.6843
    epoch    2 | step:  6200 | train_loss 1.5893
    epoch    2 | step:  6400 | train_loss 1.6546
    epoch    2 | step:  6600 | train_loss 1.6523
    epoch    2 | step:  6800 | train_loss 1.5513
    epoch    2 | step:  7000 | train_loss 1.3917
    epoch    2 | step:  7200 | train_loss 1.5078
    epoch    2 | step:  7400 | train_loss 1.6213
    epoch    2 | step:  7600 | train_loss 1.5623
    epoch    2 | step:  7800 | train_loss 1.6296
    epoch    2 | step:  8000 | train_loss 1.6375
    epoch    2 | step:  8200 | train_loss 1.5110
    epoch    2 | step:  8400 | train_loss 1.5071
    epoch    2 | step:  8600 | train_loss 1.4456
    epoch    2 | step:  8800 | train_loss 1.4847
    epoch    2 | step:  9000 | train_loss 1.4805
    epoch    2 | step:  9200 | train_loss 1.8518
    epoch    2 | step:  9400 | train_loss 1.5882
    epoch    2 | step:  9600 | train_loss 1.7177
    epoch    2 | step:  9800 | train_loss 1.5978
    epoch    2 | step: 10000 | train_loss 1.7331
    epoch    2 | step: 10200 | train_loss 1.6526
    epoch    2 | step: 10400 | train_loss 1.6067
    epoch    2 | step: 10600 | train_loss 1.4300
    epoch    2 | step: 10800 | train_loss 1.5442
    epoch    2 | step: 11000 | train_loss 1.5847
    epoch    2 | step: 11200 | train_loss 1.6772
    epoch    2 | step: 11400 | train_loss 1.6368
    epoch    2 | step: 11600 | train_loss 1.5549
    epoch    2 | step: 11800 | train_loss 1.5993
    epoch    2 | step: 12000 | train_loss 1.5827
    epoch    2 | step: 12200 | train_loss 1.4898
    epoch    2 | step: 12400 | train_loss 1.4093
    epoch    2 | step: 12600 | train_loss 1.5575
    epoch    2 | step: 12800 | train_loss 1.5254
    epoch    2 | step: 13000 | train_loss 1.3576
    epoch    2 | step: 13200 | train_loss 1.6967
    epoch    2 | step: 13400 | train_loss 1.4861
    epoch    2 | step: 13600 | train_loss 1.3442
    epoch    2 | step: 13800 | train_loss 1.5555
    epoch    2 | step: 14000 | train_loss 1.4729
    epoch    2 | step: 14200 | train_loss 1.6182
    epoch    2 | step: 14400 | train_loss 1.4084
    epoch    2 | step: 14600 | train_loss 1.5613
    epoch    2 | step: 14800 | train_loss 1.5449
    epoch    2 | step: 15000 | train_loss 1.4826
    epoch    2 | step: 15200 | train_loss 1.7067
    epoch    2 | step: 15400 | train_loss 1.4150
    epoch    2 | step: 15600 | train_loss 1.7125
    epoch    2 | step: 15800 | train_loss 1.6491
    epoch    2 | step: 16000 | train_loss 1.4633
    epoch    2 | step: 16200 | train_loss 1.4590


      0%|          | 3/1160 [00:00<01:50, 10.50it/s]

    Input: Spot. Spot saw the shiny car and said, "Wow, Kitty, your car is so bright and clean!" Kitty smiled and replied, "Thank you, Spot. I polish it every day."
    
    After playing with the car, Kitty and Spot felt thirsty. They found a small pond with clear water. They drank the water and felt very happy. They played together all day and became best friends.Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and hills. One day, Roxy found an icy hill. She had never seen anything like it before. It was shiny and cold, and she wanted to climb it.
    
    Roxy tried to climb the icy hill, but it was very slippery. She tried again and again, but she kept falling down. Roxy was sad. She wanted to climb the icy hill so much. Then, she saw a little bird named Billy. Billy saw that Roxy was sad and asked, "Why are you sad, Roxy?"
    
    Roxy told Billy about the icy hill and how she couldn't climb it. Billy said, "I have an idea! Let's find some big leaves to put under your feet. They will help you climb the icy hill." Roxy and Billy looked for big leaves and found some. Roxy put the leaves under her feet and tried to climb the icy hill again.
    
    This time, Roxy didn't slip. She climbed and climbed until she reached the top of the icy hill. Roxy was so happy! She and Billy played on the icy hill all day. From that day on, Roxy and Billy were the best of friends, and they climbed and played together all the time. And Roxy learned that with a little help from a friend, she could climb anything.Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was Daisy. Daisy was very small, but she was also very happy.
    
    One day, Daisy saw a dog. The dog was big and had a name too. His name was Max. Max liked to play in the yard. Daisy liked to watch Max play. Max and Daisy became friends.
    
    Every day, Max would come to the yard to play. Daisy would watch and smile. They were very happy together. And even though Daisy was small, she knew that she had a big friend in Max.Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around the house. One day, her mom asked her to wipe the table after they ate their lunch. Sue was happy to help.
    
    As Sue was wiping the table, she saw a pretty candle on the window sill. The candle was her mom's favorite. Sue wanted to do something nice for her mom, so she said, "Mom, can I light the candle for you?" Her mom said, "Yes, but be very careful."
    
    Sue carefully lit the candle and put it on the table. Her mom was so happy to see the pretty candle. They both sat and watched the candle burn. Sue's mom said, "Thank you, Sue, for being so thoughtful and careful." Sue felt proud that she could help her mom.
    
    The moral of the story is to always be thoughtful and careful when helping others.Once upon a time, there was a kind farmer. He had a big cow. The cow was sad. The farmer did not know why.
    
    One day, a little boy came to the farm. He saw the sad cow. The boy kneeled down to talk to the cow. "Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a friend."
    
    The kind farmer heard the cow. He wanted to help. So, he got another cow to be friends with the sad cow. The sad cow was happy now. They played together every day. And the kind farmer, the little boy, and the two cows all lived happily ever after.Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom. They loved to play together in the big green park near their house. One sunny day, they went to the park to play.
    
    While playing, Tom saw a big sour lemon on the ground. He wanted to play with it, but when he touched it, it started to roll away. Tom ran after the lemon, trying to catch it. But as he ran, Tom got lost in the park. Lucy looked around, but she could not find Tom. She was very sad.
    
    Lucy did not give up. She searched the park for her friend. At last, she found him near a big tree. Tom was trying to catch the lemon, but it vanished into a hole in the ground. Tom was happy to see Lucy again
    Output: , You was Spot ball thing and wanted, "Wow, that! that car is so pretty and shiny. Kitty smiled and said, "Thank you, Spot! I like my so day."
    
    From they with the shiny, Spot and Spot went very. They went a big pond and water water. Kitty drank the water and felt happy happy. Kitty knew with and day, became best friends.Once upon a time, there a small forest, there was a littleinoceros named Remyino. Roxy was to play trees One would trees, flowers, and even. R day, Roxy saw a o pond. She wanted never seen anything like it before.
     was very and pretty, and R wanted to climb it.
    
    Roxy started to climb the icy hill, but it was too hard. She slipped to and again, but she could falling.. Sheoxy was getting and She wanted to climb the icy hill, badly. But, she had a big bird on Ch. Billy was R Roxy was sad and asked, "Why are you sad, Roxy?"
    
    Roxy told Billy about the icy hill. how she wanted't climb it. Billy said, "Don can an idea! Let's go a water rocks and help on!" feet." Then will help you."." icy hill." Sooxy and Billy worked around leaves leaves and found many big Theyoxy was them leaves on her feet and they to climb the icy hill.. This
    When time, sheoxy and't give. She climbed the climbed until she reached the top. the icy hill. Sheoxy was so happy and She thanked Billy played together the icy hill all day long They that day on, Roxy and Billy were the best of friends. and they always the climbed together every the time.Once theyoxy learned that sometimes a little help from her friend, she could do the.Once upon a time, there a small house, there was a little,isy. The daisy was many big. The name was Lily. Daisy loved a happy and but she was very very pretty.
    
    One day, Daisy met a big. The dog was very and brown a long.. Daisy name was Max. Max was to play with the yard. He wanted to play Max run.
     liked Daisy liked good.
    
    One day, Daisy would come to the yard. play. Daisy would run Max smile. Max would very happy together. And Max though Max was small, she was Max Max was a friend heart like Max.Once upon a time, there was a little little named Lily. Sue loved to help her mom in the house. One day, her mom asked her to help the table. she were. food. Sue was very to help.
    
    S Sue was wiping the table, she saw a little butterfly on the table.. She candle was so favorite's birthday color Sue wanted to touch the fun for her mom. so she asked, "Mom, can I help the candle?" you?" Her mom smiled, "Yes, you be careful careful.
    
    Sue took wiped the candle and put it on the table. She mom was so happy and see the pretty candle. Sue both smiled down enjoyed the pretty together. Sue felt mom said, "Thank you, Sue, for helping so thoughtful and helping with Sue smiled proud and she could help her mom andOnce
    From moral of the story is to always be kind and help with helping others.Once upon a time, there was a little girl named He had a big farm named The cow was very because The farmer wanted not know how. He
    One day, the little girl came to the farm. He saw the cow cow. The boy wantedeled down and talk to the cow. TheWhy are you sad?" cow?" asked asked. The cow said, "I am sad. I cannot to friend."
    
    The boy farmer wanted the boy and He kne to help the He, he kne a cow. talk happy.. cow cow. The cow cow was not.. The played together all day.Once they cow farmer was the cow boy, and the cow were were lived happily ever after.Once upon a time, there was a little girl named Lily. She loved a big dog named Fl. Tom loved to play together in the park park yard. their house. One day day, Lucy went to the park to play.
    
    At they, Lucy saw a big, apple on the ground. He wanted to eat with it, but Lucy he tried it, it made to shrink away. Lucy was after it lemon, but to get it. Lucy he he ran, he tri too in the mud.
     was everywhere, but she could not find Tom.
     was very sad and
    
    Thency's not give up. She kept high park, Tom cat, Finally last, she found Tom hiding a big tree. Tom was so to get the lemon, but he was. the bush. the ground. Lucy was very to see his again.


    100%|██████████| 1160/1160 [01:46<00:00, 10.86it/s]


    epoch: 2 | val loss: 1.5028
    epoch    3 | step:     0 | train_loss 1.5217
    epoch    3 | step:   200 | train_loss 1.4996
    epoch    3 | step:   400 | train_loss 1.4721
    epoch    3 | step:   600 | train_loss 1.4095
    epoch    3 | step:   800 | train_loss 1.4377
    epoch    3 | step:  1000 | train_loss 1.4539
    epoch    3 | step:  1200 | train_loss 1.5956
    epoch    3 | step:  1400 | train_loss 1.6113
    epoch    3 | step:  1600 | train_loss 1.6243
    epoch    3 | step:  1800 | train_loss 1.7765
    epoch    3 | step:  2000 | train_loss 1.4571
    epoch    3 | step:  2200 | train_loss 1.4983
    epoch    3 | step:  2400 | train_loss 1.3908
    epoch    3 | step:  2600 | train_loss 1.6217
    epoch    3 | step:  2800 | train_loss 1.7734
    epoch    3 | step:  3000 | train_loss 1.4773
    epoch    3 | step:  3200 | train_loss 1.4317
    epoch    3 | step:  3400 | train_loss 1.5083
    epoch    3 | step:  3600 | train_loss 1.6837
    epoch    3 | step:  3800 | train_loss 1.3481
    epoch    3 | step:  4000 | train_loss 1.5909
    epoch    3 | step:  4200 | train_loss 1.4300
    epoch    3 | step:  4400 | train_loss 1.3238
    epoch    3 | step:  4600 | train_loss 1.6949
    epoch    3 | step:  4800 | train_loss 1.6250
    epoch    3 | step:  5000 | train_loss 1.5389
    epoch    3 | step:  5200 | train_loss 1.6320
    epoch    3 | step:  5400 | train_loss 1.4651
    epoch    3 | step:  5600 | train_loss 1.4990
    epoch    3 | step:  5800 | train_loss 1.5472
    epoch    3 | step:  6000 | train_loss 1.4303
    epoch    3 | step:  6200 | train_loss 1.3584
    epoch    3 | step:  6400 | train_loss 1.6956
    epoch    3 | step:  6600 | train_loss 1.5092
    epoch    3 | step:  6800 | train_loss 1.7392
    epoch    3 | step:  7000 | train_loss 1.3730
    epoch    3 | step:  7200 | train_loss 1.4100
    epoch    3 | step:  7400 | train_loss 1.5251
    epoch    3 | step:  7600 | train_loss 1.5168
    epoch    3 | step:  7800 | train_loss 1.7204
    epoch    3 | step:  8000 | train_loss 1.6030
    epoch    3 | step:  8200 | train_loss 1.5149
    epoch    3 | step:  8400 | train_loss 1.4272
    epoch    3 | step:  8600 | train_loss 1.4169
    epoch    3 | step:  8800 | train_loss 1.3818
    epoch    3 | step:  9000 | train_loss 1.5402
    epoch    3 | step:  9200 | train_loss 1.4073
    epoch    3 | step:  9400 | train_loss 1.3537
    epoch    3 | step:  9600 | train_loss 1.3239
    epoch    3 | step:  9800 | train_loss 1.6213
    epoch    3 | step: 10000 | train_loss 1.4822
    epoch    3 | step: 10200 | train_loss 1.6026
    epoch    3 | step: 10400 | train_loss 1.7485
    epoch    3 | step: 10600 | train_loss 1.5221
    epoch    3 | step: 10800 | train_loss 1.6458
    epoch    3 | step: 11000 | train_loss 1.5545
    epoch    3 | step: 11200 | train_loss 1.4383
    epoch    3 | step: 11400 | train_loss 1.4716
    epoch    3 | step: 11600 | train_loss 1.3558
    epoch    3 | step: 11800 | train_loss 1.5688
    epoch    3 | step: 12000 | train_loss 1.6664
    epoch    3 | step: 12200 | train_loss 1.7296
    epoch    3 | step: 12400 | train_loss 1.6216
    epoch    3 | step: 12600 | train_loss 1.3140
    epoch    3 | step: 12800 | train_loss 1.4315
    epoch    3 | step: 13000 | train_loss 1.6792
    epoch    3 | step: 13200 | train_loss 1.3401
    epoch    3 | step: 13400 | train_loss 1.5749
    epoch    3 | step: 13600 | train_loss 1.5701
    epoch    3 | step: 13800 | train_loss 1.4060
    epoch    3 | step: 14000 | train_loss 1.4751
    epoch    3 | step: 14200 | train_loss 1.4153
    epoch    3 | step: 14400 | train_loss 1.3747
    epoch    3 | step: 14600 | train_loss 1.4356
    epoch    3 | step: 14800 | train_loss 1.5549
    epoch    3 | step: 15000 | train_loss 1.3916
    epoch    3 | step: 15200 | train_loss 1.3845
    epoch    3 | step: 15400 | train_loss 1.5056
    epoch    3 | step: 15600 | train_loss 1.5471
    epoch    3 | step: 15800 | train_loss 1.5467
    epoch    3 | step: 16000 | train_loss 1.4295
    epoch    3 | step: 16200 | train_loss 1.3316


      0%|          | 3/1160 [00:00<01:50, 10.47it/s]

    Input: Spot. Spot saw the shiny car and said, "Wow, Kitty, your car is so bright and clean!" Kitty smiled and replied, "Thank you, Spot. I polish it every day."
    
    After playing with the car, Kitty and Spot felt thirsty. They found a small pond with clear water. They drank the water and felt very happy. They played together all day and became best friends.Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and hills. One day, Roxy found an icy hill. She had never seen anything like it before. It was shiny and cold, and she wanted to climb it.
    
    Roxy tried to climb the icy hill, but it was very slippery. She tried again and again, but she kept falling down. Roxy was sad. She wanted to climb the icy hill so much. Then, she saw a little bird named Billy. Billy saw that Roxy was sad and asked, "Why are you sad, Roxy?"
    
    Roxy told Billy about the icy hill and how she couldn't climb it. Billy said, "I have an idea! Let's find some big leaves to put under your feet. They will help you climb the icy hill." Roxy and Billy looked for big leaves and found some. Roxy put the leaves under her feet and tried to climb the icy hill again.
    
    This time, Roxy didn't slip. She climbed and climbed until she reached the top of the icy hill. Roxy was so happy! She and Billy played on the icy hill all day. From that day on, Roxy and Billy were the best of friends, and they climbed and played together all the time. And Roxy learned that with a little help from a friend, she could climb anything.Once upon a time, in a small yard, there was a small daisy. The daisy had a name. Her name was Daisy. Daisy was very small, but she was also very happy.
    
    One day, Daisy saw a dog. The dog was big and had a name too. His name was Max. Max liked to play in the yard. Daisy liked to watch Max play. Max and Daisy became friends.
    
    Every day, Max would come to the yard to play. Daisy would watch and smile. They were very happy together. And even though Daisy was small, she knew that she had a big friend in Max.Once upon a time, there was a thoughtful girl named Sue. Sue loved to help her mom around the house. One day, her mom asked her to wipe the table after they ate their lunch. Sue was happy to help.
    
    As Sue was wiping the table, she saw a pretty candle on the window sill. The candle was her mom's favorite. Sue wanted to do something nice for her mom, so she said, "Mom, can I light the candle for you?" Her mom said, "Yes, but be very careful."
    
    Sue carefully lit the candle and put it on the table. Her mom was so happy to see the pretty candle. They both sat and watched the candle burn. Sue's mom said, "Thank you, Sue, for being so thoughtful and careful." Sue felt proud that she could help her mom.
    
    The moral of the story is to always be thoughtful and careful when helping others.Once upon a time, there was a kind farmer. He had a big cow. The cow was sad. The farmer did not know why.
    
    One day, a little boy came to the farm. He saw the sad cow. The boy kneeled down to talk to the cow. "Why are you sad, cow?" he asked. The cow said, "I am lonely. I want a friend."
    
    The kind farmer heard the cow. He wanted to help. So, he got another cow to be friends with the sad cow. The sad cow was happy now. They played together every day. And the kind farmer, the little boy, and the two cows all lived happily ever after.Once upon a time, there was a little girl named Lucy. She had a pet cat named Tom. They loved to play together in the big green park near their house. One sunny day, they went to the park to play.
    
    While playing, Tom saw a big sour lemon on the ground. He wanted to play with it, but when he touched it, it started to roll away. Tom ran after the lemon, trying to catch it. But as he ran, Tom got lost in the park. Lucy looked around, but she could not find Tom. She was very sad.
    
    Lucy did not give up. She searched the park for her friend. At last, she found him near a big tree. Tom was trying to catch the lemon, but it vanished into a hole in the ground. Tom was happy to see Lucy again
    Output:  was He is a cat thing and wanted, "Wow, that! that car is so pretty and shiny. Kitty smiled and said, "Thank you, Spot! I love my every day."
    
    As playing with the shiny, Kitty and Spot went very. They went a big cup with water water. Kitty drank the water and felt happy happy. Kitty played with and day, had the friends.Once upon a time, there a small forest, there was a littleinoceros. Remyino. Roxy was to play trees She would trees, and, and even all One day, Roxy saw a unusual pond. She wanted never seen a like it before.
     was very and pretty. and R wanted to climb it.
    
    Roxy started to climb the icy hill, but it was too hard. She slipped to and again, but she could slipping down. Sheoxy was sad, She wanted to see the icy hill, badly. She, she had a big bird. Ch. Billy was R Roxy was sad and asked, "Why are you sad, Roxy?"
    
    Roxy told Billy about the icy hill. how she wanted't climb it. Billy said, "Don can an idea! Let's play a ice rocks and help on the feet." Then will help you climb the icy hill." Sooxy and Billy went and leaves leaves and found big big Theyoxy put the leaves under her feet and started to climb the icy hill.. This
    When time, Roxy climbed't give and She climbed the climbed until she reached the top. the icy hill. Sheoxy was so happy! She thanked Billy played on the icy hill all day long From that day on, Roxy and Billy became the best of friends. and they always the played together every the time.Once theyoxy learned that sometimes a little help, her friend, she could do the sheOnce upon a time, there a small town, there was a little,isy. The daisy was many friend. The name was Daisy. Daisy was a happy, but she was very very smart.
    
    One day, Daisy met a big. The dog was very and scary a long.. Daisy name was Max. Max was to play with the yard. Daisy and to run Max run.
     would Daisy became good.
    
    Da day, Daisy would come to Daisy yard to play. Daisy would run Max talk. Max would very happy together. And they though Max was small, she was that Max could a friend heart like Max.Once upon a time, there was a little little named Lily. Sue loved to help her mom in the house. One day, her mom asked her to help the table. she finished. lunch. Sue was very to help.
    
    S Sue wiped wiping the table, she saw a big v on the table.. She candle was glowing favorite's favorite candle Sue wanted to help the special for her mom. so she asked, "Mom, can I help the candle?" you?" Her mom smiled, "Yes, you be careful careful."
    
    Sue lit wiped the candle and put it on the window. Her mom smiled very happy and see the candle candle. She both smiled down enjoyed the candle together. Sue felt mom said, "Thank you, Sue, for being so thoughtful and helping with Sue smiled proud and she could help her mom.Once
    From moral of the story is to always be thoughtful and help with helping others.Once upon a time, there was a little girl named He had a big farm. The cow was very. The farmer wanted not know how the He
    One day, the little girl came to the farm. He saw the sad cow. The boy wantedeled down and say to the cow. TheWhy are you sad?" cow?" the asked. The cow said, "I am sad. I want to friend to
    
    The boy farmer wanted the boy. He wanted to help the He, he kne up cow. talk happy.. little cow. The cow cow was happy.. The played together and day. The they cow farmer was the cow boy, and the cow cows were lived happily ever after.Once upon a time, there was a little girl named Lily. She loved a big cat named Fl. Tom loved to play together. the park yard yard. their house. One day day, Lucy found to the park to play.
    
    While they, Lucy saw a big, apple on the ground. He wanted to eat with it, but Lucy he tried it, it made to melt away. Lucy was after the lemon, but to catch it. He the he ran, he tri lost in the mud.
     was everywhere and but she could not find Tom.
     was very sad.
    
    Lucy's not give up. She kept high park and Tom cat Tom She last, she found Tom near a big tree. Tom was so to get the lemon, but he was. thin tiny. the tree. Lucy was very to find his,.


    100%|██████████| 1160/1160 [01:46<00:00, 10.85it/s]


    epoch: 3 | val loss: 1.4441
    epoch    4 | step:     0 | train_loss 1.3455
    epoch    4 | step:   200 | train_loss 1.5556
    epoch    4 | step:   400 | train_loss 1.3295
    epoch    4 | step:   600 | train_loss 1.4188
    epoch    4 | step:   800 | train_loss 1.4042
    epoch    4 | step:  1000 | train_loss 1.3990
    epoch    4 | step:  1200 | train_loss 1.3903
    epoch    4 | step:  1400 | train_loss 1.4777
    epoch    4 | step:  1600 | train_loss 1.4189
    epoch    4 | step:  1800 | train_loss 1.4376
    epoch    4 | step:  2000 | train_loss 1.3898
    epoch    4 | step:  2200 | train_loss 1.4327
    epoch    4 | step:  2400 | train_loss 1.5217
    epoch    4 | step:  2600 | train_loss 1.4153
    epoch    4 | step:  2800 | train_loss 1.4220
    epoch    4 | step:  3000 | train_loss 1.3315


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
