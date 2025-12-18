---
layout: post
title: 'From Words to Meaning: Implementing Word2Vec from Scratch'
date: '2025-12-17 06:41:29 '
categories:
- Technology
tags:
- Jupyter
- Notebook
description: 'Word embeddings are one of the most transformative developments in Natural
  Language Processing (NLP). They solve a fundamental problem: how can we rep...'
image: /assets/img/from-words-to-meaning-implementing-word2vec-from-scratch/cover.png
image_alt: 'From Words to Meaning: Implementing Word2Vec from Scratch'
math: true
mermaid: true
pin: false
toc: true
comments: true
---

# From Words to Meaning: Implementing Word2Vec from Scratch

Word embeddings are one of the most transformative developments in Natural Language Processing (NLP). They solve a fundamental problem: how can we represent words as numerical vectors that capture their meaning and relationships?

## Why We Need Numerical Word Representations

Machine learning models, including neural networks, classifiers, and other algorithms, operate on numerical data. They can't directly process raw text like "cat" or "dog". To use words in these models, we must convert them into numerical vectors.

This conversion is crucial because it enables:

- **Mathematical operations**: Computing similarities, distances, and transformations
- **Learning from data**: Training models to recognize patterns and relationships
- **Generalization**: Understanding semantic relationships beyond the training examples

The challenge is finding a representation that not only converts words to numbers, but also preserves and captures their semantic meaning and relationships.

## The Problem with Traditional Representations

One common approach is **one-hot encoding**, where each word is represented by a sparse vector with a single 1 and all other elements as 0.

For example, if we have a vocabulary of 5 words `["dog", "cat", "car", "truck", "bird"]`, the one-hot encoding for "cat" would be:

```
[0, 1, 0, 0, 0]
```

This representation has critical limitations:

1. **No semantic relationships**: The vectors for "cat" and "dog" are as different from each other as "cat" and "car", even though cats and dogs are semantically similar. The distance between any two different words is always the same.

2. **High dimensionality**: For a vocabulary of size V, each word requires a V-dimensional vector. With large vocabularies (tens or hundreds of thousands of words), this becomes computationally expensive.

3. **Sparsity**: Each vector is mostly zeros, which wastes memory and computation.

## The Solution: Word Embeddings

Word embeddings solve these problems by learning **dense, low-dimensional vector representations** where:

- **Similar words have similar vectors**: Words that share meaning or context are positioned close together in the vector space. For example, "cat" and "dog" will have vectors that are close to each other, while "cat" and "car" will be farther apart.

- **Semantic relationships are captured**: The vector space encodes relationships like analogies (e.g., "king" - "man" + "woman" ≈ "queen") and semantic similarity.

- **Efficient representation**: Instead of V dimensions (where V is vocabulary size), we use a fixed, much smaller number of dimensions (typically 50-few thousands), making the representation both dense and computationally efficient.

## Word2Vec: A Breakthrough in Static Embeddings

Word2Vec, introduced by Mikolov et al. in 2013, was a breakthrough that made high-quality word embeddings accessible and practical. It learns static embeddings where each word gets a single, fixed vector representation regardless of context.

While modern transformer models (like BERT and GPT) learn **contextual embeddings** (where the same word can have different vectors depending on context), Word2Vec-style static embeddings remain valuable for many applications and provide an excellent foundation for understanding how neural networks can learn meaningful word representations.

In this post, we will implement Word2Vec from scratch using Python and PyTorch, learning how to generate word embeddings that capture semantic relationships from raw text.

## The Idea Behind Word2Vec

> "You shall know a word by the company it keeps" - J.R. Firth

The fundamental insight behind Word2Vec is the **distributional hypothesis**: words that appear in similar contexts tend to have similar meanings. If two words frequently appear near the same surrounding words, they likely share semantic properties.

For example, consider the sentences:
- "The **cat** sat on the mat"
- "The **dog** sat on the mat"

Since "cat" and "dog" appear in similar contexts (both before "sat on the mat"), they should have similar vector representations. This is exactly what Word2Vec learns to do.

### How Word2Vec Works

Word2Vec learns word embeddings by training a neural network to predict words from their context. The key insight is that **by learning to predict context words, the model must learn meaningful word representations** such that words that need to predict similar contexts will naturally develop similar vectors.

Word2Vec offers two architectures:

1. **Skip-Gram**: Given a target word, predict the surrounding context words. For example, given "cat", predict words like "the", "sat", "on", "mat" that appear nearby.

2. **CBOW (Continuous Bag of Words)**: Given the surrounding context words, predict the target word. For example, given "the", "sat", "on", "mat", predict "cat".

Both approaches learn embeddings that capture semantic relationships, but they optimize from different directions. We will implement the **Skip-Gram model**, which is often preferred for larger datasets and tends to produce better embeddings for infrequent words.

## Math Behind Word2Vec

Let's build up the mathematical formulation of Word2Vec step by step, starting with the intuitive goal and working our way to the optimization objective.

### The Intuitive Goal

For Skip-Gram, we want to maximize the probability of correctly predicting context words given a target word. If we see the word "cat" in our text, we want our model to assign high probability to words like "the", "sat", "on", "mat" that typically appear near "cat".

### The Objective Function

Given a sequence of training words $w_1, w_2, w_3, \ldots, w_T$, for each target word $w_t$, we want to maximize the probability of predicting all words within a context window of size $c$. 

For a single target word $w_t$, we want to maximize:
$$
\prod_{-c \leq j \leq c, j \neq 0} p(w_{t+j} \mid w_t)
$$

This is the probability of predicting all context words $w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c}$ given the target word $w_t$.

Taking the logarithm (which converts products to sums and is numerically more stable), we get:
$$
\sum_{-c \leq j \leq c, j \neq 0} \log p(w_{t+j} \mid w_t)
$$

Averaging over all positions in the training sequence, our objective becomes:
$$
\frac{1}{T} \sum_{t=1}^{T} \sum_{-c\leq j\leq c, j\neq 0} \log p(w_{t+j} \mid w_t)
$$

**Goal**: Maximize this quantity (or equivalently, minimize its negative, which becomes our loss function).

### Modeling the Conditional Probability

Now we need to define $p(w_O \mid w_I)$ : The probability of seeing output word $w_O$ given input word $w_I$. 

Word2Vec uses a simple but powerful approach: it represents each word with two vectors:
- **Input embedding** $v_{w_I}$: used when the word is the target (input)
- **Output embedding** $v'_{w_O}$: used when the word is in the context (output)

The similarity between a target word and a context word is measured by their dot product: $(v'_{w_O})^{\top} v_{w_I}$. Higher dot product means higher similarity, which should correspond to higher probability.

To convert these similarity scores into probabilities (which must sum to 1 over all possible context words), we use the **softmax function**:

$$
p(w_O \mid w_I) = \frac{\exp((v'_{w_O})^{\top} v_{w_I})}{\sum_{w=1}^V \exp((v'_{w})^{\top} v_{w_I})}
$$

The numerator
$$
\exp((v'_{w_O})^{\top} v_{w_I})
$$ 
gives higher probability to words with high similarity. The denominator 
$$
\sum_{w=1}^V \exp((v'_{w})^{\top} v_{w_I})
$$ normalizes over all $V$ words in the vocabulary, ensuring probabilities sum to 1.

### The Computational Challenge

Here's the problem: **computing the softmax is expensive**. This is fundamentally a multi-class classification problem: given a target word, we need to assign higher probability to the actual context words and lower probability to all other words in the vocabulary.

For each training example, computing the softmax requires:
1. Compute dot products with all $V$ words in the vocabulary
2. Exponentiate all $V$ values
3. Sum them for normalization

With a vocabulary of 50,000 words and millions of training examples, this becomes computationally prohibitive. Each gradient update would require $O(V)$ operations, making training extremely slow.

### Negative Sampling 

Instead of comparing the target word against all $V$ words every time (expensive), we can approximate this by sampling. Over many training iterations, we'll eventually compare against most words, but we only need to compute a few at a time.

**Negative sampling** transforms the multi-class classification problem into a simpler binary classification task:
- **Positive examples**: (target word, actual context word) pairs that appear together in the text
- **Negative examples**: (target word, random word) pairs that don't appear together

Instead of computing probabilities over all $V$ words, we:
1. Maximize the probability of the actual context word (positive example)
2. Minimize the probability of $k$ randomly sampled words (negative examples)

**Mathematical Formulation**: We model this as binary classification where $D$ is a binary indicator:
- $D=1$ means words $w_I$ and $w_O$ appear together (positive pair)
- $D=0$ means they don't appear together (negative pair)

The probability of a positive pair is:
$$
p(D=1 \mid w_I, w_O) = \sigma((v'_{w_O})^{\top} v_{w_I})
$$

The probability of a negative pair is:
$$
p(D=0 \mid w_I, w_O) = \sigma(-(v'_{w_O})^{\top} v_{w_I})
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function. The negative sign in $p(D=0)$ ensures that words with low similarity (low dot product) get low probability.

**Why This Approximates Softmax**: 
- The softmax compares the target word against all $V$ words simultaneously
- Negative sampling compares against $k$ random words at a time
- Over many training iterations with different random samples, we effectively compare against most words in the vocabulary
- This stochastic approximation achieves similar learning while being orders of magnitude faster

For a target word $w_I$ and context word $w_O$, the objective becomes:


$$
\log \sigma((v'_{w_O})^{\top} v_{w_I}) + \sum_{i=1}^{k} \log \sigma(-(v'_{w_i})^{\top} v_{w_I})
$$

where:
- The first term $\log \sigma((v'_{w_O})^{\top} v_{w_I})$ maximizes the probability that the actual context word $w_O$ appears with $w_I$
- The second term $\sum_{i=1}^{k} \log \sigma(-(v'_{w_i})^{\top} v_{w_I})$ minimizes the probability that $k$ randomly sampled words $w_i$ appear with $w_I$

**Efficiency Gain**: Instead of computing over all $V$ words, we only need to compute $k+1$ dot products (typically $k=5$ to $20$), reducing complexity from $O(V)$ to $O(k)$, which is orders of magnitude faster!

### Step 6: Why This Works

By learning to distinguish real context words from random words, the model must learn embeddings where:
- Words that appear together have high dot products (similar vectors)
- Words that don't appear together have low dot products (dissimilar vectors)

This naturally leads to the semantic clustering we want: similar words end up with similar embeddings. The stochastic sampling of negative words over many iterations ensures that the model eventually learns to distinguish the target word from most other words in the vocabulary, approximating the effect of the full softmax.

### Alternative: Hierarchical Softmax

Another approach to avoid the full softmax is **hierarchical softmax**, which uses a binary tree structure over the vocabulary. Instead of computing over all words, it follows a path through the tree (requiring only $\log_2(V)$ operations). However, negative sampling is simpler to implement and often performs better in practice, which is why we'll use it in our implementation.












## Setup 

Let's start by importing the necessary libraries and loading the data.


```python
import os
import re
import math
import random
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import collections
```

```python
import nltk
nltk.download('punkt_tab')
```

    [nltk_data] Downloading package punkt_tab to
    [nltk_data]     /Users/ravi.mandliya/nltk_data...
    [nltk_data]   Package punkt_tab is already up-to-date!





    True



## Download and Prepare the Data
We will use the text8 dataset, which is a preprocessed version of the Wikipedia corpus.

```python
DATA_URL = 'https://mattmahoney.net/dc/text8.zip'
DATA_DIR = './data/'
ZIP_FILE_NAME = 'text8.zip'
TEXT_FILE_NAME = 'text8'

ZIP_FILE_PATH = os.path.join(DATA_DIR, ZIP_FILE_NAME)
TEXT_FILE_PATH = os.path.join(DATA_DIR, TEXT_FILE_NAME)

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Download and extract if the text file doesn't exist
if not os.path.exists(TEXT_FILE_PATH):
    print(f"Downloading {ZIP_FILE_NAME}...")
    urllib.request.urlretrieve(DATA_URL, ZIP_FILE_PATH)
    
    print(f"Extracting {ZIP_FILE_NAME}...")
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print(f"Data extracted to {DATA_DIR}")
else:
    print(f"Text file already exists at {TEXT_FILE_PATH}")

# Read and tokenize the text
print(f"Reading text from {TEXT_FILE_PATH}...")
with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()
    words = word_tokenize(text)

print(f"Total words: {len(words)}")
print(f"First 100 words: {words[:100]}")
```

    Text file already exists at ./data/text8
    Reading text from ./data/text8...
    Total words: 17007698
    First 100 words: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against', 'early', 'working', 'class', 'radicals', 'including', 'the', 'diggers', 'of', 'the', 'english', 'revolution', 'and', 'the', 'sans', 'culottes', 'of', 'the', 'french', 'revolution', 'whilst', 'the', 'term', 'is', 'still', 'used', 'in', 'a', 'pejorative', 'way', 'to', 'describe', 'any', 'act', 'that', 'used', 'violent', 'means', 'to', 'destroy', 'the', 'organization', 'of', 'society', 'it', 'has', 'also', 'been', 'taken', 'up', 'as', 'a', 'positive', 'label', 'by', 'self', 'defined', 'anarchists', 'the', 'word', 'anarchism', 'is', 'derived', 'from', 'the', 'greek', 'without', 'archons', 'ruler', 'chief', 'king', 'anarchism', 'as', 'a', 'political', 'philosophy', 'is', 'the', 'belief', 'that', 'rulers', 'are', 'unnecessary', 'and', 'should', 'be', 'abolished', 'although', 'there', 'are', 'differing']


## Build the Vocabulary
We'll create mappings between words and integer IDs and remove rare words to keep vocabulary manageable. The code is below followed by the explanation.




```python
class Word2VecDataset(Dataset):
  def __init__(self, words, word_to_idx, word_freqs, window_size=5,
               negative_sample_counts=5, subsample_threshold=1e-3):
    # words: already tokenized list of words (from Word2Vec class)
    # word_to_idx: {word: idx} mapping from Word2Vec class
    # word_freqs: {word: frequency} mapping from Word2Vec class (already computed)
    self.words = words
    self.word_to_idx = word_to_idx
    self.word_freqs = word_freqs  # Use precomputed frequencies from Word2Vec
    self.vocab_size = len(word_to_idx)  # Vocabulary size for negative sampling
    self.window_size = window_size
    self.negative_sample_counts = negative_sample_counts
    self.subsample_threshold = subsample_threshold

    # Convert words to indices, filtering out words not in vocabulary
    self.encoded_text = [self.word_to_idx[word] for word in words if word in self.word_to_idx]

    # Compute word frequencies by index for subsampling
    # word_freqs is {word: frequency}, we need {idx: frequency}
    self.word_freqs_by_idx = {word_to_idx[word]: freq for word, freq in word_freqs.items() if word in word_to_idx}

    # subsample probability: probability to keep each word
    # Formula: P(keep) = min(1.0, sqrt(t / f(w))) where t is threshold, f(w) is frequency
    # Rare words (f < t): P(keep) = 1.0 (always kept)
    # Frequent words (f > t): P(keep) = sqrt(t/f) < 1.0 (discarded more often)
    self.subsample_probs = {
      idx: min(1.0, np.sqrt(self.subsample_threshold / freq)) if freq > 0 else 1.0
      for idx, freq in self.word_freqs_by_idx.items()
    }
    
    self.pairs = []
    np.random.seed(42)  # Fixed seed for deterministic subsampling
    random.seed(42)
    
    for i, center_word in enumerate(self.encoded_text):
      # Apply subsampling deterministically
      if center_word not in self.subsample_probs:
        continue
      if np.random.rand() >= self.subsample_probs[center_word]:
        continue
      
      # Generate all context pairs for this center word
      # Use random window size for variety (but deterministic with fixed seed)
      context_window = random.randint(1, self.window_size)
      start = max(0, i - context_window)
      end = min(len(self.encoded_text), i + context_window + 1)
      
      for j in range(start, end):
        if i != j:
          self.pairs.append((center_word, self.encoded_text[j]))
    
    # Reset random seed for training randomness
    np.random.seed()
    random.seed()
    
    self.negative_sample_weights = self._generate_negative_sample_weights()

  def _generate_negative_sample_weights(self):
    # Use vocab_size to ensure correct array size for negative sampling
    # Convert word frequencies to counts for negative sampling distribution
    # Only use words that are in the vocabulary
    total_count = len(self.words)
    freq = np.zeros(self.vocab_size)
    for word in self.word_to_idx.keys():
      if word in self.word_freqs:
        idx = self.word_to_idx[word]
        word_freq = self.word_freqs[word]
        # Convert frequency back to count
        freq[idx] = word_freq * total_count
      else:
        # If word is in vocab but not in word_freqs (shouldn't happen, but handle gracefully)
        idx = self.word_to_idx[word]
        freq[idx] = 0.0
    
    # Apply power law: freq^0.75 (standard Word2Vec approach)
    # This raises each word's count to the 3/4 power, which helps balance
    # the distribution between frequent and rare words
    freq = np.power(freq, 0.75)
    # Normalize to get probability distribution
    freq = freq / np.sum(freq)
    return torch.FloatTensor(freq)

  def __len__(self):
    return len(self.pairs)

  def __getitem__(self, idx):
    # Simple O(1) lookup - pairs are already generated
    target, context = self.pairs[idx]
    
    # Generate negative samples on-the-fly
    negs = torch.multinomial(
      self.negative_sample_weights,
      self.negative_sample_counts,
      replacement=True
    )
    
    return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long), negs
```

## Building the Word2Vec Dataset

The `Word2VecDataset` class transforms raw text into training examples for the Skip-Gram model. This process involves several key steps, each with important mathematical foundations. Let's break down how the dataset is constructed:

### Step 1: Encoding Text to Indices

First, we convert words to integer indices for efficient processing:

```python
self.encoded_text = [self.word_to_idx[word] for word in text if word in self.word_to_idx]
```

This creates a sequence of integer IDs, filtering out words not in our vocabulary.

### Step 2: Subsampling Frequent Words

Word2Vec uses **subsampling** to balance the training data. Very frequent words (like "the", "a", "of") provide less semantic information but dominate the training signal. Subsampling randomly discards these words with probability proportional to their frequency.

**Mathematical Formulation**: For a word $w$ with frequency $f(w)$ in the corpus, the probability of **keeping** the word is:

$$
P(\text{keep } w) = \min\left(1.0, \sqrt{\frac{t}{f(w)}}\right)
$$

where $t$ is the subsampling threshold (typically $10^{-3}$ to $10^{-5}$).

**Intuition**:
- **Rare words** ($f(w) < t$): $P(\text{keep}) = 1.0$ : Always kept, preserving important semantic information
- **Frequent words** ($f(w) > t$): $P(\text{keep}) = \sqrt{t/f(w)} < 1.0$ : Discarded more often, reducing their dominance

For example, if $t = 0.001$ and a word appears with frequency $f(w) = 0.01$ (1% of all words):
- $P(\text{keep}) = \sqrt{0.001/0.01} = \sqrt{0.1} \approx 0.316$

This means we keep this word only about 31.6% of the time, effectively downweighting it in training.

**Why $\sqrt{t/f(w)}$?** This formula ensures that:
- Words with frequency exactly equal to the threshold are kept with probability 1.0
- The probability decreases smoothly as frequency increases
- Very frequent words (like "the") are heavily downweighted

### Step 3: Generating Training Pairs

For each center word $w_t$ at position $t$ in the (subsampled) text, we generate context pairs:

1. **Random Context Window**: Instead of a fixed window size, we use a random window size $c \in [1, \text{window\_size}]$ for each center word. This adds variety to the training data.

2. **Context Extraction**: For a center word at position $i$ with window size $c$, we extract context words from positions $[i-c, i-1]$ and $[i+1, i+c]$.

3. **Pair Generation**: Each context word $w_{t+j}$ (where $j \neq 0$ and $|j| \leq c$) is paired with the center word $w_t$ to create a training example $(w_t, w_{t+j})$.

**Example**: For the sentence "the quick brown fox jumps" with `window_size=2`:
- Center word "brown" at position 2
- Random window $c=2$ selected
- Context words: "the" (position 0), "quick" (position 1), "fox" (position 3), "jumps" (position 4)
- Generated pairs: `("brown", "the")`, `("brown", "quick")`, `("brown", "fox")`, `("brown", "jumps")`

### Step 4: Negative Sampling Distribution

Negative sampling requires a probability distribution over the vocabulary to sample "negative" (non-context) words. The original Word2Vec implementation uses a **unigram distribution raised to the 3/4 power**.

**Mathematical Formulation**: For each word $w$ with count $c(w)$ in the corpus, we compute:

$$
P_n(w) = \frac{c(w)^{3/4}}{\sum_{w' \in V} c(w')^{3/4}}
$$

where $P_n(w)$ is the probability of sampling word $w$ as a negative example, and $V$ is the vocabulary.

**Why $c(w)^{3/4}$?** The 3/4 power (0.75) is a heuristic that:
- **Reduces the dominance of very frequent words**: Without the power, words like "the" would be sampled too often as negatives
- **Increases the probability of moderately frequent words**: This helps the model learn better distinctions
- **Still favors frequent words over rare ones**: Rare words are less useful as negative examples

**Example**: If "the" appears 100,000 times and "cat" appears 1,000 times:
- Without power: $P_n(\text{the}) / P_n(\text{cat}) = 100,000 / 1,000 = 100$
- With 3/4 power: $P_n(\text{the}) / P_n(\text{cat}) = 100,000^{0.75} / 1,000^{0.75} \approx 31,623 / 178 \approx 178$

The ratio is still large (178:1), but much smaller than 100:1, giving "cat" a better chance of being sampled.

### Step 5: On-the-Fly Negative Sampling

During training, for each positive pair $(w_t, w_c)$, we sample $k$ negative words from $P_n(w)$:

```python
negs = torch.multinomial(self.negative_sample_weights, self.negative_samples, replacement=True)
```

This creates $k$ negative examples $(w_t, w_{\text{neg}_1}), \ldots, (w_t, w_{\text{neg}_k})$ where each $w_{\text{neg}_i}$ is sampled according to $P_n(w)$.

**Why on-the-fly?** Generating negative samples during `__getitem__` is efficient because:
- It avoids storing millions of pre-generated negative samples
- `torch.multinomial` is highly optimized
- The randomness helps with generalization


We created a dataset class that generates pairs of target and context words. It also generates negative samples on-the-fly.

## Define the Model

We'll use PyTorch to define the Word2Vec model. We will have two embedding matrices:
- **Input embedding matrix** $V \times d$: maps target words to their embeddings
- **Output embedding matrix** $V \times d$: maps context words to their embeddings

The model will learn two embedding matrices, one for the target words and one for the context words.
We'll use uniform initialization for the embeddings to ease the computation of the loss function.

```python
class SkipGramModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim):
    super().__init__()
    self.in_embed = nn.Embedding(vocab_size, embedding_dim)
    self.out_embed = nn.Embedding(vocab_size, embedding_dim)
    self._init_weights()

  def _init_weights(self):
    initrange = 0.5 / self.in_embed.embedding_dim
    self.in_embed.weight.data.uniform_(-initrange, initrange)
    self.out_embed.weight.data.uniform_(-initrange, initrange)

  def forward(self, center_words, context_words, negative_samples):
    center_embeds = self.in_embed(center_words) # (batch_size, embedding_dim)
    context_embeds = self.out_embed(context_words) # (batch_size, embedding_dim)
    negative_embeds = self.out_embed(negative_samples) # (batch_size, negative_samples, embedding_dim)

    # Compute the loss
    pos_score = torch.sum(center_embeds * context_embeds, dim=1) # (batch_size,)
    pos_score = torch.sigmoid(pos_score)

    # Compute dot products: (batch_size, negative_samples, embedding_dim) @ (batch_size, embedding_dim, 1)
    # center_embeds.unsqueeze(2) adds dimension: (batch_size, embedding_dim) -> (batch_size, embedding_dim, 1)
    neg_score = torch.bmm(negative_embeds, center_embeds.unsqueeze(2)) # (batch_size, negative_samples, 1)
    neg_score = torch.sigmoid(neg_score).squeeze(2) # (batch_size, negative_samples)

    return pos_score, neg_score

  def get_embedding(self, word_idx):
    return self.in_embed(torch.LongTensor([word_idx])).detach()

  def get_all_center_embeddings(self):
    return self.in_embed.weight.data

```

## Understanding the Skip-Gram Model Architecture

The `SkipGramModel` implements the core Word2Vec architecture using PyTorch. Let's break down how it works:

### Dual Embedding Matrices

Word2Vec uses **two separate embedding matrices** for each word:

- **`in_embed`**: Input embeddings used when a word is the **center/target word** (the word we're predicting context for)
- **`out_embed`**: Output embeddings used when a word is a **context word** (or negative sample)

This dual representation is a key design choice. While it doubles the number of parameters, it often leads to better embeddings because the model can learn different representations for the same word depending on its role (center vs. context).

### Weight Initialization

The embeddings are initialized uniformly in the range $[-0.5/d, 0.5/d]$ where $d$ is the embedding dimension:

```python
initrange = 0.5 / embedding_dim
```

This small initialization range helps with:
- **Stable training**: Prevents gradients from exploding early in training
- **Symmetry breaking**: Small random values ensure different words start with different embeddings
- **Numerical stability**: Keeps initial dot products in a reasonable range for the sigmoid function

### Forward Pass: Computing Similarity Scores

The `forward` method computes how well the model predicts context words:

1. **Embedding Lookup**: 
   - `center_embeds`: Gets embeddings for target words (shape: `(batch_size, embedding_dim)`)
   - `context_embeds`: Gets embeddings for actual context words (shape: `(batch_size, embedding_dim)`)
   - `negative_embeds`: Gets embeddings for negative samples (shape: `(batch_size, negative_samples, embedding_dim)`)

2. **Positive Score**: Measures similarity between center and context words:
   ```python
   pos_score = torch.sum(center_embeds * context_embeds, dim=1)
   pos_score = torch.sigmoid(pos_score)
   ```
   - Element-wise multiplication followed by sum computes the **dot product** (cosine similarity when embeddings are normalized)
   - The sigmoid converts the dot product into a probability $p \in (0, 1)$
   - Higher dot product → higher probability → model is confident these words appear together

3. **Negative Score**: Measures similarity between center words and negative samples:
   ```python
   neg_score = torch.bmm(negative_embeds, center_embeds.unsqueeze(2))
   neg_score = torch.sigmoid(neg_score).squeeze(2)
   ```
   - `torch.bmm` performs **batch matrix multiplication**: for each sample in the batch, it computes dot products between the center embedding and all negative embeddings
   - `unsqueeze(2)` adds a dimension to `center_embeds`: `(batch_size, embedding_dim)` → `(batch_size, embedding_dim, 1)` to make it compatible with batch matrix multiplication
   - The matrix multiplication: `(batch_size, negative_samples, embedding_dim) @ (batch_size, embedding_dim, 1)` = `(batch_size, negative_samples, 1)`
   - We `squeeze(2)` to remove the last dimension, getting `(batch_size, negative_samples)`
   - Each value represents how similar the center word is to a negative sample (we want this to be low)

### The Learning Objective

The model returns `pos_score` and `neg_score`, which will be used in the loss function:

$$
\mathcal{L} = -\log(\text{pos\_score}) - \sum_{i=1}^{k} \log(1 - \text{neg\_score}_i)
$$

This loss function:
- **Maximizes** `pos_score` (probability that real context words appear with the center word)
- **Minimizes** `neg_score` (probability that random words appear with the center word)

By optimizing this objective, the model learns embeddings where:
- Words that appear together in text have high dot products (similar vectors)
- Words that don't appear together have low dot products (dissimilar vectors)

## Training the Word2Vec Model

Now in this section, we will train the Word2Vec model. We will use the `Word2Vec` class to train the model. The code is below followed by the explanation.

```python
class Word2Vec:
  def __init__(self, embedding_dim=128, window_size=5, negative_samples_counts=5, min_count=5, learning_rate=0.001, num_epochs=5):
    self.embedding_dim = embedding_dim
    self.window_size = window_size
    self.negative_samples_counts = negative_samples_counts
    self.min_count = min_count
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs

    self.word_to_idx = None
    self.idx_to_word = None
    self.dataset = None

    self.model = None
    self.optimizer = None
    
  def _preprocess_text_and_build_vocab(self, text):
    # Tokenize the text once
    self.words = word_tokenize(text)
    # Count words for vocabulary building and frequency calculations
    word_counts = collections.Counter(self.words)
    # Filter words that meet min_count, then assign contiguous indices 0, 1, 2, ...
    # This ensures indices are in range [0, vocab_size-1]
    words_meeting_min_count = [(word, count) for word, count in word_counts.items() if count >= self.min_count]
    self.word_to_idx = {word: idx for idx, (word, count) in enumerate(words_meeting_min_count)}
    # Build reverse mapping: {idx: word}
    self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    # Compute word frequencies for subsampling and negative sampling
    # Only compute frequencies for words in vocabulary (meets min_count)
    total_count = len(self.words)
    self.word_freqs = {}
    for word, count in word_counts.items():
      if word in self.word_to_idx:  # Only include words in vocabulary
        self.word_freqs[word] = count / total_count
    print(f"Vocabulary size: {len(self.word_to_idx)}")

  def _compute_loss(self, pos_score, neg_score):
    # pos_score: (batch_size,)
    # neg_score: (batch_size, negative_samples)
    # Loss formula: -log(pos_score) - sum_{i=1}^k log(1 - neg_score_i)
    pos_loss = -torch.log(pos_score + 1e-10).mean()
    # Sum over negative_samples dimension first, then average over batch
    neg_loss = -torch.log(1 - neg_score + 1e-10).sum(dim=1).mean()
    return pos_loss + neg_loss

  def train(self, text, batch_size=128):
    self._preprocess_text_and_build_vocab(text)
    # Pass preprocessed data to the dataset
    # Dataset only handles training example generation, not text processing
    self.dataset = Word2VecDataset(self.words, self.word_to_idx, self.word_freqs,
      self.window_size, self.negative_samples_counts)
    dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    self.model = SkipGramModel(len(self.word_to_idx), self.embedding_dim)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


    print(f"Training for {self.num_epochs} epochs...")
    for epoch in range(self.num_epochs):
      epoch_loss = 0
      # Create tqdm wrapper inside the loop for each epoch
      tqdm_dataloader = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{self.num_epochs}")
      for idx, (target, context, negs) in enumerate(tqdm_dataloader):
        self.optimizer.zero_grad()
        pos_score, neg_score = self.model(target, context, negs)
        loss = self._compute_loss(pos_score, neg_score)
        loss.backward()
        self.optimizer.step()
        epoch_loss += loss.item()
        if (idx + 1) % 100 == 0:
          print(f"Epoch {epoch+1} Batch {idx+1} loss: {loss.item()}")
          # checkpointing
          torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss.item()
          }, f"data/checkpoint_{epoch+1}_{idx+1}.pth")
          # print closest words to the center word
          print(f"Closest words to the center word {self.idx_to_word[target[0].item()]}: {self.most_similar(self.idx_to_word[target[0].item()])}")
      avg_loss = epoch_loss / len(dataloader)
      print(f"Epoch {epoch+1} loss: {avg_loss}")

    print("Training complete!")

  def most_similar(self, word, top_k=10):
    word_idx = self.word_to_idx[word]
    embedding = self.model.get_embedding(word_idx)
    all_embeddings = self.model.get_all_center_embeddings()
    similarities = torch.matmul(embedding, all_embeddings.T) # (1, vocab_size)
    similarities = similarities.squeeze(0) # (vocab_size,)
    _, indices = torch.topk(similarities, top_k)
    return [self.idx_to_word[idx.item()] for idx in indices]

  
  def get_embedding(self, word):
    word_idx = self.word_to_idx[word]
    return self.model.get_embedding(word_idx).detach()


```

```python
if __name__ == "__main__":
  word2vec = Word2Vec(embedding_dim=128, window_size=5,
    negative_samples_counts=5, min_count=5, learning_rate=0.001, num_epochs=2)
```

```python
word2vec.train(text, batch_size=2048)
```

    Vocabulary size: 71286
    Training for 2 epochs...


    Epoch 1/2:   0%|          | 99/34343 [01:33<2:15:28,  4.21it/s] 

    Epoch 1 Batch 100 loss: 3.774440050125122


    Epoch 1/2:   0%|          | 100/34343 [01:34<3:06:45,  3.06it/s]

    Closest words to the center word stones: ['the', 'of', 'and', 'in', 'a', 'to', 'as', 'by', 'that', 'at']


    Epoch 1/2:   1%|          | 199/34343 [03:11<38:21:02,  4.04s/it]

    Epoch 1 Batch 200 loss: 3.144181489944458


    Epoch 1/2:   1%|          | 200/34343 [03:12<28:15:59,  2.98s/it]

    Closest words to the center word through: ['the', 'in', 'a', 'of', 'zero', 'on', 'at', 'and', 'first', 'is']


    Epoch 1/2:   1%|          | 299/34343 [04:30<5:17:29,  1.79it/s] 

    Epoch 1 Batch 300 loss: 2.9165852069854736


    Epoch 1/2:   1%|          | 300/34343 [04:31<5:19:34,  1.78it/s]

    Closest words to the center word that: ['part', 'number', 'time', 'used', 'be', 'same', 'state', 'use', 'such', 'known']


    Epoch 1/2:   1%|          | 399/34343 [05:48<2:37:18,  3.60it/s] 

    Epoch 1 Batch 400 loss: 2.7336912155151367


    Epoch 1/2:   1%|          | 400/34343 [05:48<3:10:16,  2.97it/s]

    Closest words to the center word president: ['nine', 'eight', 'seven', 'zero', 'six', 'one', 'four', 'five', 'two', 'three']


    Epoch 1/2:   1%|▏         | 500/34343 [06:59<2:40:33,  3.51it/s] 

    Epoch 1 Batch 500 loss: 2.6643283367156982
    Closest words to the center word special: ['eight', 'nine', 'four', 'seven', 'case', 'zero', 'name', 'republic', 'isbn', 'referred']


    Epoch 1/2:   2%|▏         | 599/34343 [08:10<18:33:53,  1.98s/it]

    Epoch 1 Batch 600 loss: 2.6351046562194824


    Epoch 1/2:   2%|▏         | 600/34343 [08:11<14:17:31,  1.52s/it]

    Closest words to the center word zero: ['isbn', 'eight', 'seven', 'nine', 'zero', 'six', 'four', 'five', 'one', 'three']


    Epoch 1/2:   2%|▏         | 699/34343 [09:21<3:55:13,  2.38it/s] 

    Epoch 1 Batch 700 loss: 2.578791856765747


    Epoch 1/2:   2%|▏         | 700/34343 [09:22<4:28:48,  2.09it/s]

    Closest words to the center word dharma: ['isbn', 'nine', 'eight', 'six', 'zero', 'seven', 'creation', 'four', 'july', 'one']


    Epoch 1/2:   2%|▏         | 799/34343 [10:30<2:06:13,  4.43it/s] 

    Epoch 1 Batch 800 loss: 2.5281713008880615


    Epoch 1/2:   2%|▏         | 800/34343 [10:30<3:08:00,  2.97it/s]

    Closest words to the center word victor: ['isbn', 'nine', 'seven', 'zero', 'eight', 'six', 'births', 'three', 'km', 'four']


    Epoch 1/2:   3%|▎         | 899/34343 [11:28<1:57:59,  4.72it/s] 

    Epoch 1 Batch 900 loss: 2.517162799835205


    Epoch 1/2:   3%|▎         | 900/34343 [11:29<3:03:03,  3.04it/s]

    Closest words to the center word radiation: ['isbn', 'nine', 'seven', 'births', 'ability', 'rest', 'six', 'result', 'zero', 'eight']


    Epoch 1/2:   3%|▎         | 999/34343 [12:52<16:58:02,  1.83s/it]

    Epoch 1 Batch 1000 loss: 2.5241129398345947


    Epoch 1/2:   3%|▎         | 1000/34343 [12:52<13:35:01,  1.47s/it]

    Closest words to the center word precocious: ['isbn', 'nine', 'zero', 'km', 'seven', 'june', 'december', 'births', 'eight', 'july']


    Epoch 1/2:   3%|▎         | 1099/34343 [13:57<2:48:34,  3.29it/s] 

    Epoch 1 Batch 1100 loss: 2.4730119705200195


    Epoch 1/2:   3%|▎         | 1100/34343 [13:58<3:36:10,  2.56it/s]

    Closest words to the center word a: ['isbn', 'not', 'zero', 'births', 'december', 'refer', 'referred', 'nine', 'used', 'be']


    Epoch 1/2:   3%|▎         | 1199/34343 [15:00<2:00:17,  4.59it/s] 

    Epoch 1 Batch 1200 loss: 2.4821088314056396


    Epoch 1/2:   3%|▎         | 1200/34343 [15:00<3:06:05,  2.97it/s]

    Closest words to the center word justice: ['isbn', 'births', 'september', 'nine', 'eight', 'december', 'km', 'zero', 'june', 'july']


    Epoch 1/2:   4%|▍         | 1299/34343 [16:03<1:55:52,  4.75it/s] 

    Epoch 1 Batch 1300 loss: 2.4716808795928955


    Epoch 1/2:   4%|▍         | 1300/34343 [16:04<3:02:40,  3.01it/s]

    Closest words to the center word least: ['isbn', 'births', 'december', 'june', 'km', 'january', 'july', 'november', 'zero', 'september']


    Epoch 1/2:   4%|▍         | 1399/34343 [17:11<6:05:14,  1.50it/s] 

    Epoch 1 Batch 1400 loss: 2.4694762229919434


    Epoch 1/2:   4%|▍         | 1400/34343 [17:12<6:05:56,  1.50it/s]

    Closest words to the center word gettier: ['isbn', 'births', 'december', 'nine', 'km', 'july', 'eight', 'january', 'seven', 'six']


    Epoch 1/2:   4%|▍         | 1499/34343 [18:19<2:48:39,  3.25it/s] 

    Epoch 1 Batch 1500 loss: 2.460878849029541


    Epoch 1/2:   4%|▍         | 1500/34343 [18:19<3:38:57,  2.50it/s]

    Closest words to the center word music: ['isbn', 'births', 'importance', 'km', 'november', 'february', 'july', 'pp', 'eight', 'basis']


    Epoch 1/2:   5%|▍         | 1599/34343 [19:22<1:59:59,  4.55it/s] 

    Epoch 1 Batch 1600 loss: 2.453460931777954


    Epoch 1/2:   5%|▍         | 1600/34343 [19:22<3:04:47,  2.95it/s]

    Closest words to the center word popular: ['referred', 'births', 'isbn', 'true', 'believed', 'rest', 'importance', 'idea', 'result', 'absence']


    Epoch 1/2:   5%|▍         | 1699/34343 [20:20<1:59:22,  4.56it/s] 

    Epoch 1 Batch 1700 loss: 2.4053971767425537


    Epoch 1/2:   5%|▍         | 1700/34343 [20:31<31:43:42,  3.50s/it]

    Closest words to the center word old: ['births', 'isbn', 'km', 'january', 'november', 'zero', 'june', 'eight', 'pp', 'th']


    Epoch 1/2:   5%|▌         | 1799/34343 [21:30<4:53:54,  1.85it/s] 

    Epoch 1 Batch 1800 loss: 2.4356462955474854


    Epoch 1/2:   5%|▌         | 1800/34343 [21:31<5:09:49,  1.75it/s]

    Closest words to the center word in: ['births', 'isbn', 'pp', 'february', 'november', 'inducted', 'june', 'eight', 'km', 'nine']


    Epoch 1/2:   6%|▌         | 1899/34343 [22:38<1:57:59,  4.58it/s] 

    Epoch 1 Batch 1900 loss: 2.4106228351593018


    Epoch 1/2:   6%|▌         | 1900/34343 [22:38<3:03:19,  2.95it/s]

    Closest words to the center word bradley: ['births', 'isbn', 'pp', 'km', 'nine', 'february', 'november', 'laureate', 'inducted', 'eight']


    Epoch 1/2:   6%|▌         | 1999/34343 [23:47<1:55:14,  4.68it/s] 

    Epoch 1 Batch 2000 loss: 2.4157354831695557


    Epoch 1/2:   6%|▌         | 2000/34343 [23:48<2:59:00,  3.01it/s]

    Closest words to the center word fermentation: ['births', 'isbn', 'pp', 'qur', 'referred', 'importance', 'rest', 'refer', 'difficult', 'easier']


    Epoch 1/2:   6%|▌         | 2099/34343 [25:11<30:41:01,  3.43s/it]

    Epoch 1 Batch 2100 loss: 2.407923698425293


    Epoch 1/2:   6%|▌         | 2100/34343 [25:12<23:08:32,  2.58s/it]

    Closest words to the center word doing: ['able', 'believed', 'know', 'referred', 'expected', 'want', 'do', 'seen', 'necessary', 'difficult']


    Epoch 1/2:   6%|▋         | 2199/34343 [26:20<2:31:25,  3.54it/s] 

    Epoch 1 Batch 2200 loss: 2.4096288681030273


    Epoch 1/2:   6%|▋         | 2200/34343 [26:20<3:25:12,  2.61it/s]

    Closest words to the center word may: ['may', 'births', 'can', 'inducted', 'there', 'must', 'would', 'than', 'pp', 'isbn']


    Epoch 1/2:   7%|▋         | 2299/34343 [27:29<2:08:58,  4.14it/s] 

    Epoch 1 Batch 2300 loss: 2.3787875175476074


    Epoch 1/2:   7%|▋         | 2300/34343 [27:30<3:14:13,  2.75it/s]

    Closest words to the center word outside: ['qur', 'rest', 'origin', 'importance', 'happens', 'prepare', 'variety', 'idea', 'collapse', 'consequence']


    Epoch 1/2:   7%|▋         | 2399/34343 [28:38<1:53:21,  4.70it/s] 

    Epoch 1 Batch 2400 loss: 2.392396926879883


    Epoch 1/2:   7%|▋         | 2400/34343 [28:38<2:52:15,  3.09it/s]

    Closest words to the center word added: ['believed', 'able', 'referred', 'inducted', 'fact', 'difficult', 'rest', 'possible', 'stored', 'know']


    Epoch 1/2:   7%|▋         | 2499/34343 [29:53<6:49:19,  1.30it/s] 

    Epoch 1 Batch 2500 loss: 2.417815685272217


    Epoch 1/2:   7%|▋         | 2500/34343 [29:53<6:20:36,  1.39it/s]

    Closest words to the center word private: ['difficult', 'referred', 'stored', 'remove', 'beings', 'able', 'quantity', 'regard', 'violation', 'sizes']


    Epoch 1/2:   8%|▊         | 2599/34343 [31:01<2:18:07,  3.83it/s] 

    Epoch 1 Batch 2600 loss: 2.3879668712615967


    Epoch 1/2:   8%|▊         | 2600/34343 [31:02<3:13:45,  2.73it/s]

    Closest words to the center word estate: ['births', 'isbn', 'inducted', 'pp', 'th', 'december', 'june', 'km', 'nine', 'january']


    Epoch 1/2:   8%|▊         | 2699/34343 [32:11<1:54:11,  4.62it/s] 

    Epoch 1 Batch 2700 loss: 2.4030895233154297


    Epoch 1/2:   8%|▊         | 2700/34343 [32:11<2:55:18,  3.01it/s]

    Closest words to the center word helmets: ['isbn', 'births', 'inducted', 'pp', 'km', 'pmid', 'laureate', 'july', 'agave', 'rfc']


    Epoch 1/2:   8%|▊         | 2799/34343 [33:34<43:04:08,  4.92s/it]

    Epoch 1 Batch 2800 loss: 2.378714084625244


    Epoch 1/2:   8%|▊         | 2800/34343 [33:34<31:51:41,  3.64s/it]

    Closest words to the center word computation: ['know', 'determine', 'does', 'stored', 'referred', 'difficult', 'argue', 'reason', 'believe', 'belief']


    Epoch 1/2:   8%|▊         | 2899/34343 [34:30<3:25:10,  2.55it/s] 

    Epoch 1 Batch 2900 loss: 2.374361276626587


    Epoch 1/2:   8%|▊         | 2900/34343 [34:31<4:04:08,  2.15it/s]

    Closest words to the center word mostly: ['births', 'fact', 'inducted', 'isbn', 'pp', 'referred', 'considered', 'treated', 'argue', 'understood']


    Epoch 1/2:   9%|▊         | 2999/34343 [35:16<2:00:09,  4.35it/s] 

    Epoch 1 Batch 3000 loss: 2.3956961631774902


    Epoch 1/2:   9%|▊         | 3000/34343 [35:16<3:06:41,  2.80it/s]

    Closest words to the center word relationship: ['lack', 'distinguish', 'suited', 'fact', 'belief', 'treated', 'regard', 'stored', 'easy', 'kind']


    Epoch 1/2:   9%|▉         | 3099/34343 [36:00<1:51:54,  4.65it/s] 

    Epoch 1 Batch 3100 loss: 2.348349094390869


    Epoch 1/2:   9%|▉         | 3100/34343 [36:01<3:00:04,  2.89it/s]

    Closest words to the center word absent: ['births', 'inducted', 'isbn', 'years', 'december', 'pp', 'gwh', 'know', 'expected', 'september']


    Epoch 1/2:   9%|▉         | 3199/34343 [37:16<30:25:10,  3.52s/it]

    Epoch 1 Batch 3200 loss: 2.376477003097534


    Epoch 1/2:   9%|▉         | 3200/34343 [37:17<22:53:02,  2.65s/it]

    Closest words to the center word kahn: ['laureate', 'births', 'isbn', 'inducted', 'pp', 'gwh', 'pmid', 'nobel', 'actress', 'nine']


    Epoch 1/2:  10%|▉         | 3299/34343 [38:26<4:09:28,  2.07it/s] 

    Epoch 1 Batch 3300 loss: 2.341855764389038


    Epoch 1/2:  10%|▉         | 3300/34343 [38:26<4:33:14,  1.89it/s]

    Closest words to the center word lit: ['frac', 'laureate', 'mathbf', 'x', 'y', 'births', 'isbn', 'inducted', 'you', 'f']


    Epoch 1/2:  10%|▉         | 3399/34343 [39:36<1:57:18,  4.40it/s] 

    Epoch 1 Batch 3400 loss: 2.33795166015625


    Epoch 1/2:  10%|▉         | 3400/34343 [39:36<2:58:23,  2.89it/s]

    Closest words to the center word switched: ['births', 'laureate', 'inducted', 'isbn', 'able', 'pmid', 'remove', 'january', 'february', 'know']


    Epoch 1/2:  10%|█         | 3499/34343 [40:45<1:49:59,  4.67it/s] 

    Epoch 1 Batch 3500 loss: 2.36179780960083


    Epoch 1/2:  10%|█         | 3500/34343 [40:45<2:49:20,  3.04it/s]

    Closest words to the center word makes: ['know', 'difficult', 'does', 'must', 'can', 'should', 'want', 'need', 'you', 'might']


    Epoch 1/2:  10%|█         | 3599/34343 [42:09<21:06:08,  2.47s/it]

    Epoch 1 Batch 3600 loss: 2.352299690246582


    Epoch 1/2:  10%|█         | 3600/34343 [42:10<16:24:09,  1.92s/it]

    Closest words to the center word with: ['inducted', 'births', 'isbn', 'pmid', 'gwh', 'had', 'will', 'into', 'explain', 'than']


    Epoch 1/2:  11%|█         | 3699/34343 [43:19<3:12:54,  2.65it/s] 

    Epoch 1 Batch 3700 loss: 2.3344993591308594


    Epoch 1/2:  11%|█         | 3700/34343 [43:20<3:48:33,  2.23it/s]

    Closest words to the center word frankie: ['births', 'laureate', 'inducted', 'isbn', 'gwh', 'pmid', 'frac', 'sep', 'pp', 'oct']


    Epoch 1/2:  11%|█         | 3799/34343 [44:28<1:51:11,  4.58it/s] 

    Epoch 1 Batch 3800 loss: 2.371368408203125


    Epoch 1/2:  11%|█         | 3800/34343 [44:28<2:48:44,  3.02it/s]

    Closest words to the center word eight: ['births', 'laureate', 'isbn', 'inducted', 'gwh', 'pmid', 'sep', 'nine', 'pp', 'seven']


    Epoch 1/2:  11%|█▏        | 3899/34343 [45:26<1:47:46,  4.71it/s] 

    Epoch 1 Batch 3900 loss: 2.3470311164855957


    Epoch 1/2:  11%|█▏        | 3900/34343 [45:26<2:48:37,  3.01it/s]

    Closest words to the center word ratification: ['inducted', 'births', 'pmid', 'midst', 'gregorian', 'laureate', 'united', 'republic', 'february', 'gwh']


    Epoch 1/2:  12%|█▏        | 3999/34343 [46:48<11:21:33,  1.35s/it]

    Epoch 1 Batch 4000 loss: 2.332225799560547


    Epoch 1/2:  12%|█▏        | 4000/34343 [46:49<9:31:30,  1.13s/it] 

    Closest words to the center word ends: ['inducted', 'pmid', 'births', 'years', 'gwh', 'isbn', 'june', 'february', 'ft', 'sep']


    Epoch 1/2:  12%|█▏        | 4099/34343 [47:58<2:22:11,  3.54it/s] 

    Epoch 1 Batch 4100 loss: 2.3461973667144775


    Epoch 1/2:  12%|█▏        | 4100/34343 [47:58<3:15:17,  2.58it/s]

    Closest words to the center word modern: ['spoken', 'languages', 'eastern', 'countries', 'areas', 'european', 'europe', 'nations', 'context', 'western']


    Epoch 1/2:  12%|█▏        | 4199/34343 [49:05<1:46:33,  4.71it/s] 

    Epoch 1 Batch 4200 loss: 2.3501133918762207


    Epoch 1/2:  12%|█▏        | 4200/34343 [49:06<2:45:06,  3.04it/s]

    Closest words to the center word nine: ['births', 'inducted', 'laureate', 'gwh', 'isbn', 'pmid', 'sep', 'nine', 'pp', 'oct']


    Epoch 1/2:  13%|█▎        | 4299/34343 [50:14<1:45:16,  4.76it/s] 

    Epoch 1 Batch 4300 loss: 2.3361902236938477


    Epoch 1/2:  13%|█▎        | 4300/34343 [50:15<2:46:05,  3.01it/s]

    Closest words to the center word for: ['for', 'frac', 'than', 'if', 'own', 'likely', 'there', 'during', 'ago', 'term']


    Epoch 1/2:  13%|█▎        | 4399/34343 [51:39<6:07:27,  1.36it/s] 

    Epoch 1 Batch 4400 loss: 2.3040339946746826


    Epoch 1/2:  13%|█▎        | 4400/34343 [51:40<5:52:04,  1.42it/s]

    Closest words to the center word three: ['births', 'inducted', 'gwh', 'isbn', 'pmid', 'sep', 'unpaved', 'laureate', 'sq', 'est']


    Epoch 1/2:  13%|█▎        | 4499/34343 [52:49<1:50:00,  4.52it/s] 

    Epoch 1 Batch 4500 loss: 2.3336434364318848


    Epoch 1/2:  13%|█▎        | 4500/34343 [52:49<2:52:41,  2.88it/s]

    Closest words to the center word shutout: ['gwh', 'births', 'inducted', 'laureate', 'isbn', 'pmid', 'sep', 'unpaved', 'pp', 'oct']


    Epoch 1/2:  13%|█▎        | 4599/34343 [53:58<1:47:07,  4.63it/s] 

    Epoch 1 Batch 4600 loss: 2.305166482925415


    Epoch 1/2:  13%|█▎        | 4600/34343 [53:59<2:56:24,  2.81it/s]

    Closest words to the center word episode: ['inducted', 'gregorian', 'pmid', 'years', 'gwh', 'sep', 'january', 'february', 'june', 'laureate']


    Epoch 1/2:  14%|█▎        | 4699/34343 [55:22<39:56:33,  4.85s/it]

    Epoch 1 Batch 4700 loss: 2.291905403137207


    Epoch 1/2:  14%|█▎        | 4700/34343 [55:23<29:29:32,  3.58s/it]

    Closest words to the center word have: ['have', 'has', 'had', 'be', 'are', 'know', 'argue', 'been', 'were', 'want']


    Epoch 1/2:  14%|█▍        | 4799/34343 [56:31<2:32:50,  3.22it/s] 

    Epoch 1 Batch 4800 loss: 2.3587260246276855


    Epoch 1/2:  14%|█▍        | 4800/34343 [56:32<3:19:07,  2.47it/s]

    Closest words to the center word disk: ['mathbf', 'frac', 'insubstantial', 'x', 'mbox', 'cdot', 'import', 'function', 'disk', 'file']


    Epoch 1/2:  14%|█▍        | 4899/34343 [57:39<1:52:51,  4.35it/s] 

    Epoch 1 Batch 4900 loss: 2.3159313201904297


    Epoch 1/2:  14%|█▍        | 4900/34343 [57:39<2:52:15,  2.85it/s]

    Closest words to the center word years: ['years', 'gregorian', 'leap', 'million', 'ago', 'year', 'days', 'hours', 'months', 'minutes']


    Epoch 1/2:  15%|█▍        | 4999/34343 [58:46<1:43:51,  4.71it/s] 

    Epoch 1 Batch 5000 loss: 2.2998197078704834


    Epoch 1/2:  15%|█▍        | 5000/34343 [58:47<2:45:45,  2.95it/s]

    Closest words to the center word the: ['the', 'states', 'this', 'its', 'kingdom', 'united', 'war', 'th', 'nations', 'a']


    Epoch 1/2:  15%|█▍        | 5099/34343 [1:00:10<8:10:56,  1.01s/it] 

    Epoch 1 Batch 5100 loss: 2.3164725303649902


    Epoch 1/2:  15%|█▍        | 5100/34343 [1:00:11<7:21:15,  1.10it/s]

    Closest words to the center word that: ['that', 'not', 'we', 'know', 'what', 'necessarily', 'you', 'why', 'if', 'never']


    Epoch 1/2:  15%|█▌        | 5199/34343 [1:01:20<2:16:53,  3.55it/s] 

    Epoch 1 Batch 5200 loss: 2.3047921657562256


    Epoch 1/2:  15%|█▌        | 5200/34343 [1:01:21<3:05:59,  2.61it/s]

    Closest words to the center word theories: ['argue', 'materials', 'practices', 'properties', 'topics', 'aspects', 'particles', 'beings', 'knowledge', 'types']


    Epoch 1/2:  15%|█▌        | 5299/34343 [1:02:30<1:45:58,  4.57it/s] 

    Epoch 1 Batch 5300 loss: 2.3116066455841064


    Epoch 1/2:  15%|█▌        | 5300/34343 [1:02:30<2:45:16,  2.93it/s]

    Closest words to the center word book: ['book', 'wife', 'pmid', 'son', 'laureate', 'gwh', 'frac', 'inducted', 'father', 'feb']


    Epoch 1/2:  16%|█▌        | 5399/34343 [1:03:39<1:43:14,  4.67it/s] 

    Epoch 1 Batch 5400 loss: 2.2891645431518555


    Epoch 1/2:  16%|█▌        | 5400/34343 [1:03:56<40:44:25,  5.07s/it]

    Closest words to the center word owned: ['gwh', 'pmid', 'sep', 'elected', 'united', 'asia', 'east', 'union', 'afc', 'territories']


    Epoch 1/2:  16%|█▌        | 5499/34343 [1:05:04<6:11:32,  1.29it/s] 

    Epoch 1 Batch 5500 loss: 2.292771577835083


    Epoch 1/2:  16%|█▌        | 5500/34343 [1:05:05<5:45:38,  1.39it/s]

    Closest words to the center word real: ['frac', 'mathbf', 'x', 'finite', 'cdot', 'phi', 'sum', 'y', 'mathrm', 'rangle']


    Epoch 1/2:  16%|█▋        | 5599/34343 [1:06:13<2:01:59,  3.93it/s] 

    Epoch 1 Batch 5600 loss: 2.282719135284424


    Epoch 1/2:  16%|█▋        | 5600/34343 [1:06:13<2:51:15,  2.80it/s]

    Closest words to the center word cgi: ['mathbf', 'frac', 'mbox', 'cdot', 'algebra', 'http', 'laureate', 'www', 'linear', 'programming']


    Epoch 1/2:  17%|█▋        | 5699/34343 [1:07:23<1:43:02,  4.63it/s] 

    Epoch 1 Batch 5700 loss: 2.2954509258270264


    Epoch 1/2:  17%|█▋        | 5700/34343 [1:07:24<2:52:42,  2.76it/s]

    Closest words to the center word blues: ['laureate', 'pmid', 'actress', 'actor', 'gwh', 'inducted', 'births', 'comedian', 'physicist', 'musician']


    Epoch 1/2:  17%|█▋        | 5799/34343 [1:08:47<38:01:39,  4.80s/it]

    Epoch 1 Batch 5800 loss: 2.3222241401672363


    Epoch 1/2:  17%|█▋        | 5800/34343 [1:08:48<28:04:33,  3.54s/it]

    Closest words to the center word escape: ['bring', 'return', 'returned', 'throne', 'decided', 'want', 'keep', 'able', 'him', 'refused']


    Epoch 1/2:  17%|█▋        | 5899/34343 [1:09:56<4:51:39,  1.63it/s] 

    Epoch 1 Batch 5900 loss: 2.214003324508667


    Epoch 1/2:  17%|█▋        | 5900/34343 [1:09:57<4:51:12,  1.63it/s]

    Closest words to the center word ions: ['mathbf', 'frac', 'molecules', 'acids', 'carbon', 'finite', 'algebra', 'dioxide', 'properties', 'linear']


    Epoch 1/2:  17%|█▋        | 5999/34343 [1:10:53<1:50:19,  4.28it/s] 

    Epoch 1 Batch 6000 loss: 2.3203701972961426


    Epoch 1/2:  17%|█▋        | 6000/34343 [1:10:54<2:45:13,  2.86it/s]

    Closest words to the center word water: ['est', 'km', 'water', 'carbon', 'dioxide', 'capita', 'energy', 'molecules', 'gwh', 'temperature']


    Epoch 1/2:  18%|█▊        | 6099/34343 [1:12:02<1:40:58,  4.66it/s] 

    Epoch 1 Batch 6100 loss: 2.3139994144439697


    Epoch 1/2:  18%|█▊        | 6100/34343 [1:12:03<2:42:57,  2.89it/s]

    Closest words to the center word lm: ['gwh', 'pmid', 'frac', 'mathbf', 'grt', 'inducted', 'cdot', 'mbox', 'cdots', 'unpaved']


    Epoch 1/2:  18%|█▊        | 6199/34343 [1:13:22<28:33:27,  3.65s/it]

    Epoch 1 Batch 6200 loss: 2.263488531112671


    Epoch 1/2:  18%|█▊        | 6200/34343 [1:13:23<21:26:26,  2.74s/it]

    Closest words to the center word town: ['nfc', 'gregorian', 'city', 'town', 'wales', 'south', 'located', 'york', 'county', 'afc']


    Epoch 1/2:  18%|█▊        | 6299/34343 [1:14:32<3:08:11,  2.48it/s] 

    Epoch 1 Batch 6300 loss: 2.2832627296447754


    Epoch 1/2:  18%|█▊        | 6300/34343 [1:14:32<4:03:19,  1.92it/s]

    Closest words to the center word west: ['nfc', 'east', 'south', 'west', 'north', 'afc', 'ocean', 'coast', 'southeast', 'atlantic']


    Epoch 1/2:  19%|█▊        | 6399/34343 [1:15:37<1:43:55,  4.48it/s] 

    Epoch 1 Batch 6400 loss: 2.263874053955078


    Epoch 1/2:  19%|█▊        | 6400/34343 [1:15:38<2:34:04,  3.02it/s]

    Closest words to the center word his: ['his', 'her', 'him', 'brother', 'wife', 'my', 'father', 'she', 'son', 'shortly']


    Epoch 1/2:  19%|█▉        | 6499/34343 [1:16:45<1:37:36,  4.75it/s] 

    Epoch 1 Batch 6500 loss: 2.2927496433258057


    Epoch 1/2:  19%|█▉        | 6500/34343 [1:16:45<2:33:43,  3.02it/s]

    Closest words to the center word seven: ['gwh', 'pmid', 'births', 'inducted', 'kwh', 'sep', 'isbn', 'feb', 'laureate', 'jul']


    Epoch 1/2:  19%|█▉        | 6599/34343 [1:18:10<14:51:21,  1.93s/it]

    Epoch 1 Batch 6600 loss: 2.261687994003296


    Epoch 1/2:  19%|█▉        | 6600/34343 [1:18:11<11:51:08,  1.54s/it]

    Closest words to the center word renamed: ['pmid', 'inducted', 'gwh', 'sep', 'afc', 'nfc', 'nfl', 'gregorian', 'feb', 'jul']


    Epoch 1/2:  20%|█▉        | 6699/34343 [1:19:21<2:23:05,  3.22it/s] 

    Epoch 1 Batch 6700 loss: 2.2591235637664795


    Epoch 1/2:  20%|█▉        | 6700/34343 [1:19:22<3:03:21,  2.51it/s]

    Closest words to the center word greek: ['roman', 'greek', 'orthodox', 'ancient', 'catholic', 'mythology', 'century', 'spoken', 'indo', 'eastern']


    Epoch 1/2:  20%|█▉        | 6799/34343 [1:20:30<1:39:12,  4.63it/s] 

    Epoch 1 Batch 6800 loss: 2.2623863220214844


    Epoch 1/2:  20%|█▉        | 6800/34343 [1:20:30<2:33:05,  3.00it/s]

    Closest words to the center word than: ['than', 'much', 'less', 'est', 'expensive', 'efficient', 'gwh', 'faster', 'km', 'rate']


    Epoch 1/2:  20%|██        | 6899/34343 [1:21:39<1:37:05,  4.71it/s] 

    Epoch 1 Batch 6900 loss: 2.270664691925049


    Epoch 1/2:  20%|██        | 6900/34343 [1:21:40<2:32:36,  3.00it/s]

    Closest words to the center word easier: ['easier', 'difficult', 'enough', 'prove', 'anything', 'sure', 'necessary', 'understand', 'impossible', 'interact']


    Epoch 1/2:  20%|██        | 6999/34343 [1:23:04<7:41:38,  1.01s/it] 

    Epoch 1 Batch 7000 loss: 2.2632012367248535


    Epoch 1/2:  20%|██        | 7000/34343 [1:23:05<6:48:42,  1.12it/s]

    Closest words to the center word flying: ['gwh', 'pmid', 'cdot', 'frac', 'inducted', 'cdots', 'jul', 'mbox', 'grt', 'unpaved']


    Epoch 1/2:  21%|██        | 7099/34343 [1:24:13<1:44:05,  4.36it/s] 

    Epoch 1 Batch 7100 loss: 2.292957305908203


    Epoch 1/2:  21%|██        | 7100/34343 [1:24:14<2:33:31,  2.96it/s]

    Closest words to the center word solemnly: ['distinguish', 'necessarily', 'know', 'prove', 'understood', 'perceive', 'verify', 'beings', 'believe', 'understand']


    Epoch 1/2:  21%|██        | 7199/34343 [1:25:21<1:36:40,  4.68it/s] 

    Epoch 1 Batch 7200 loss: 2.2485849857330322


    Epoch 1/2:  21%|██        | 7200/34343 [1:25:22<2:32:12,  2.97it/s]

    Closest words to the center word discovered: ['inducted', 'gregorian', 'published', 'discovered', 'pmid', 'century', 'superseded', 'believed', 'leap', 'buried']


    Epoch 1/2:  21%|██▏       | 7299/34343 [1:26:31<1:39:22,  4.54it/s] 

    Epoch 1 Batch 7300 loss: 2.3111839294433594


    Epoch 1/2:  21%|██▏       | 7300/34343 [1:26:47<37:32:27,  5.00s/it]

    Closest words to the center word his: ['his', 'her', 'him', 'brother', 'my', 'wife', 'eldest', 'jesus', 'she', 'father']


    Epoch 1/2:  22%|██▏       | 7399/34343 [1:27:57<2:35:28,  2.89it/s] 

    Epoch 1 Batch 7400 loss: 2.267235279083252


    Epoch 1/2:  22%|██▏       | 7400/34343 [1:27:57<3:10:57,  2.35it/s]

    Closest words to the center word principia: ['laureate', 'cdots', 'gwh', 'frac', 'cdot', 'mathbf', 'isbn', 'mbox', 'pmid', 'mathrm']


    Epoch 1/2:  22%|██▏       | 7499/34343 [1:29:06<1:39:41,  4.49it/s] 

    Epoch 1 Batch 7500 loss: 2.2627291679382324


    Epoch 1/2:  22%|██▏       | 7500/34343 [1:29:06<2:30:43,  2.97it/s]

    Closest words to the center word object: ['subset', 'vector', 'mathbf', 'finite', 'algebra', 'euclidean', 'integer', 'topological', 'function', 'boolean']


    Epoch 1/2:  22%|██▏       | 7599/34343 [1:30:14<1:34:52,  4.70it/s] 

    Epoch 1 Batch 7600 loss: 2.241898536682129


    Epoch 1/2:  22%|██▏       | 7600/34343 [1:30:15<2:27:03,  3.03it/s]

    Closest words to the center word aesthetic: ['cognitive', 'processes', 'topics', 'disorders', 'genetic', 'behaviors', 'computational', 'cognition', 'ethical', 'aspects']


    Epoch 1/2:  22%|██▏       | 7699/34343 [1:31:41<10:01:32,  1.35s/it]

    Epoch 1 Batch 7700 loss: 2.234434127807617


    Epoch 1/2:  22%|██▏       | 7700/34343 [1:31:41<8:23:50,  1.13s/it] 

    Closest words to the center word battleship: ['gwh', 'pmid', 'nfc', 'afc', 'gregorian', 'capita', 'median', 'cyg', 'championship', 'grt']


    Epoch 1/2:  23%|██▎       | 7799/34343 [1:32:52<2:18:28,  3.19it/s] 

    Epoch 1 Batch 7800 loss: 2.2651569843292236


    Epoch 1/2:  23%|██▎       | 7800/34343 [1:32:53<2:57:01,  2.50it/s]

    Closest words to the center word george: ['laureate', 'politician', 'physicist', 'cricketer', 'ois', 'footballer', 'jr', 'actress', 'pmid', 'earl']


    Epoch 1/2:  23%|██▎       | 7899/34343 [1:34:01<1:37:20,  4.53it/s] 

    Epoch 1 Batch 7900 loss: 2.230804443359375


    Epoch 1/2:  23%|██▎       | 7900/34343 [1:34:01<2:31:54,  2.90it/s]

    Closest words to the center word occasionally: ['understood', 'difficult', 'interpreted', 'easier', 'interact', 'willing', 'treat', 'understand', 'verify', 'durable']


    Epoch 1/2:  23%|██▎       | 7999/34343 [1:35:12<1:36:30,  4.55it/s] 

    Epoch 1 Batch 8000 loss: 2.285022258758545


    Epoch 1/2:  23%|██▎       | 8000/34343 [1:35:12<2:30:09,  2.92it/s]

    Closest words to the center word english: ['english', 'laureate', 'nobel', 'physicist', 'prize', 'languages', 'language', 'spoken', 'footballer', 'french']


    Epoch 1/2:  24%|██▎       | 8099/34343 [1:36:37<7:26:38,  1.02s/it] 

    Epoch 1 Batch 8100 loss: 2.292471408843994


    Epoch 1/2:  24%|██▎       | 8100/34343 [1:36:38<6:28:18,  1.13it/s]

    Closest words to the center word western: ['eastern', 'asia', 'caribbean', 'western', 'nfc', 'southeast', 'populous', 'bordering', 'south', 'africa']


    Epoch 1/2:  24%|██▍       | 8199/34343 [1:37:49<2:03:11,  3.54it/s] 

    Epoch 1 Batch 8200 loss: 2.2767786979675293


    Epoch 1/2:  24%|██▍       | 8200/34343 [1:37:49<2:48:22,  2.59it/s]

    Closest words to the center word united: ['united', 'federated', 'states', 'nfc', 'senate', 'republic', 'commonwealth', 'canada', 'nations', 'micronesia']


    Epoch 1/2:  24%|██▍       | 8299/34343 [1:39:00<1:33:34,  4.64it/s] 

    Epoch 1 Batch 8300 loss: 2.2504992485046387


    Epoch 1/2:  24%|██▍       | 8300/34343 [1:39:00<2:27:19,  2.95it/s]

    Closest words to the center word commonly: ['commonly', 'widely', 'referred', 'spoken', 'regarded', 'used', 'understood', 'compounds', 'sometimes', 'often']


    Epoch 1/2:  24%|██▍       | 8399/34343 [1:40:10<1:32:05,  4.69it/s] 

    Epoch 1 Batch 8400 loss: 2.262690544128418


    Epoch 1/2:  24%|██▍       | 8400/34343 [1:40:27<37:48:40,  5.25s/it]

    Closest words to the center word series: ['series', 'game', 'television', 'animated', 'video', 'episode', 'fantasy', 'games', 'cdots', 'album']


    Epoch 1/2:  25%|██▍       | 8499/34343 [1:41:37<5:26:37,  1.32it/s] 

    Epoch 1 Batch 8500 loss: 2.2831146717071533


    Epoch 1/2:  25%|██▍       | 8500/34343 [1:41:37<5:09:21,  1.39it/s]

    Closest words to the center word network: ['intelsat', 'gwh', 'internet', 'software', 'directory', 'isps', 'gnu', 'computer', 'protocol', 'server']


    Epoch 1/2:  25%|██▌       | 8599/34343 [1:42:46<1:44:06,  4.12it/s] 

    Epoch 1 Batch 8600 loss: 2.2123119831085205


    Epoch 1/2:  25%|██▌       | 8600/34343 [1:42:47<2:34:09,  2.78it/s]

    Closest words to the center word those: ['those', 'denominations', 'themselves', 'individuals', 'kinds', 'christians', 'citizens', 'muslims', 'births', 'speakers']


    Epoch 1/2:  25%|██▌       | 8699/34343 [1:44:00<1:31:31,  4.67it/s] 

    Epoch 1 Batch 8700 loss: 2.2346959114074707


    Epoch 1/2:  25%|██▌       | 8700/34343 [1:44:00<2:25:05,  2.95it/s]

    Closest words to the center word catalogue: ['gwh', 'pmid', 'cyg', 'twh', 'laureate', 'jul', 'jun', 'oct', 'kwh', 'runways']


    Epoch 1/2:  26%|██▌       | 8799/34343 [1:45:25<37:38:29,  5.30s/it]

    Epoch 1 Batch 8800 loss: 2.221660852432251


    Epoch 1/2:  26%|██▌       | 8800/34343 [1:45:26<27:53:43,  3.93s/it]

    Closest words to the center word or: ['than', 'or', 'molecules', 'frac', 'amino', 'mathbf', 'membrane', 'atoms', 'dioxide', 'lossy']


    Epoch 1/2:  26%|██▌       | 8899/34343 [1:46:35<3:24:55,  2.07it/s] 

    Epoch 1 Batch 8900 loss: 2.273404359817505


    Epoch 1/2:  26%|██▌       | 8900/34343 [1:46:36<3:53:33,  1.82it/s]

    Closest words to the center word publish: ['publish', 'parents', 'convince', 'respond', 'devote', 'compelled', 'desire', 'understand', 'him', 'pursue']


    Epoch 1/2:  26%|██▌       | 8999/34343 [1:47:34<1:34:00,  4.49it/s] 

    Epoch 1 Batch 9000 loss: 2.2859272956848145


    Epoch 1/2:  26%|██▌       | 9000/34343 [1:47:35<2:24:46,  2.92it/s]

    Closest words to the center word born: ['laureate', 'born', 'actress', 'actor', 'inducted', 'pmid', 'prize', 'nobel', 'births', 'cricketer']


    Epoch 1/2:  26%|██▋       | 9099/34343 [1:48:34<1:31:28,  4.60it/s] 

    Epoch 1 Batch 9100 loss: 2.2506296634674072


    Epoch 1/2:  26%|██▋       | 9100/34343 [1:48:34<2:22:49,  2.95it/s]

    Closest words to the center word databases: ['lossy', 'databases', 'cognitive', 'boolean', 'computational', 'dynamical', 'applications', 'algebra', 'processes', 'spectroscopy']


    Epoch 1/2:  27%|██▋       | 9199/34343 [1:49:56<13:42:11,  1.96s/it]

    Epoch 1 Batch 9200 loss: 2.211230516433716


    Epoch 1/2:  27%|██▋       | 9200/34343 [1:49:57<10:55:05,  1.56s/it]

    Closest words to the center word references: ['www', 'icrm', 'links', 'http', 'isbn', 'ifrcs', 'org', 'edu', 'encyclopedia', 'laureate']


    Epoch 1/2:  27%|██▋       | 9299/34343 [1:50:57<2:17:39,  3.03it/s] 

    Epoch 1 Batch 9300 loss: 2.2425570487976074


    Epoch 1/2:  27%|██▋       | 9300/34343 [1:50:57<2:58:55,  2.33it/s]

    Closest words to the center word root: ['mathbf', 'frac', 'infty', 'cdot', 'rangle', 'vector', 'mathrm', 'langle', 'inverse', 'euclidean']


    Epoch 1/2:  27%|██▋       | 9399/34343 [1:51:59<1:30:32,  4.59it/s] 

    Epoch 1 Batch 9400 loss: 2.238952398300171


    Epoch 1/2:  27%|██▋       | 9400/34343 [1:52:00<2:23:05,  2.91it/s]

    Closest words to the center word of: ['of', 'est', 'expectancy', 'populous', 'ottoman', 'icrm', 'catholic', 'judicial', 'same', 'holy']


    Epoch 1/2:  28%|██▊       | 9499/34343 [1:53:02<1:28:45,  4.67it/s] 

    Epoch 1 Batch 9500 loss: 2.2426815032958984


    Epoch 1/2:  28%|██▊       | 9500/34343 [1:53:03<2:21:21,  2.93it/s]

    Closest words to the center word asw: ['gwh', 'runways', 'laureate', 'pmid', 'cdot', 'cyg', 'jul', 'twh', 'icrm', 'grt']


    Epoch 1/2:  28%|██▊       | 9599/34343 [1:54:22<9:16:01,  1.35s/it] 

    Epoch 1 Batch 9600 loss: 2.2299399375915527


    Epoch 1/2:  28%|██▊       | 9600/34343 [1:54:22<7:52:19,  1.15s/it]

    Closest words to the center word etherboot: ['gwh', 'cyg', 'pmid', 'cdot', 'twh', 'runways', 'jun', 'kwh', 'grt', 'jul']


    Epoch 1/2:  28%|██▊       | 9699/34343 [1:55:19<1:36:25,  4.26it/s] 

    Epoch 1 Batch 9700 loss: 2.273279905319214


    Epoch 1/2:  28%|██▊       | 9700/34343 [1:55:19<2:25:33,  2.82it/s]

    Closest words to the center word dvd: ['windows', 'cdot', 'pc', 'gwh', 'floppy', 'dvd', 'mbox', 'video', 'os', 'mac']


    Epoch 1/2:  29%|██▊       | 9799/34343 [1:56:14<1:26:32,  4.73it/s] 

    Epoch 1 Batch 9800 loss: 2.206517219543457


    Epoch 1/2:  29%|██▊       | 9800/34343 [1:56:14<2:17:25,  2.98it/s]

    Closest words to the center word behaviour: ['cognitive', 'beings', 'processes', 'behaviors', 'perception', 'symptoms', 'subjective', 'phenomena', 'belief', 'ideas']


    Epoch 1/2:  29%|██▉       | 9899/34343 [1:57:03<1:25:38,  4.76it/s] 

    Epoch 1 Batch 9900 loss: 2.2395756244659424


    Epoch 1/2:  29%|██▉       | 9900/34343 [1:57:04<2:15:31,  3.01it/s]

    Closest words to the center word area: ['km', 'capita', 'kilometers', 'area', 'coastline', 'sq', 'runways', 'meters', 'metropolitan', 'nfc']


    Epoch 1/2:  29%|██▉       | 9999/34343 [1:58:10<2:44:55,  2.46it/s] 

    Epoch 1 Batch 10000 loss: 2.2685863971710205


    Epoch 1/2:  29%|██▉       | 10000/34343 [1:58:11<3:13:13,  2.10it/s]

    Closest words to the center word crime: ['sexual', 'communism', 'health', 'political', 'courts', 'elections', 'jurisdiction', 'economic', 'proponent', 'institutions']


    Epoch 1/2:  29%|██▉       | 10099/34343 [1:59:15<1:30:44,  4.45it/s] 

    Epoch 1 Batch 10100 loss: 2.2157480716705322


    Epoch 1/2:  29%|██▉       | 10100/34343 [1:59:15<2:23:56,  2.81it/s]

    Closest words to the center word serves: ['serves', 'referred', 'known', 'regarded', 'became', 'insofar', 'is', 'served', 'allows', 'refers']


    Epoch 1/2:  30%|██▉       | 10199/34343 [2:00:03<1:26:28,  4.65it/s] 

    Epoch 1 Batch 10200 loss: 2.212567090988159


    Epoch 1/2:  30%|██▉       | 10200/34343 [2:00:04<2:18:40,  2.90it/s]

    Closest words to the center word for: ['for', 'manpower', 'gwh', 'cdots', 'cdot', 'lossy', 'gregorian', 'capita', 'best', 'pmid']


    Epoch 1/2:  30%|██▉       | 10299/34343 [2:01:07<7:14:17,  1.08s/it] 

    Epoch 1 Batch 10300 loss: 2.2562217712402344


    Epoch 1/2:  30%|██▉       | 10300/34343 [2:01:08<6:20:08,  1.05it/s]

    Closest words to the center word generation: ['cdots', 'mathbf', 'macintosh', 'frac', 'generation', 'cdot', 'median', 'mac', 'os', 'infty']


    Epoch 1/2:  30%|███       | 10399/34343 [2:02:11<2:01:47,  3.28it/s] 

    Epoch 1 Batch 10400 loss: 2.2322494983673096


    Epoch 1/2:  30%|███       | 10400/34343 [2:02:12<2:42:17,  2.46it/s]

    Closest words to the center word density: ['density', 'capita', 'runways', 'gwh', 'median', 'km', 'meters', 'gdp', 'unpaved', 'est']


    Epoch 1/2:  31%|███       | 10499/34343 [2:03:19<1:41:39,  3.91it/s] 

    Epoch 1 Batch 10500 loss: 2.2334342002868652


    Epoch 1/2:  31%|███       | 10500/34343 [2:03:19<2:27:20,  2.70it/s]

    Closest words to the center word president: ['president', 'minister', 'elected', 'appointed', 'governor', 'secretary', 'appoints', 'chairman', 'presidential', 'vice']


    Epoch 1/2:  31%|███       | 10599/34343 [2:04:07<1:23:37,  4.73it/s] 

    Epoch 1 Batch 10600 loss: 2.2524309158325195


    Epoch 1/2:  31%|███       | 10600/34343 [2:04:08<2:13:14,  2.97it/s]

    Closest words to the center word marie: ['laureate', 'footballer', 'ois', 'actress', 'cricketer', 'fran', 'pngimage', 'pmid', 'actor', 'chemist']


    Epoch 1/2:  31%|███       | 10699/34343 [2:05:01<5:32:27,  1.19it/s] 

    Epoch 1 Batch 10700 loss: 2.226073741912842


    Epoch 1/2:  31%|███       | 10700/34343 [2:05:02<5:08:25,  1.28it/s]

    Closest words to the center word slightly: ['slightly', 'inches', 'grt', 'meters', 'than', 'median', 'capita', 'temperatures', 'unpaved', 'faster']


    Epoch 1/2:  31%|███▏      | 10799/34343 [2:06:02<1:50:07,  3.56it/s] 

    Epoch 1 Batch 10800 loss: 2.213536024093628


    Epoch 1/2:  31%|███▏      | 10800/34343 [2:06:02<2:24:40,  2.71it/s]

    Closest words to the center word who: ['who', 'whom', 'married', 'younger', 'births', 'actors', 'killed', 'singers', 'son', 'accused']


    Epoch 1/2:  32%|███▏      | 10899/34343 [2:07:03<1:24:56,  4.60it/s] 

    Epoch 1 Batch 10900 loss: 2.2501258850097656


    Epoch 1/2:  32%|███▏      | 10900/34343 [2:07:03<2:09:57,  3.01it/s]

    Closest words to the center word calendar: ['gregorian', 'calendar', 'leap', 'day', 'lunisolar', 'year', 'month', 'bce', 'observances', 'nfc']


    Epoch 1/2:  32%|███▏      | 10999/34343 [2:08:06<1:23:34,  4.66it/s] 

    Epoch 1 Batch 11000 loss: 2.234565019607544


    Epoch 1/2:  32%|███▏      | 11000/34343 [2:08:07<2:12:19,  2.94it/s]

    Closest words to the center word communications: ['ifrcs', 'intelsat', 'communications', 'demographics', 'telephones', 'opcw', 'iom', 'ifad', 'transportation', 'isps']


    Epoch 1/2:  32%|███▏      | 11099/34343 [2:09:00<3:11:12,  2.03it/s] 

    Epoch 1 Batch 11100 loss: 2.1692628860473633


    Epoch 1/2:  32%|███▏      | 11100/34343 [2:09:01<3:32:05,  1.83it/s]

    Closest words to the center word names: ['languages', 'dialects', 'names', 'words', 'alphabet', 'cdots', 'verbs', 'nouns', 'phrases', 'speakers']


    Epoch 1/2:  33%|███▎      | 11199/34343 [2:09:45<1:27:29,  4.41it/s] 

    Epoch 1 Batch 11200 loss: 2.212488889694214


    Epoch 1/2:  33%|███▎      | 11200/34343 [2:09:46<2:17:52,  2.80it/s]

    Closest words to the center word denmark: ['denmark', 'republic', 'slovenia', 'serbia', 'austria', 'observances', 'poland', 'hungary', 'lithuania', 'montenegro']


    Epoch 1/2:  33%|███▎      | 11299/34343 [2:10:34<1:21:01,  4.74it/s] 

    Epoch 1 Batch 11300 loss: 2.241395950317383


    Epoch 1/2:  33%|███▎      | 11300/34343 [2:10:34<2:13:54,  2.87it/s]

    Closest words to the center word proved: ['gwh', 'grt', 'proved', 'lasted', 'twh', 'vote', 'cyg', 'election', 'householder', 'speculated']


    Epoch 1/2:  33%|███▎      | 11399/34343 [2:11:38<31:19:09,  4.91s/it]

    Epoch 1 Batch 11400 loss: 2.2049591541290283


    Epoch 1/2:  33%|███▎      | 11400/34343 [2:11:39<23:23:50,  3.67s/it]

    Closest words to the center word food: ['commodities', 'imports', 'textiles', 'ifad', 'exports', 'fuels', 'machinery', 'ifrcs', 'vegetables', 'pollution']


    Epoch 1/2:  33%|███▎      | 11499/34343 [2:12:49<3:07:23,  2.03it/s] 

    Epoch 1 Batch 11500 loss: 2.1915111541748047


    Epoch 1/2:  33%|███▎      | 11500/34343 [2:12:49<3:28:27,  1.83it/s]

    Closest words to the center word opposed: ['opposed', 'referred', 'republics', 'conservative', 'wipo', 'regarded', 'communist', 'socialist', 'conservatives', 'appointed']


    Epoch 1/2:  34%|███▍      | 11599/34343 [2:13:44<1:24:20,  4.49it/s] 

    Epoch 1 Batch 11600 loss: 2.1804370880126953


    Epoch 1/2:  34%|███▍      | 11600/34343 [2:13:45<2:18:10,  2.74it/s]

    Closest words to the center word leadership: ['leadership', 'socialist', 'privy', 'communist', 'democratic', 'liberties', 'overthrow', 'ministers', 'republics', 'party']


    Epoch 1/2:  34%|███▍      | 11699/34343 [2:14:38<1:20:35,  4.68it/s] 

    Epoch 1 Batch 11700 loss: 2.2079873085021973


    Epoch 1/2:  34%|███▍      | 11700/34343 [2:14:38<2:05:47,  3.00it/s]

    Closest words to the center word two: ['gwh', 'cyg', 'twh', 'pmid', 'grt', 'kwh', 'unpaved', 'jun', 'sep', 'runways']


    Epoch 1/2:  34%|███▍      | 11799/34343 [2:15:43<9:35:59,  1.53s/it] 

    Epoch 1 Batch 11800 loss: 2.204357385635376


    Epoch 1/2:  34%|███▍      | 11800/34343 [2:15:44<8:02:17,  1.28s/it]

    Closest words to the center word upgrade: ['gwh', 'kwh', 'unpaved', 'runways', 'grt', 'capita', 'twh', 'isps', 'cyg', 'mhz']


    Epoch 1/2:  35%|███▍      | 11899/34343 [2:16:33<1:30:12,  4.15it/s] 

    Epoch 1 Batch 11900 loss: 2.221238136291504


    Epoch 1/2:  35%|███▍      | 11900/34343 [2:16:33<2:19:56,  2.67it/s]

    Closest words to the center word chesterton: ['cdot', 'cdots', 'rightarrow', 'rangle', 'mathbf', 'otimes', 'frac', 'rang', 'qquad', 'cos']


    Epoch 1/2:  35%|███▍      | 11999/34343 [2:17:19<1:25:11,  4.37it/s] 

    Epoch 1 Batch 12000 loss: 2.1798417568206787


    Epoch 1/2:  35%|███▍      | 12000/34343 [2:17:20<2:14:34,  2.77it/s]

    Closest words to the center word they: ['they', 'we', 'you', 'happen', 'not', 'he', 'decide', 'themselves', 'occur', 'able']


    Epoch 1/2:  35%|███▌      | 12099/34343 [2:18:06<1:21:18,  4.56it/s] 

    Epoch 1 Batch 12100 loss: 2.215872287750244


    Epoch 1/2:  35%|███▌      | 12100/34343 [2:18:06<2:10:05,  2.85it/s]

    Closest words to the center word informal: ['empirical', 'implications', 'ethical', 'grammatical', 'cognitive', 'anarcho', 'rabbinic', 'computational', 'grammar', 'verbs']


    Epoch 1/2:  36%|███▌      | 12199/34343 [2:19:00<2:28:31,  2.48it/s] 

    Epoch 1 Batch 12200 loss: 2.1946401596069336


    Epoch 1/2:  36%|███▌      | 12200/34343 [2:19:01<3:02:58,  2.02it/s]

    Closest words to the center word games: ['games', 'game', 'consoles', 'championship', 'video', 'playstation', 'console', 'football', 'baseball', 'kart']


    Epoch 1/2:  36%|███▌      | 12299/34343 [2:19:44<1:22:36,  4.45it/s] 

    Epoch 1 Batch 12300 loss: 2.2030444145202637


    Epoch 1/2:  36%|███▌      | 12300/34343 [2:19:45<2:12:41,  2.77it/s]

    Closest words to the center word reach: ['reach', 'grt', 'meters', 'nfc', 'kilometers', 'km', 'lose', 'playoffs', 'afc', 'kilometres']


    Epoch 1/2:  36%|███▌      | 12399/34343 [2:20:19<1:21:03,  4.51it/s] 

    Epoch 1 Batch 12400 loss: 2.1845791339874268


    Epoch 1/2:  36%|███▌      | 12400/34343 [2:20:20<2:04:08,  2.95it/s]

    Closest words to the center word mary: ['mary', 'eldest', 'jesus', 'nephew', 'son', 'aragon', 'brother', 'grandson', 'earl', 'duke']


    Epoch 1/2:  36%|███▋      | 12499/34343 [2:21:26<17:57:48,  2.96s/it]

    Epoch 1 Batch 12500 loss: 2.248018980026245


    Epoch 1/2:  36%|███▋      | 12500/34343 [2:21:26<13:45:35,  2.27s/it]

    Closest words to the center word mean: ['mathbf', 'frac', 'infty', 'cdots', 'inverse', 'cdot', 'mathrm', 'rangle', 'mbox', 'mean']


    Epoch 1/2:  37%|███▋      | 12599/34343 [2:22:10<2:21:06,  2.57it/s] 

    Epoch 1 Batch 12600 loss: 2.2393078804016113


    Epoch 1/2:  37%|███▋      | 12600/34343 [2:22:11<2:49:14,  2.14it/s]

    Closest words to the center word method: ['boolean', 'entropy', 'vector', 'mechanics', 'differential', 'quantum', 'dynamical', 'equations', 'coding', 'lossy']


    Epoch 1/2:  37%|███▋      | 12699/34343 [2:22:57<1:19:39,  4.53it/s] 

    Epoch 1 Batch 12700 loss: 2.1995980739593506


    Epoch 1/2:  37%|███▋      | 12700/34343 [2:22:58<2:22:52,  2.52it/s]

    Closest words to the center word subsequent: ['reign', 'subsequent', 'semitism', 'political', 'economic', 'reforms', 'lasted', 'stalin', 'abuses', 'napoleonic']


    Epoch 1/2:  37%|███▋      | 12799/34343 [2:23:49<1:17:18,  4.64it/s] 

    Epoch 1 Batch 12800 loss: 2.1995325088500977


    Epoch 1/2:  37%|███▋      | 12800/34343 [2:23:50<2:00:18,  2.98it/s]

    Closest words to the center word contingent: ['pluriform', 'socio', 'desertification', 'vested', 'judiciary', 'monetary', 'economic', 'bordering', 'judicial', 'coasts']


    Epoch 1/2:  38%|███▊      | 12899/34343 [2:24:42<7:27:26,  1.25s/it] 

    Epoch 1 Batch 12900 loss: 2.247729539871216


    Epoch 1/2:  38%|███▊      | 12900/34343 [2:24:43<6:31:54,  1.10s/it]

    Closest words to the center word british: ['british', 'laureate', 'politician', 'american', 'canadian', 'nobel', 'ifrcs', 'footballer', 'ifad', 'actor']


    Epoch 1/2:  38%|███▊      | 12999/34343 [2:25:31<2:17:46,  2.58it/s] 

    Epoch 1 Batch 13000 loss: 2.226405620574951


    Epoch 1/2:  38%|███▊      | 13000/34343 [2:25:32<2:43:28,  2.18it/s]

    Closest words to the center word battles: ['battles', 'gregorian', 'wars', 'ottoman', 'reign', 'war', 'outbreak', 'napoleonic', 'fought', 'invaded']


    Epoch 1/2:  38%|███▊      | 13099/34343 [2:26:19<1:20:39,  4.39it/s] 

    Epoch 1 Batch 13100 loss: 2.228623628616333


    Epoch 1/2:  38%|███▊      | 13100/34343 [2:26:19<2:01:37,  2.91it/s]

    Closest words to the center word ring: ['mathbf', 'cdots', 'frac', 'rightarrow', 'rangle', 'cdot', 'rang', 'langle', 'infty', 'mathcal']


    Epoch 1/2:  38%|███▊      | 13199/34343 [2:27:01<1:14:37,  4.72it/s] 

    Epoch 1 Batch 13200 loss: 2.221473217010498


    Epoch 1/2:  38%|███▊      | 13200/34343 [2:27:01<1:57:55,  2.99it/s]

    Closest words to the center word jeane: ['laureate', 'cyg', 'gwh', 'comedian', 'actor', 'footballer', 'grt', 'actress', 'inducted', 'pmid']


    Epoch 1/2:  39%|███▊      | 13299/34343 [2:27:58<6:38:36,  1.14s/it] 

    Epoch 1 Batch 13300 loss: 2.1854989528656006


    Epoch 1/2:  39%|███▊      | 13300/34343 [2:27:59<5:47:55,  1.01it/s]

    Closest words to the center word v: ['cdots', 'cdot', 'frac', 'rangle', 'rightarrow', 'mathbf', 'qquad', 'infty', 'rang', 'leq']


    Epoch 1/2:  39%|███▉      | 13399/34343 [2:28:42<1:34:43,  3.69it/s] 

    Epoch 1 Batch 13400 loss: 2.2130494117736816


    Epoch 1/2:  39%|███▉      | 13400/34343 [2:28:43<2:11:23,  2.66it/s]

    Closest words to the center word type: ['vector', 'electromagnetic', 'covalent', 'subset', 'morphism', 'ions', 'type', 'hydrogen', 'lossy', 'euclidean']


    Epoch 1/2:  39%|███▉      | 13499/34343 [2:29:30<1:16:09,  4.56it/s] 

    Epoch 1 Batch 13500 loss: 2.2179203033447266


    Epoch 1/2:  39%|███▉      | 13500/34343 [2:29:30<1:57:49,  2.95it/s]

    Closest words to the center word freemasons: ['wftu', 'wipo', 'unido', 'wmo', 'upu', 'ifrcs', 'ifad', 'wtoo', 'tribes', 'aryans']


    Epoch 1/2:  40%|███▉      | 13599/34343 [2:30:30<1:12:31,  4.77it/s] 

    Epoch 1 Batch 13600 loss: 2.207820177078247


    Epoch 1/2:  40%|███▉      | 13600/34343 [2:30:31<1:51:01,  3.11it/s]

    Closest words to the center word for: ['for', 'manpower', 'capita', 'parity', 'lossy', 'gwh', 'gregorian', 'cdot', 'copyleft', 'newnode']


    Epoch 1/2:  40%|███▉      | 13699/34343 [2:31:16<2:26:15,  2.35it/s] 

    Epoch 1 Batch 13700 loss: 2.229189395904541


    Epoch 1/2:  40%|███▉      | 13700/34343 [2:31:17<2:50:38,  2.02it/s]

    Closest words to the center word ale: ['ifad', 'ifrcs', 'icrm', 'callithrix', 'fricative', 'classis', 'ifc', 'sauce', 'agave', 'aloe']


    Epoch 1/2:  40%|████      | 13799/34343 [2:31:59<1:18:16,  4.37it/s] 

    Epoch 1 Batch 13800 loss: 2.2239222526550293


    Epoch 1/2:  40%|████      | 13800/34343 [2:31:59<2:00:38,  2.84it/s]

    Closest words to the center word stored: ['stored', 'storage', 'cathode', 'disk', 'lossy', 'floppy', 'input', 'vector', 'circuits', 'combustion']


    Epoch 1/2:  40%|████      | 13899/34343 [2:32:36<1:13:06,  4.66it/s] 

    Epoch 1 Batch 13900 loss: 2.2027957439422607


    Epoch 1/2:  40%|████      | 13900/34343 [2:32:37<1:55:49,  2.94it/s]

    Closest words to the center word will: ['will', 'shall', 'would', 'must', 'can', 'should', 'could', 'does', 'might', 'doesn']


    Epoch 1/2:  41%|████      | 13999/34343 [2:33:13<1:10:47,  4.79it/s] 

    Epoch 1 Batch 14000 loss: 2.2237448692321777


    Epoch 1/2:  41%|████      | 14000/34343 [2:33:19<10:25:14,  1.84s/it]

    Closest words to the center word changed: ['changed', 'gregorian', 'sixteenth', 'traced', 'lasted', 'weakened', 'ratified', 'ottoman', 'postponed', 'adopted']


    Epoch 1/2:  41%|████      | 14099/34343 [2:33:56<2:02:07,  2.76it/s] 

    Epoch 1 Batch 14100 loss: 2.222094774246216


    Epoch 1/2:  41%|████      | 14100/34343 [2:33:56<2:29:22,  2.26it/s]

    Closest words to the center word wages: ['manpower', 'wages', 'expenditures', 'income', 'males', 'revenues', 'investment', 'prices', 'mortality', 'rates']


    Epoch 1/2:  41%|████▏     | 14199/34343 [2:34:34<1:13:00,  4.60it/s] 

    Epoch 1 Batch 14200 loss: 2.184338092803955


    Epoch 1/2:  41%|████▏     | 14200/34343 [2:34:35<1:52:56,  2.97it/s]

    Closest words to the center word a: ['a', 'any', 'mathbf', 'every', 'cdot', 'an', 'infty', 'mathrm', 'countable', 'another']


    Epoch 1/2:  42%|████▏     | 14299/34343 [2:35:09<1:09:43,  4.79it/s] 

    Epoch 1 Batch 14300 loss: 2.216555595397949


    Epoch 1/2:  42%|████▏     | 14300/34343 [2:35:09<1:49:01,  3.06it/s]

    Closest words to the center word same: ['same', 'gregorian', 'polynomial', 'frac', 'cardinality', 'mathbf', 'inverse', 'discrete', 'exact', 'focal']


    Epoch 1/2:  42%|████▏     | 14399/34343 [2:35:52<8:07:24,  1.47s/it] 

    Epoch 1 Batch 14400 loss: 2.2166624069213867


    Epoch 1/2:  42%|████▏     | 14400/34343 [2:35:53<6:44:04,  1.22s/it]

    Closest words to the center word user: ['user', 'findable', 'interface', 'gnu', 'server', 'lossy', 'browser', 'browsers', 'dns', 'graphical']


    Epoch 1/2:  42%|████▏     | 14499/34343 [2:36:33<1:19:21,  4.17it/s] 

    Epoch 1 Batch 14500 loss: 2.2113637924194336


    Epoch 1/2:  42%|████▏     | 14500/34343 [2:36:34<2:10:10,  2.54it/s]

    Closest words to the center word fine: ['sauce', 'beverages', 'nickel', 'vegetables', 'liquid', 'hydroxide', 'dairy', 'minerals', 'metal', 'fine']


    Epoch 1/2:  43%|████▎     | 14599/34343 [2:37:09<1:10:03,  4.70it/s] 

    Epoch 1 Batch 14600 loss: 2.209588050842285


    Epoch 1/2:  43%|████▎     | 14600/34343 [2:37:10<1:50:39,  2.97it/s]

    Closest words to the center word for: ['for', 'manpower', 'newnode', 'lossy', 'parity', 'insubstantial', 'capita', 'lossless', 'median', 'copyleft']


    Epoch 1/2:  43%|████▎     | 14699/34343 [2:37:46<1:11:10,  4.60it/s] 

    Epoch 1 Batch 14700 loss: 2.2102251052856445


    Epoch 1/2:  43%|████▎     | 14700/34343 [2:37:46<1:50:45,  2.96it/s]

    Closest words to the center word following: ['gregorian', 'following', 'calendar', 'leap', 'punic', 'lunisolar', 'napoleonic', 'factbook', 'ottoman', 'feast']


    Epoch 1/2:  43%|████▎     | 14799/34343 [2:38:31<3:12:17,  1.69it/s] 

    Epoch 1 Batch 14800 loss: 2.21830677986145


    Epoch 1/2:  43%|████▎     | 14800/34343 [2:38:32<3:19:50,  1.63it/s]

    Closest words to the center word dancing: ['piano', 'jazz', 'solo', 'singers', 'dancing', 'genres', 'musical', 'singing', 'folk', 'violin']


    Epoch 1/2:  43%|████▎     | 14899/34343 [2:39:29<1:22:30,  3.93it/s] 

    Epoch 1 Batch 14900 loss: 2.1905465126037598


    Epoch 1/2:  43%|████▎     | 14900/34343 [2:39:30<2:00:53,  2.68it/s]

    Closest words to the center word gorges: ['km', 'unpaved', 'runways', 'humid', 'irrigated', 'temperate', 'pastures', 'arable', 'coastline', 'elevation']


    Epoch 1/2:  44%|████▎     | 14999/34343 [2:40:11<1:08:21,  4.72it/s] 

    Epoch 1 Batch 15000 loss: 2.2109179496765137


    Epoch 1/2:  44%|████▎     | 15000/34343 [2:40:11<1:51:52,  2.88it/s]

    Closest words to the center word compiling: ['lossy', 'browsers', 'input', 'lossless', 'boolean', 'bilinear', 'formats', 'dynamical', 'homomorphism', 'applications']


    Epoch 1/2:  44%|████▍     | 15099/34343 [2:41:10<22:00:38,  4.12s/it]

    Epoch 1 Batch 15100 loss: 2.195054054260254


    Epoch 1/2:  44%|████▍     | 15100/34343 [2:41:11<16:38:17,  3.11s/it]

    Closest words to the center word amd: ['gwh', 'kwh', 'cyg', 'twh', 'rfc', 'pmid', 'intel', 'isps', 'unpaved', 'ieee']


    Epoch 1/2:  44%|████▍     | 15199/34343 [2:41:56<2:02:27,  2.61it/s] 

    Epoch 1 Batch 15200 loss: 2.172524929046631


    Epoch 1/2:  44%|████▍     | 15200/34343 [2:41:57<2:30:38,  2.12it/s]

    Closest words to the center word al: ['al', 'ibn', 'abu', 'abd', 'agave', 'bin', 'ifrcs', 'unctad', 'wahhab', 'unido']


    Epoch 1/2:  45%|████▍     | 15299/34343 [2:43:09<1:19:00,  4.02it/s] 

    Epoch 1 Batch 15300 loss: 2.2157206535339355


    Epoch 1/2:  45%|████▍     | 15300/34343 [2:43:10<1:54:51,  2.76it/s]

    Closest words to the center word most: ['most', 'more', 'many', 'less', 'largest', 'earliest', 'best', 'populous', 'widely', 'among']


    Epoch 1/2:  45%|████▍     | 15399/34343 [2:44:20<1:07:51,  4.65it/s] 

    Epoch 1 Batch 15400 loss: 2.1843748092651367


    Epoch 1/2:  45%|████▍     | 15400/34343 [2:44:21<1:44:43,  3.01it/s]

    Closest words to the center word estonian: ['laureate', 'footballer', 'finalist', 'ivoire', 'estonian', 'ibrd', 'slovak', 'icrm', 'czech', 'physiologist']


    Epoch 1/2:  45%|████▌     | 15499/34343 [2:45:47<19:14:30,  3.68s/it]

    Epoch 1 Batch 15500 loss: 2.217909336090088


    Epoch 1/2:  45%|████▌     | 15500/34343 [2:45:47<14:24:56,  2.75s/it]

    Closest words to the center word portions: ['portions', 'denominations', 'ghats', 'basins', 'deserts', 'peoples', 'uplands', 'deposits', 'tributaries', 'shores']


    Epoch 1/2:  45%|████▌     | 15599/34343 [2:46:44<1:55:41,  2.70it/s] 

    Epoch 1 Batch 15600 loss: 2.2558064460754395


    Epoch 1/2:  45%|████▌     | 15600/34343 [2:46:45<2:19:36,  2.24it/s]

    Closest words to the center word by: ['by', 'agave', 'ifrcs', 'ifc', 'unctad', 'icrm', 'ifad', 'gwh', 'cyg', 'newly']


    Epoch 1/2:  46%|████▌     | 15699/34343 [2:47:36<1:11:29,  4.35it/s] 

    Epoch 1 Batch 15700 loss: 2.2247748374938965


    Epoch 1/2:  46%|████▌     | 15700/34343 [2:47:37<1:46:19,  2.92it/s]

    Closest words to the center word sade: ['sade', 'anh', 'baptiste', 'ois', 'biography', 'cegep', 'medici', 'johann', 'renoir', 'laureate']


    Epoch 1/2:  46%|████▌     | 15799/34343 [2:48:44<1:05:14,  4.74it/s] 

    Epoch 1 Batch 15800 loss: 2.1729252338409424


    Epoch 1/2:  46%|████▌     | 15800/34343 [2:48:45<1:42:05,  3.03it/s]

    Closest words to the center word emerging: ['ifrcs', 'desertification', 'ifad', 'ilo', 'unctad', 'faire', 'socio', 'oau', 'iom', 'ifc']


    Epoch 1/2:  46%|████▋     | 15899/34343 [2:49:45<9:27:05,  1.84s/it] 

    Epoch 1 Batch 15900 loss: 2.181105613708496


    Epoch 1/2:  46%|████▋     | 15900/34343 [2:49:45<7:33:59,  1.48s/it]

    Closest words to the center word in: ['in', 'throughout', 'pmid', 'nfc', 'kwh', 'gregorian', 'during', 'until', 'gwh', 'cyg']


    Epoch 1/2:  47%|████▋     | 15999/34343 [2:50:37<1:35:13,  3.21it/s] 

    Epoch 1 Batch 16000 loss: 2.2073497772216797


    Epoch 1/2:  47%|████▋     | 16000/34343 [2:50:38<2:01:31,  2.52it/s]

    Closest words to the center word groups: ['groups', 'ethnic', 'denominations', 'religions', 'minorities', 'parties', 'dialects', 'genders', 'minority', 'faiths']


    Epoch 1/2:  47%|████▋     | 16099/34343 [2:51:19<1:09:39,  4.36it/s] 

    Epoch 1 Batch 16100 loss: 2.1873230934143066


    Epoch 1/2:  47%|████▋     | 16100/34343 [2:51:19<1:46:05,  2.87it/s]

    Closest words to the center word india: ['timor', 'tajikistan', 'namibia', 'lanka', 'asia', 'swaziland', 'zambia', 'indonesia', 'india', 'nepal']


    Epoch 1/2:  47%|████▋     | 16199/34343 [2:52:04<1:04:15,  4.71it/s] 

    Epoch 1 Batch 16200 loss: 2.1928420066833496


    Epoch 1/2:  47%|████▋     | 16200/34343 [2:52:05<1:43:11,  2.93it/s]

    Closest words to the center word coaches: ['coach', 'playoffs', 'afc', 'nfc', 'gwh', 'kwh', 'runways', 'finalist', 'basketball', 'footballer']


    Epoch 1/2:  47%|████▋     | 16299/34343 [2:53:12<4:34:32,  1.10it/s] 

    Epoch 1 Batch 16300 loss: 2.2128429412841797


    Epoch 1/2:  47%|████▋     | 16300/34343 [2:53:13<4:09:27,  1.21it/s]

    Closest words to the center word found: ['found', 'mentioned', 'buried', 'agave', 'soluble', 'located', 'explained', 'shown', 'discovered', 'traced']


    Epoch 1/2:  48%|████▊     | 16399/34343 [2:54:06<1:17:20,  3.87it/s] 

    Epoch 1 Batch 16400 loss: 2.2028470039367676


    Epoch 1/2:  48%|████▊     | 16400/34343 [2:54:07<1:49:27,  2.73it/s]

    Closest words to the center word officer: ['officer', 'secretary', 'commander', 'deputy', 'unido', 'minister', 'appoints', 'marshal', 'politician', 'unctad']


    Epoch 1/2:  48%|████▊     | 16499/34343 [2:55:03<1:02:54,  4.73it/s] 

    Epoch 1 Batch 16500 loss: 2.228391408920288


    Epoch 1/2:  48%|████▊     | 16500/34343 [2:55:03<1:40:02,  2.97it/s]

    Closest words to the center word nero: ['tiberius', 'emperor', 'elector', 'grandson', 'throne', 'alexius', 'constantius', 'eldest', 'maximilian', 'aurelius']


    Epoch 1/2:  48%|████▊     | 16599/34343 [2:55:51<1:05:26,  4.52it/s] 

    Epoch 1 Batch 16600 loss: 2.1958260536193848


    Epoch 1/2:  48%|████▊     | 16600/34343 [2:55:51<1:37:40,  3.03it/s]

    Closest words to the center word europe: ['asia', 'populous', 'europe', 'scandinavia', 'nfc', 'bordering', 'annexed', 'subcontinent', 'timor', 'eurasia']


    Epoch 1/2:  49%|████▊     | 16699/34343 [2:56:46<2:28:17,  1.98it/s] 

    Epoch 1 Batch 16700 loss: 2.1679584980010986


    Epoch 1/2:  49%|████▊     | 16700/34343 [2:56:46<2:39:58,  1.84it/s]

    Closest words to the center word the: ['the', 'nfc', 'vernal', 'populous', 'afc', 'solar', 'scrimmage', 'its', 'frac', 'ecliptic']


    Epoch 1/2:  49%|████▉     | 16799/34343 [2:57:45<1:04:42,  4.52it/s] 

    Epoch 1 Batch 16800 loss: 2.1783878803253174


    Epoch 1/2:  49%|████▉     | 16800/34343 [2:57:46<1:40:17,  2.92it/s]

    Closest words to the center word order: ['order', 'differential', 'attempt', 'node', 'alphabetical', 'relation', 'accordance', 'topological', 'predicate', 'isomorphic']


    Epoch 1/2:  49%|████▉     | 16899/34343 [2:58:33<1:01:27,  4.73it/s] 

    Epoch 1 Batch 16900 loss: 2.2228479385375977


    Epoch 1/2:  49%|████▉     | 16900/34343 [2:58:33<1:39:04,  2.93it/s]

    Closest words to the center word included: ['included', 'published', 'playstation', 'appeared', 'featured', 'twh', 'bwv', 'released', 'gregorian', 'cyg']


    Epoch 1/2:  49%|████▉     | 16999/34343 [2:59:30<14:07:21,  2.93s/it]

    Epoch 1 Batch 17000 loss: 2.196394205093384


    Epoch 1/2:  50%|████▉     | 17000/34343 [2:59:30<10:56:10,  2.27s/it]

    Closest words to the center word scottish: ['laureate', 'footballer', 'theologian', 'politician', 'cricketer', 'scottish', 'physiologist', 'statesman', 'novelists', 'chemist']


    Epoch 1/2:  50%|████▉     | 17099/34343 [3:00:18<1:39:05,  2.90it/s] 

    Epoch 1 Batch 17100 loss: 2.187544584274292


    Epoch 1/2:  50%|████▉     | 17100/34343 [3:00:18<2:03:07,  2.33it/s]

    Closest words to the center word pilots: ['grt', 'playoffs', 'missiles', 'pilots', 'aircraft', 'manpower', 'runways', 'ships', 'gwh', 'helicopters']


    Epoch 1/2:  50%|█████     | 17199/34343 [3:01:01<1:01:09,  4.67it/s] 

    Epoch 1 Batch 17200 loss: 2.1696689128875732


    Epoch 1/2:  50%|█████     | 17200/34343 [3:01:01<1:42:26,  2.79it/s]

    Closest words to the center word comstock: ['gwh', 'twh', 'ifrcs', 'ifad', 'icrm', 'ifc', 'cyg', 'jul', 'agave', 'pngimage']


    Epoch 1/2:  50%|█████     | 17299/34343 [3:01:58<1:00:05,  4.73it/s] 

    Epoch 1 Batch 17300 loss: 2.1982107162475586


    Epoch 1/2:  50%|█████     | 17300/34343 [3:01:59<1:34:10,  3.02it/s]

    Closest words to the center word and: ['ifad', 'ifrcs', 'ifc', 'icrm', 'icftu', 'gwh', 'kwh', 'unctad', 'and', 'unido']


    Epoch 1/2:  51%|█████     | 17399/34343 [3:03:02<3:11:06,  1.48it/s] 

    Epoch 1 Batch 17400 loss: 2.195021629333496


    Epoch 1/2:  51%|█████     | 17400/34343 [3:03:03<3:19:05,  1.42it/s]

    Closest words to the center word during: ['during', 'period', 'manpower', 'war', 'outbreak', 'napoleonic', 'triassic', 'cretaceous', 'nfc', 'ottoman']


    Epoch 1/2:  51%|█████     | 17499/34343 [3:03:46<1:07:06,  4.18it/s] 

    Epoch 1 Batch 17500 loss: 2.182107925415039


    Epoch 1/2:  51%|█████     | 17500/34343 [3:03:47<1:39:42,  2.82it/s]

    Closest words to the center word line: ['scrimmage', 'line', 'mjs', 'cdots', 'perpendicular', 'node', 'pngimage', 'cue', 'playoffs', 'ecliptic']


    Epoch 1/2:  51%|█████     | 17599/34343 [3:04:33<59:36,  4.68it/s]   

    Epoch 1 Batch 17600 loss: 2.21285080909729


    Epoch 1/2:  51%|█████     | 17600/34343 [3:04:33<1:32:42,  3.01it/s]

    Closest words to the center word or: ['or', 'ifad', 'nucleic', 'reactive', 'carboxylic', 'ifrcs', 'covalent', 'than', 'carbohydrates', 'ions']


    Epoch 1/2:  52%|█████▏    | 17699/34343 [3:05:15<1:00:43,  4.57it/s] 

    Epoch 1 Batch 17700 loss: 2.1553187370300293


    Epoch 1/2:  52%|█████▏    | 17700/34343 [3:05:30<21:04:49,  4.56s/it]

    Closest words to the center word plateau: ['plateau', 'bordering', 'mountainous', 'humid', 'temperate', 'hilly', 'subtropical', 'southwest', 'basin', 'coastal']


    Epoch 1/2:  52%|█████▏    | 17799/34343 [3:06:38<3:27:30,  1.33it/s] 

    Epoch 1 Batch 17800 loss: 2.158411979675293


    Epoch 1/2:  52%|█████▏    | 17800/34343 [3:06:39<3:15:00,  1.41it/s]

    Closest words to the center word name: ['name', 'alphabet', 'word', 'title', 'derives', 'surname', 'derived', 'testament', 'hebrew', 'names']


    Epoch 1/2:  52%|█████▏    | 17899/34343 [3:07:28<1:07:02,  4.09it/s] 

    Epoch 1 Batch 17900 loss: 2.1678919792175293


    Epoch 1/2:  52%|█████▏    | 17900/34343 [3:07:29<1:40:59,  2.71it/s]

    Closest words to the center word tr: ['cdots', 'rightarrow', 'cdot', 'leq', 'otimes', 'qquad', 'equiv', 'aq', 'mbox', 'rang']


    Epoch 1/2:  52%|█████▏    | 17999/34343 [3:08:10<58:23,  4.66it/s]   

    Epoch 1 Batch 18000 loss: 2.214918375015259


    Epoch 1/2:  52%|█████▏    | 18000/34343 [3:08:11<1:31:00,  2.99it/s]

    Closest words to the center word one: ['gwh', 'cyg', 'twh', 'kwh', 'jul', 'grt', 'pmid', 'unpaved', 'lup', 'pngimage']


    Epoch 1/2:  53%|█████▎    | 18099/34343 [3:09:12<11:17:49,  2.50s/it]

    Epoch 1 Batch 18100 loss: 2.172701597213745


    Epoch 1/2:  53%|█████▎    | 18100/34343 [3:09:12<8:46:37,  1.95s/it] 

    Closest words to the center word khartoum: ['opcw', 'iom', 'rupee', 'swaziland', 'nfc', 'tajikistan', 'barbuda', 'zambia', 'ifc', 'bordering']


    Epoch 1/2:  53%|█████▎    | 18199/34343 [3:09:55<1:38:37,  2.73it/s] 

    Epoch 1 Batch 18200 loss: 2.181840658187866


    Epoch 1/2:  53%|█████▎    | 18200/34343 [3:09:55<2:00:25,  2.23it/s]

    Closest words to the center word mitsubishi: ['gwh', 'kwh', 'unpaved', 'twh', 'runways', 'cyg', 'mjs', 'dwt', 'pngimage', 'telephones']


    Epoch 1/2:  53%|█████▎    | 18299/34343 [3:10:52<1:03:13,  4.23it/s] 

    Epoch 1 Batch 18300 loss: 2.1708874702453613


    Epoch 1/2:  53%|█████▎    | 18300/34343 [3:10:53<1:33:06,  2.87it/s]

    Closest words to the center word apple: ['macintosh', 'apple', 'amiga', 'ibm', 'intel', 'microsoft', 'atari', 'pc', 'mac', 'playstation']


    Epoch 1/2:  54%|█████▎    | 18399/34343 [3:11:50<56:32,  4.70it/s]   

    Epoch 1 Batch 18400 loss: 2.180054187774658


    Epoch 1/2:  54%|█████▎    | 18400/34343 [3:11:51<1:28:37,  3.00it/s]

    Closest words to the center word prior: ['prior', 'grt', 'gwh', 'twh', 'spend', 'manpower', 'devote', 'kwh', 'leap', 'resolve']


    Epoch 1/2:  54%|█████▍    | 18499/34343 [3:12:47<7:25:49,  1.69s/it] 

    Epoch 1 Batch 18500 loss: 2.1845858097076416


    Epoch 1/2:  54%|█████▍    | 18500/34343 [3:12:47<6:00:32,  1.37s/it]

    Closest words to the center word horse: ['horse', 'sox', 'callithrix', 'humid', 'playoffs', 'grt', 'equus', 'playoff', 'tamarin', 'wild']


    Epoch 1/2:  54%|█████▍    | 18599/34343 [3:13:35<1:22:50,  3.17it/s] 

    Epoch 1 Batch 18600 loss: 2.209273099899292


    Epoch 1/2:  54%|█████▍    | 18600/34343 [3:13:36<1:46:07,  2.47it/s]

    Closest words to the center word different: ['different', 'distinct', 'variety', 'countably', 'eukaryotic', 'abelian', 'grammatical', 'various', 'finite', 'multicellular']


    Epoch 1/2:  54%|█████▍    | 18699/34343 [3:14:18<1:03:57,  4.08it/s] 

    Epoch 1 Batch 18700 loss: 2.2018415927886963


    Epoch 1/2:  54%|█████▍    | 18700/34343 [3:14:18<1:32:36,  2.82it/s]

    Closest words to the center word catastrophic: ['humid', 'ottoman', 'outbreak', 'rainfall', 'ottomans', 'recession', 'manpower', 'grt', 'erosion', 'humidity']


    Epoch 1/2:  55%|█████▍    | 18799/34343 [3:15:04<54:35,  4.75it/s]   

    Epoch 1 Batch 18800 loss: 2.192572593688965


    Epoch 1/2:  55%|█████▍    | 18800/34343 [3:15:05<1:23:57,  3.09it/s]

    Closest words to the center word m: ['cdots', 'cdot', 'rightarrow', 'qquad', 'mathbf', 'leq', 'cyg', 'otimes', 'frac', 'mathrm']


    Epoch 1/2:  55%|█████▌    | 18899/34343 [3:16:10<4:38:27,  1.08s/it] 

    Epoch 1 Batch 18900 loss: 2.1832780838012695


    Epoch 1/2:  55%|█████▌    | 18900/34343 [3:16:10<4:07:20,  1.04it/s]

    Closest words to the center word dawlah: ['gwh', 'cyg', 'twh', 'pngimage', 'jul', 'kwh', 'grt', 'pmid', 'finalist', 'icrm']


    Epoch 1/2:  55%|█████▌    | 18999/34343 [3:17:00<1:09:28,  3.68it/s] 

    Epoch 1 Batch 19000 loss: 2.15401029586792


    Epoch 1/2:  55%|█████▌    | 19000/34343 [3:17:00<1:35:46,  2.67it/s]

    Closest words to the center word runs: ['nfc', 'runs', 'afc', 'playoffs', 'runways', 'unpaved', 'divisional', 'grt', 'playoff', 'kilometers']


    Epoch 1/2:  56%|█████▌    | 19099/34343 [3:17:46<53:35,  4.74it/s]   

    Epoch 1 Batch 19100 loss: 2.178741455078125


    Epoch 1/2:  56%|█████▌    | 19100/34343 [3:17:47<1:24:18,  3.01it/s]

    Closest words to the center word word: ['word', 'verb', 'alphabet', 'nouns', 'hebrew', 'infinitive', 'derives', 'adjectives', 'declension', 'aramaic']


    Epoch 1/2:  56%|█████▌    | 19199/34343 [3:18:34<53:49,  4.69it/s]   

    Epoch 1 Batch 19200 loss: 2.2004618644714355


    Epoch 1/2:  56%|█████▌    | 19200/34343 [3:18:35<1:23:38,  3.02it/s]

    Closest words to the center word coherent: ['topological', 'priori', 'homomorphism', 'eukaryotic', 'pluriform', 'countably', 'coherent', 'holomorphic', 'irreducibly', 'thermodynamic']


    Epoch 1/2:  56%|█████▌    | 19299/34343 [3:19:39<4:12:36,  1.01s/it] 

    Epoch 1 Batch 19300 loss: 2.1489410400390625


    Epoch 1/2:  56%|█████▌    | 19300/34343 [3:19:40<3:45:58,  1.11it/s]

    Closest words to the center word national: ['national', 'unctad', 'ifc', 'ifrcs', 'opcw', 'icrm', 'iom', 'unicameral', 'ifad', 'wto']


    Epoch 1/2:  56%|█████▋    | 19399/34343 [3:20:36<55:27,  4.49it/s]   

    Epoch 1 Batch 19400 loss: 2.1447086334228516


    Epoch 1/2:  56%|█████▋    | 19400/34343 [3:20:36<1:24:31,  2.95it/s]

    Closest words to the center word nine: ['gwh', 'kwh', 'twh', 'cyg', 'pngimage', 'jul', 'grt', 'pmid', 'births', 'nfc']


    Epoch 1/2:  57%|█████▋    | 19499/34343 [3:21:32<52:41,  4.70it/s]   

    Epoch 1 Batch 19500 loss: 2.176616668701172


    Epoch 1/2:  57%|█████▋    | 19500/34343 [3:21:32<1:23:06,  2.98it/s]

    Closest words to the center word scripture: ['scripture', 'jesus', 'testament', 'judaism', 'scriptures', 'prophet', 'christianity', 'baptism', 'tanakh', 'teachings']


    Epoch 1/2:  57%|█████▋    | 19599/34343 [3:22:07<51:53,  4.74it/s]  

    Epoch 1 Batch 19600 loss: 2.1506729125976562


    Epoch 1/2:  57%|█████▋    | 19600/34343 [3:22:12<7:41:34,  1.88s/it]

    Closest words to the center word catholic: ['catholic', 'orthodox', 'lutheran', 'anglican', 'churches', 'church', 'episcopal', 'communion', 'protestant', 'denominations']


    Epoch 1/2:  57%|█████▋    | 19699/34343 [3:22:52<58:26,  4.18it/s]   

    Epoch 1 Batch 19700 loss: 2.2006030082702637


    Epoch 1/2:  57%|█████▋    | 19700/34343 [3:22:53<1:30:42,  2.69it/s]

    Closest words to the center word term: ['term', 'noun', 'pejorative', 'adjective', 'word', 'usage', 'derogatory', 'declension', 'synonym', 'verb']


    Epoch 1/2:  58%|█████▊    | 19799/34343 [3:23:37<52:12,  4.64it/s]   

    Epoch 1 Batch 19800 loss: 2.1924495697021484


    Epoch 1/2:  58%|█████▊    | 19800/34343 [3:23:37<1:28:34,  2.74it/s]

    Closest words to the center word exchange: ['kwh', 'exchange', 'gwh', 'unpaved', 'runways', 'expenditures', 'imports', 'telephones', 'capita', 'income']


    Epoch 1/2:  58%|█████▊    | 19899/34343 [3:24:21<53:40,  4.49it/s]   

    Epoch 1 Batch 19900 loss: 2.175184965133667


    Epoch 1/2:  58%|█████▊    | 19900/34343 [3:24:22<1:23:48,  2.87it/s]

    Closest words to the center word immigrant: ['immigrant', 'novelists', 'ethnic', 'entertainers', 'naturalized', 'hispanic', 'indigenous', 'immigrants', 'minority', 'lithuanians']


    Epoch 1/2:  58%|█████▊    | 19999/34343 [3:25:13<1:38:46,  2.42it/s] 

    Epoch 1 Batch 20000 loss: 2.192317008972168


    Epoch 1/2:  58%|█████▊    | 20000/34343 [3:25:14<1:53:43,  2.10it/s]

    Closest words to the center word downtown: ['downtown', 'township', 'devry', 'erie', 'urbana', 'nfc', 'lansing', 'county', 'afc', 'abet']


    Epoch 1/2:  59%|█████▊    | 20099/34343 [3:25:57<1:00:47,  3.91it/s]

    Epoch 1 Batch 20100 loss: 2.1599984169006348


    Epoch 1/2:  59%|█████▊    | 20100/34343 [3:25:58<1:27:51,  2.70it/s]

    Closest words to the center word four: ['gwh', 'cyg', 'grt', 'kwh', 'twh', 'unpaved', 'pngimage', 'pmid', 'jul', 'runways']


    Epoch 1/2:  59%|█████▉    | 20199/34343 [3:26:42<49:56,  4.72it/s]   

    Epoch 1 Batch 20200 loss: 2.2375783920288086


    Epoch 1/2:  59%|█████▉    | 20200/34343 [3:26:42<1:20:05,  2.94it/s]

    Closest words to the center word after: ['after', 'before', 'months', 'shortly', 'grt', 'during', 'lasted', 'leap', 'thereafter', 'ottoman']


    Epoch 1/2:  59%|█████▉    | 20299/34343 [3:27:25<49:14,  4.75it/s]   

    Epoch 1 Batch 20300 loss: 2.1636946201324463


    Epoch 1/2:  59%|█████▉    | 20300/34343 [3:27:26<1:19:07,  2.96it/s]

    Closest words to the center word layer: ['layer', 'intelsat', 'ifrcs', 'iom', 'icrm', 'ozone', 'ifc', 'ifad', 'tanker', 'gwh']


    Epoch 1/2:  59%|█████▉    | 20399/34343 [3:28:18<2:50:54,  1.36it/s] 

    Epoch 1 Batch 20400 loss: 2.1505420207977295


    Epoch 1/2:  59%|█████▉    | 20400/34343 [3:28:19<3:12:03,  1.21it/s]

    Closest words to the center word automobile: ['kwh', 'gwh', 'icrm', 'ifrcs', 'ifc', 'finalist', 'iom', 'pngimage', 'ifad', 'mjs']


    Epoch 1/2:  60%|█████▉    | 20499/34343 [3:29:02<55:40,  4.14it/s]   

    Epoch 1 Batch 20500 loss: 2.1120963096618652


    Epoch 1/2:  60%|█████▉    | 20500/34343 [3:29:03<1:23:43,  2.76it/s]

    Closest words to the center word wives: ['wives', 'daughters', 'eldest', 'sons', 'prophecies', 'pregnant', 'niece', 'elector', 'heirs', 'remarried']


    Epoch 1/2:  60%|█████▉    | 20599/34343 [3:29:44<48:07,  4.76it/s]   

    Epoch 1 Batch 20600 loss: 2.2029638290405273


    Epoch 1/2:  60%|█████▉    | 20600/34343 [3:29:44<1:17:44,  2.95it/s]

    Closest words to the center word set: ['topological', 'morphism', 'set', 'morphisms', 'countably', 'finite', 'homomorphism', 'subset', 'infty', 'countable']


    Epoch 1/2:  60%|██████    | 20699/34343 [3:30:34<47:55,  4.74it/s]   

    Epoch 1 Batch 20700 loss: 2.179687261581421


    Epoch 1/2:  60%|██████    | 20700/34343 [3:30:44<12:17:34,  3.24s/it]

    Closest words to the center word classical: ['classical', 'baroque', 'analytic', 'liberalism', 'composers', 'philosophers', 'renaissance', 'individualist', 'genres', 'physics']


    Epoch 1/2:  61%|██████    | 20799/34343 [3:31:28<2:23:32,  1.57it/s] 

    Epoch 1 Batch 20800 loss: 2.1586945056915283


    Epoch 1/2:  61%|██████    | 20800/34343 [3:31:28<2:26:01,  1.55it/s]

    Closest words to the center word main: ['main', 'telephones', 'demographics', 'factbook', 'bissau', 'faso', 'verde', 'windward', 'federated', 'comoros']


    Epoch 1/2:  61%|██████    | 20899/34343 [3:32:08<49:56,  4.49it/s]  

    Epoch 1 Batch 20900 loss: 2.179399013519287


    Epoch 1/2:  61%|██████    | 20900/34343 [3:32:09<1:20:35,  2.78it/s]

    Closest words to the center word st: ['st', 'nd', 'saint', 'afc', 'nfc', 'earl', 'agave', 'rd', 'th', 'yoannis']


    Epoch 1/2:  61%|██████    | 20999/34343 [3:32:47<47:55,  4.64it/s]   

    Epoch 1 Batch 21000 loss: 2.194061279296875


    Epoch 1/2:  61%|██████    | 21000/34343 [3:32:48<1:24:07,  2.64it/s]

    Closest words to the center word entirely: ['entirely', 'evenly', 'completely', 'exclusively', 'supplanted', 'universally', 'intelligible', 'falsifiable', 'axiomatic', 'identical']


    Epoch 1/2:  61%|██████▏   | 21099/34343 [3:33:47<9:11:50,  2.50s/it] 

    Epoch 1 Batch 21100 loss: 2.1672215461730957


    Epoch 1/2:  61%|██████▏   | 21100/34343 [3:33:48<7:13:04,  1.96s/it]

    Closest words to the center word griffin: ['footballer', 'cricketer', 'politician', 'actor', 'songwriter', 'wrestler', 'philanthropist', 'pianist', 'actress', 'swimmer']


    Epoch 1/2:  62%|██████▏   | 21199/34343 [3:34:28<1:18:32,  2.79it/s] 

    Epoch 1 Batch 21200 loss: 2.168684720993042


    Epoch 1/2:  62%|██████▏   | 21200/34343 [3:34:29<1:40:58,  2.17it/s]

    Closest words to the center word but: ['but', 'falsifiable', 'universally', 'though', 'necessarily', 'countably', 'happen', 'supplanted', 'gregorian', 'because']


    Epoch 1/2:  62%|██████▏   | 21299/34343 [3:35:07<48:27,  4.49it/s]   

    Epoch 1 Batch 21300 loss: 2.2020010948181152


    Epoch 1/2:  62%|██████▏   | 21300/34343 [3:35:08<1:14:56,  2.90it/s]

    Closest words to the center word genres: ['genres', 'singers', 'composers', 'songwriters', 'styles', 'westerns', 'jazz', 'compositions', 'musical', 'themes']


    Epoch 1/2:  62%|██████▏   | 21399/34343 [3:35:48<45:23,  4.75it/s]   

    Epoch 1 Batch 21400 loss: 2.167635202407837


    Epoch 1/2:  62%|██████▏   | 21400/34343 [3:35:48<1:14:18,  2.90it/s]

    Closest words to the center word both: ['both', 'various', 'faiths', 'many', 'several', 'vested', 'pluriform', 'intelligible', 'unicellular', 'incomprehensible']


    Epoch 1/2:  63%|██████▎   | 21499/34343 [3:36:29<4:33:00,  1.28s/it]

    Epoch 1 Batch 21500 loss: 2.1763830184936523


    Epoch 1/2:  63%|██████▎   | 21500/34343 [3:36:30<3:49:15,  1.07s/it]

    Closest words to the center word anka: ['pngimage', 'footballer', 'cyg', 'cricketer', 'jul', 'swimmer', 'finalist', 'laureate', 'bwv', 'eug']


    Epoch 1/2:  63%|██████▎   | 21599/34343 [3:37:11<56:37,  3.75it/s]  

    Epoch 1 Batch 21600 loss: 2.194138526916504


    Epoch 1/2:  63%|██████▎   | 21600/34343 [3:37:11<1:19:39,  2.67it/s]

    Closest words to the center word however: ['gwh', 'argue', 'however', 'kwh', 'believe', 'lifes', 'falsifiable', 'though', 'necessarily', 'vested']


    Epoch 1/2:  63%|██████▎   | 21699/34343 [3:37:54<45:08,  4.67it/s]   

    Epoch 1 Batch 21700 loss: 2.184887409210205


    Epoch 1/2:  63%|██████▎   | 21700/34343 [3:37:55<1:08:40,  3.07it/s]

    Closest words to the center word than: ['than', 'lifes', 'slightly', 'less', 'expectancy', 'millimeters', 'gwh', 'more', 'faster', 'considerably']


    Epoch 1/2:  63%|██████▎   | 21799/34343 [3:38:42<44:36,  4.69it/s]   

    Epoch 1 Batch 21800 loss: 2.1667675971984863


    Epoch 1/2:  63%|██████▎   | 21800/34343 [3:38:42<1:08:38,  3.05it/s]

    Closest words to the center word even: ['even', 'lifes', 'expensive', 'noticeable', 'worse', 'perceive', 'consuming', 'grt', 'occur', 'tolerated']


    Epoch 1/2:  64%|██████▍   | 21899/34343 [3:39:35<2:30:40,  1.38it/s] 

    Epoch 1 Batch 21900 loss: 2.1998085975646973


    Epoch 1/2:  64%|██████▍   | 21900/34343 [3:39:36<2:21:39,  1.46it/s]

    Closest words to the center word most: ['most', 'more', 'less', 'lifes', 'many', 'earliest', 'largest', 'highly', 'populous', 'especially']


    Epoch 1/2:  64%|██████▍   | 21999/34343 [3:40:12<45:34,  4.51it/s]  

    Epoch 1 Batch 22000 loss: 2.1783924102783203


    Epoch 1/2:  64%|██████▍   | 22000/34343 [3:40:13<1:12:41,  2.83it/s]

    Closest words to the center word land: ['arable', 'land', 'irrigated', 'pastures', 'km', 'sq', 'unpaved', 'runways', 'kilometers', 'hydropower']


    Epoch 1/2:  64%|██████▍   | 22099/34343 [3:40:54<43:41,  4.67it/s]   

    Epoch 1 Batch 22100 loss: 2.199709415435791


    Epoch 1/2:  64%|██████▍   | 22100/34343 [3:40:55<1:07:33,  3.02it/s]

    Closest words to the center word descended: ['descended', 'aryans', 'indo', 'slavic', 'assimilated', 'germanic', 'borrowed', 'aramaic', 'inhabited', 'migrated']


    Epoch 1/2:  65%|██████▍   | 22199/34343 [3:41:38<43:12,  4.68it/s]  

    Epoch 1 Batch 22200 loss: 2.1871800422668457


    Epoch 1/2:  65%|██████▍   | 22200/34343 [3:41:38<1:07:47,  2.99it/s]

    Closest words to the center word then: ['otimes', 'infty', 'cdots', 'rangle', 'operatorname', 'qquad', 'leq', 'ldots', 'cdot', 'bigg']


    Epoch 1/2:  65%|██████▍   | 22299/34343 [3:42:30<1:14:40,  2.69it/s] 

    Epoch 1 Batch 22300 loss: 2.1602892875671387


    Epoch 1/2:  65%|██████▍   | 22300/34343 [3:42:31<1:29:12,  2.25it/s]

    Closest words to the center word th: ['th', 'nd', 'nfc', 'nineteenth', 'rd', 'afc', 'gwh', 'twentieth', 'seventeenth', 'sixteenth']


    Epoch 1/2:  65%|██████▌   | 22399/34343 [3:43:19<59:56,  3.32it/s]   

    Epoch 1 Batch 22400 loss: 2.1497557163238525


    Epoch 1/2:  65%|██████▌   | 22400/34343 [3:43:19<1:20:30,  2.47it/s]

    Closest words to the center word broadly: ['broadly', 'nouns', 'linguists', 'goidelic', 'fundamentalist', 'heretical', 'dialects', 'phonology', 'dravidian', 'urdu']


    Epoch 1/2:  66%|██████▌   | 22499/34343 [3:44:05<41:26,  4.76it/s]   

    Epoch 1 Batch 22500 loss: 2.182007312774658


    Epoch 1/2:  66%|██████▌   | 22500/34343 [3:44:05<1:04:32,  3.06it/s]

    Closest words to the center word nation: ['populous', 'republic', 'nation', 'legislature', 'bicameral', 'timor', 'territory', 'subcontinent', 'factbook', 'macedonia']


    Epoch 1/2:  66%|██████▌   | 22599/34343 [3:44:57<1:35:55,  2.04it/s] 

    Epoch 1 Batch 22600 loss: 2.1575498580932617


    Epoch 1/2:  66%|██████▌   | 22600/34343 [3:44:58<1:46:26,  1.84it/s]

    Closest words to the center word all: ['all', 'abelian', 'morphisms', 'integers', 'identical', 'reals', 'various', 'finite', 'many', 'countably']


    Epoch 1/2:  66%|██████▌   | 22699/34343 [3:45:41<1:04:36,  3.00it/s] 

    Epoch 1 Batch 22700 loss: 2.166353702545166


    Epoch 1/2:  66%|██████▌   | 22700/34343 [3:45:41<1:21:13,  2.39it/s]

    Closest words to the center word n: ['cdots', 'cdot', 'otimes', 'rightarrow', 'qquad', 'leq', 'mathrm', 'frac', 'ldots', 'operatorname']


    Epoch 1/2:  66%|██████▋   | 22799/34343 [3:46:24<41:33,  4.63it/s]   

    Epoch 1 Batch 22800 loss: 2.1599996089935303


    Epoch 1/2:  66%|██████▋   | 22800/34343 [3:46:24<1:04:53,  2.96it/s]

    Closest words to the center word century: ['century', 'centuries', 'nfc', 'afc', 'bc', 'nineteenth', 'bce', 'dynasty', 'philosophers', 'th']


    Epoch 1/2:  67%|██████▋   | 22899/34343 [3:47:20<40:37,  4.69it/s]   

    Epoch 1 Batch 22900 loss: 2.165294885635376


    Epoch 1/2:  67%|██████▋   | 22900/34343 [3:47:20<1:02:53,  3.03it/s]

    Closest words to the center word qquad: ['qquad', 'cdot', 'cdots', 'otimes', 'frac', 'mathbf', 'leq', 'rangle', 'ldots', 'mathrm']


    Epoch 1/2:  67%|██████▋   | 22999/34343 [3:48:13<2:27:26,  1.28it/s]

    Epoch 1 Batch 23000 loss: 2.1681160926818848


    Epoch 1/2:  67%|██████▋   | 23000/34343 [3:48:14<2:22:11,  1.33it/s]

    Closest words to the center word organization: ['ifrcs', 'ifc', 'iom', 'unido', 'unctad', 'ifad', 'opcw', 'ilo', 'upu', 'iho']


    Epoch 1/2:  67%|██████▋   | 23099/34343 [3:49:02<46:25,  4.04it/s]  

    Epoch 1 Batch 23100 loss: 2.163196325302124


    Epoch 1/2:  67%|██████▋   | 23100/34343 [3:49:02<1:10:34,  2.66it/s]

    Closest words to the center word could: ['could', 'can', 'should', 'would', 'might', 'must', 'shall', 'will', 'wouldn', 'did']


    Epoch 1/2:  68%|██████▊   | 23199/34343 [3:49:45<41:47,  4.44it/s]  

    Epoch 1 Batch 23200 loss: 2.1257312297821045


    Epoch 1/2:  68%|██████▊   | 23200/34343 [3:49:46<1:03:32,  2.92it/s]

    Closest words to the center word situation: ['situation', 'pluriform', 'vested', 'recession', 'impose', 'coercion', 'coup', 'laissez', 'regimes', 'restructuring']


    Epoch 1/2:  68%|██████▊   | 23299/34343 [3:50:34<38:57,  4.72it/s]   

    Epoch 1 Batch 23300 loss: 2.1690590381622314


    Epoch 1/2:  68%|██████▊   | 23300/34343 [3:50:34<1:00:40,  3.03it/s]

    Closest words to the center word has: ['has', 'had', 'have', 'having', 'pluriform', 'ifrcs', 'been', 'ifc', 'ifad', 'enjoys']


    Epoch 1/2:  68%|██████▊   | 23399/34343 [3:51:32<2:22:20,  1.28it/s] 

    Epoch 1 Batch 23400 loss: 2.1786019802093506


    Epoch 1/2:  68%|██████▊   | 23400/34343 [3:51:33<2:12:50,  1.37it/s]

    Closest words to the center word see: ['see', 'list', 'disambiguation', 'newnode', 'est', 'topics', 'wtoo', 'redirects', 'wmo', 'ifrcs']


    Epoch 1/2:  68%|██████▊   | 23499/34343 [3:52:33<43:18,  4.17it/s]   

    Epoch 1 Batch 23500 loss: 2.1621744632720947


    Epoch 1/2:  68%|██████▊   | 23500/34343 [3:52:34<1:04:50,  2.79it/s]

    Closest words to the center word heard: ['heard', 'sorry', 'wouldn', 'loved', 'tonight', 'couldn', 'cried', 'replied', 'll', 'hadn']


    Epoch 1/2:  69%|██████▊   | 23599/34343 [3:53:22<38:29,  4.65it/s]  

    Epoch 1 Batch 23600 loss: 2.185946464538574


    Epoch 1/2:  69%|██████▊   | 23600/34343 [3:53:23<59:35,  3.00it/s]

    Closest words to the center word observations: ['observations', 'relativity', 'assumptions', 'aether', 'planets', 'gravitational', 'galaxies', 'diffraction', 'findable', 'mechanics']


    Epoch 1/2:  69%|██████▉   | 23699/34343 [3:54:27<14:50:07,  5.02s/it]

    Epoch 1 Batch 23700 loss: 2.178269386291504


    Epoch 1/2:  69%|██████▉   | 23700/34343 [3:54:27<10:55:07,  3.69s/it]

    Closest words to the center word people: ['people', 'births', 'albanians', 'jews', 'refugees', 'americans', 'nationality', 'immigrants', 'africans', 'citizens']


    Epoch 1/2:  69%|██████▉   | 23799/34343 [3:55:17<1:07:02,  2.62it/s] 

    Epoch 1 Batch 23800 loss: 2.169729709625244


    Epoch 1/2:  69%|██████▉   | 23800/34343 [3:55:18<1:30:42,  1.94it/s]

    Closest words to the center word game: ['game', 'games', 'console', 'nintendo', 'playstation', 'championship', 'consoles', 'playoff', 'nfc', 'afc']


    Epoch 1/2:  70%|██████▉   | 23899/34343 [3:56:25<38:16,  4.55it/s]   

    Epoch 1 Batch 23900 loss: 2.174274444580078


    Epoch 1/2:  70%|██████▉   | 23900/34343 [3:56:26<57:46,  3.01it/s]

    Closest words to the center word minister: ['minister', 'prime', 'deputy', 'ministers', 'cdu', 'chancellor', 'boutros', 'attlee', 'secretary', 'president']


    Epoch 1/2:  70%|██████▉   | 23999/34343 [3:57:19<37:32,  4.59it/s]   

    Epoch 1 Batch 24000 loss: 2.1914472579956055


    Epoch 1/2:  70%|██████▉   | 24000/34343 [3:57:20<59:25,  2.90it/s]

    Closest words to the center word known: ['known', 'referred', 'regarded', 'agave', 'classified', 'suited', 'classed', 'described', 'abbreviated', 'argumentum']


    Epoch 1/2:  70%|███████   | 24099/34343 [3:58:33<7:27:52,  2.62s/it] 

    Epoch 1 Batch 24100 loss: 2.1623618602752686


    Epoch 1/2:  70%|███████   | 24100/34343 [3:58:34<5:44:44,  2.02s/it]

    Closest words to the center word the: ['pluriform', 'the', 'nfc', 'ifrcs', 'vernal', 'icrm', 'ifc', 'afc', 'intelsat', 'desertification']


    Epoch 1/2:  70%|███████   | 24199/34343 [3:59:34<43:35,  3.88it/s]   

    Epoch 1 Batch 24200 loss: 2.130772352218628


    Epoch 1/2:  70%|███████   | 24200/34343 [3:59:34<1:01:35,  2.74it/s]

    Closest words to the center word left: ['left', 'rangle', 'otimes', 'cdots', 'cdot', 'frac', 'mathbf', 'operatorname', 'rang', 'right']


    Epoch 1/2:  71%|███████   | 24299/34343 [4:00:27<37:45,  4.43it/s]   

    Epoch 1 Batch 24300 loss: 2.1426546573638916


    Epoch 1/2:  71%|███████   | 24300/34343 [4:00:28<1:06:34,  2.51it/s]

    Closest words to the center word grover: ['township', 'footballer', 'earl', 'leiserson', 'cricketer', 'viscount', 'finalist', 'ssn', 'laureate', 'cyg']


    Epoch 1/2:  71%|███████   | 24399/34343 [4:01:32<35:22,  4.68it/s]   

    Epoch 1 Batch 24400 loss: 2.164360523223877


    Epoch 1/2:  71%|███████   | 24400/34343 [4:01:33<57:52,  2.86it/s]

    Closest words to the center word catholics: ['catholics', 'orthodox', 'denominations', 'christians', 'churches', 'theologians', 'anglicans', 'protestants', 'protestant', 'bishops']


    Epoch 1/2:  71%|███████▏  | 24499/34343 [4:02:38<1:34:36,  1.73it/s] 

    Epoch 1 Batch 24500 loss: 2.15714693069458


    Epoch 1/2:  71%|███████▏  | 24500/34343 [4:02:39<1:36:26,  1.70it/s]

    Closest words to the center word after: ['after', 'before', 'shortly', 'months', 'during', 'afterwards', 'lasted', 'thereafter', 'grt', 'abdur']


    Epoch 1/2:  72%|███████▏  | 24599/34343 [4:03:26<37:17,  4.35it/s]  

    Epoch 1 Batch 24600 loss: 2.1690213680267334


    Epoch 1/2:  72%|███████▏  | 24600/34343 [4:03:27<1:00:02,  2.70it/s]

    Closest words to the center word others: ['others', 'agnostics', 'psychologists', 'scholars', 'neopagans', 'masculists', 'atheists', 'sects', 'unido', 'feminists']


    Epoch 1/2:  72%|███████▏  | 24699/34343 [4:04:25<34:29,  4.66it/s]   

    Epoch 1 Batch 24700 loss: 2.1629137992858887


    Epoch 1/2:  72%|███████▏  | 24700/34343 [4:04:26<57:08,  2.81it/s]

    Closest words to the center word history: ['history', 'demographics', 'allafrica', 'factbook', 'geography', 'federated', 'timeline', 'barbuda', 'topics', 'archaeology']


    Epoch 1/2:  72%|███████▏  | 24799/34343 [4:05:37<8:46:35,  3.31s/it] 

    Epoch 1 Batch 24800 loss: 2.1606433391571045


    Epoch 1/2:  72%|███████▏  | 24800/34343 [4:05:38<6:41:02,  2.52s/it]

    Closest words to the center word tao: ['tao', 'tiberian', 'otimes', 'ching', 'landsmannschaft', 'nevi', 'factum', 'ctus', 'infinitive', 'polskiej']


    Epoch 1/2:  73%|███████▎  | 24899/34343 [4:06:36<1:05:32,  2.40it/s] 

    Epoch 1 Batch 24900 loss: 2.2006969451904297


    Epoch 1/2:  73%|███████▎  | 24900/34343 [4:06:37<1:13:43,  2.13it/s]

    Closest words to the center word criminal: ['criminal', 'enforcement', 'tribunal', 'unido', 'crimes', 'jurisdiction', 'appellate', 'liability', 'defendant', 'judicial']


    Epoch 1/2:  73%|███████▎  | 24999/34343 [4:07:33<34:59,  4.45it/s]   

    Epoch 1 Batch 25000 loss: 2.1721980571746826


    Epoch 1/2:  73%|███████▎  | 25000/34343 [4:07:34<57:53,  2.69it/s]

    Closest words to the center word and: ['ifrcs', 'ifad', 'kwh', 'gwh', 'ifc', 'icrm', 'unctad', 'ibrd', 'opcw', 'icftu']


    Epoch 1/2:  73%|███████▎  | 25099/34343 [4:08:22<32:50,  4.69it/s]  

    Epoch 1 Batch 25100 loss: 2.1649208068847656


    Epoch 1/2:  73%|███████▎  | 25100/34343 [4:08:22<53:48,  2.86it/s]

    Closest words to the center word vfl: ['nfc', 'afc', 'playoffs', 'championship', 'nfl', 'divisional', 'playoff', 'finals', 'steelers', 'broncos']


    Epoch 1/2:  73%|███████▎  | 25199/34343 [4:09:34<6:38:34,  2.62s/it] 

    Epoch 1 Batch 25200 loss: 2.140643358230591


    Epoch 1/2:  73%|███████▎  | 25200/34343 [4:09:34<5:11:34,  2.04s/it]

    Closest words to the center word one: ['gwh', 'kwh', 'cyg', 'grt', 'twh', 'pngimage', 'jul', 'unpaved', 'pmid', 'hbk']


    Epoch 1/2:  74%|███████▎  | 25299/34343 [4:10:41<57:17,  2.63it/s]   

    Epoch 1 Batch 25300 loss: 2.167630910873413


    Epoch 1/2:  74%|███████▎  | 25300/34343 [4:10:41<1:11:59,  2.09it/s]

    Closest words to the center word some: ['some', 'many', 'various', 'several', 'certain', 'anecdotal', 'hundreds', 'lifes', 'most', 'faiths']


    Epoch 1/2:  74%|███████▍  | 25399/34343 [4:11:50<33:00,  4.52it/s]   

    Epoch 1 Batch 25400 loss: 2.109294891357422


    Epoch 1/2:  74%|███████▍  | 25400/34343 [4:11:50<53:27,  2.79it/s]

    Closest words to the center word added: ['added', 'bits', 'gwh', 'kwh', 'telephones', 'twh', 'unpaved', 'cyg', 'grt', 'demographics']


    Epoch 1/2:  74%|███████▍  | 25499/34343 [4:12:56<31:12,  4.72it/s]   

    Epoch 1 Batch 25500 loss: 2.1761088371276855


    Epoch 1/2:  74%|███████▍  | 25500/34343 [4:12:57<52:49,  2.79it/s]

    Closest words to the center word beginner: ['beginner', 'findable', 'tutorials', 'karaoke', 'brainfuck', 'venues', 'downloads', 'databases', 'lossless', 'lossy']


    Epoch 1/2:  75%|███████▍  | 25599/34343 [4:14:13<4:34:06,  1.88s/it] 

    Epoch 1 Batch 25600 loss: 2.1636462211608887


    Epoch 1/2:  75%|███████▍  | 25600/34343 [4:14:14<3:43:12,  1.53s/it]

    Closest words to the center word the: ['the', 'nfc', 'pluriform', 'vernal', 'populous', 'desertification', 'vested', 'scrimmage', 'bicameral', 'ifrcs']


    Epoch 1/2:  75%|███████▍  | 25699/34343 [4:15:16<39:02,  3.69it/s]   

    Epoch 1 Batch 25700 loss: 2.1753110885620117


    Epoch 1/2:  75%|███████▍  | 25700/34343 [4:15:16<53:50,  2.68it/s]

    Closest words to the center word property: ['property', 'pluriform', 'ethical', 'monetary', 'welfare', 'equality', 'liability', 'intellectual', 'rights', 'profit']


    Epoch 1/2:  75%|███████▌  | 25799/34343 [4:16:05<30:36,  4.65it/s]  

    Epoch 1 Batch 25800 loss: 2.143805503845215


    Epoch 1/2:  75%|███████▌  | 25800/34343 [4:16:06<46:22,  3.07it/s]

    Closest words to the center word six: ['gwh', 'kwh', 'grt', 'cyg', 'pngimage', 'twh', 'unpaved', 'mjs', 'pmid', 'jul']


    Epoch 1/2:  75%|███████▌  | 25899/34343 [4:17:04<29:39,  4.74it/s]   

    Epoch 1 Batch 25900 loss: 2.1542530059814453


    Epoch 1/2:  75%|███████▌  | 25900/34343 [4:17:05<46:21,  3.04it/s]

    Closest words to the center word were: ['were', 'are', 'camps', 'persecuted', 'romans', 'been', 'serbs', 'brutally', 'greeks', 'mongols']


    Epoch 1/2:  76%|███████▌  | 25999/34343 [4:18:06<1:58:38,  1.17it/s]

    Epoch 1 Batch 26000 loss: 2.15213680267334


    Epoch 1/2:  76%|███████▌  | 26000/34343 [4:18:07<1:49:01,  1.28it/s]

    Closest words to the center word reject: ['reject', 'laissez', 'ethical', 'egoism', 'reconstructionist', 'fundamentalists', 'naturalism', 'deny', 'relativism', 'anarcho']


    Epoch 1/2:  76%|███████▌  | 26099/34343 [4:19:15<35:07,  3.91it/s]   

    Epoch 1 Batch 26100 loss: 2.172351837158203


    Epoch 1/2:  76%|███████▌  | 26100/34343 [4:19:16<51:11,  2.68it/s]

    Closest words to the center word supported: ['unido', 'supported', 'unctad', 'pluriform', 'upu', 'wmo', 'wipo', 'criticized', 'exercised', 'vested']


    Epoch 1/2:  76%|███████▋  | 26199/34343 [4:20:09<28:37,  4.74it/s]  

    Epoch 1 Batch 26200 loss: 2.1398873329162598


    Epoch 1/2:  76%|███████▋  | 26200/34343 [4:20:10<46:27,  2.92it/s]

    Closest words to the center word often: ['often', 'commonly', 'sometimes', 'interchangeably', 'widely', 'generally', 'frequently', 'synonymously', 'usually', 'chemically']


    Epoch 1/2:  77%|███████▋  | 26299/34343 [4:21:13<28:15,  4.74it/s]   

    Epoch 1 Batch 26300 loss: 2.1213347911834717


    Epoch 1/2:  77%|███████▋  | 26300/34343 [4:21:30<11:31:49,  5.16s/it]

    Closest words to the center word noblemen: ['noblemen', 'nobles', 'brutally', 'bolsheviks', 'vassals', 'persecuted', 'fled', 'persecutions', 'pashtun', 'raped']


    Epoch 1/2:  77%|███████▋  | 26399/34343 [4:22:28<1:00:32,  2.19it/s] 

    Epoch 1 Batch 26400 loss: 2.1519837379455566


    Epoch 1/2:  77%|███████▋  | 26400/34343 [4:22:29<1:09:46,  1.90it/s]

    Closest words to the center word at: ['at', 'devry', 'graduate', 'abet', 'near', 'ssn', 'utc', 'hochschule', 'attended', 'urbana']


    Epoch 1/2:  77%|███████▋  | 26499/34343 [4:23:38<28:52,  4.53it/s]   

    Epoch 1 Batch 26500 loss: 2.1253271102905273


    Epoch 1/2:  77%|███████▋  | 26500/34343 [4:23:39<45:35,  2.87it/s]

    Closest words to the center word k: ['otimes', 'cdot', 'rightarrow', 'cdots', 'leq', 'operatorname', 'qquad', 'ldots', 'rangle', 'mathrm']


    Epoch 1/2:  77%|███████▋  | 26599/34343 [4:24:44<27:19,  4.72it/s]   

    Epoch 1 Batch 26600 loss: 2.144522190093994


    Epoch 1/2:  77%|███████▋  | 26600/34343 [4:24:45<46:49,  2.76it/s]

    Closest words to the center word rex: ['tamarin', 'callithrix', 'marmoset', 'leiserson', 'cegep', 'wco', 'sieur', 'eulemur', 'eug', 'agave']


    Epoch 1/2:  78%|███████▊  | 26699/34343 [4:26:10<7:26:30,  3.50s/it] 

    Epoch 1 Batch 26700 loss: 2.166740655899048


    Epoch 1/2:  78%|███████▊  | 26700/34343 [4:26:11<5:37:01,  2.65s/it]

    Closest words to the center word air: ['air', 'ifrcs', 'icrm', 'ifad', 'navy', 'ifc', 'opcw', 'missiles', 'pollution', 'iom']


    Epoch 1/2:  78%|███████▊  | 26799/34343 [4:27:08<30:46,  4.09it/s]   

    Epoch 1 Batch 26800 loss: 2.105173110961914


    Epoch 1/2:  78%|███████▊  | 26800/34343 [4:27:09<43:43,  2.88it/s]

    Closest words to the center word word: ['word', 'kanji', 'declension', 'pronoun', 'sanskrit', 'alphabet', 'verb', 'hebrew', 'tiberian', 'transliteration']


    Epoch 1/2:  78%|███████▊  | 26899/34343 [4:28:17<26:42,  4.64it/s]   

    Epoch 1 Batch 26900 loss: 2.165642023086548


    Epoch 1/2:  78%|███████▊  | 26900/34343 [4:28:18<1:08:59,  1.80it/s]

    Closest words to the center word its: ['its', 'their', 'ifrcs', 'telephones', 'ifad', 'our', 'depends', 'vested', 'iho', 'varies']


    Epoch 1/2:  79%|███████▊  | 26999/34343 [4:29:10<26:29,  4.62it/s]  

    Epoch 1 Batch 27000 loss: 2.178239583969116


    Epoch 1/2:  79%|███████▊  | 27000/34343 [4:29:10<41:04,  2.98it/s]

    Closest words to the center word accessible: ['accessible', 'findable', 'lossy', 'apis', 'gprs', 'expensive', 'accessing', 'available', 'browsers', 'cheaper']


    Epoch 1/2:  79%|███████▉  | 27099/34343 [4:30:30<1:32:48,  1.30it/s] 

    Epoch 1 Batch 27100 loss: 2.155221462249756


    Epoch 1/2:  79%|███████▉  | 27100/34343 [4:30:30<1:27:15,  1.38it/s]

    Closest words to the center word clergyman: ['clergyman', 'politician', 'statesman', 'theologian', 'novelist', 'laureate', 'philanthropist', 'cricketer', 'footballer', 'dramatist']


    Epoch 1/2:  79%|███████▉  | 27199/34343 [4:31:32<29:56,  3.98it/s]   

    Epoch 1 Batch 27200 loss: 2.1478168964385986


    Epoch 1/2:  79%|███████▉  | 27200/34343 [4:31:32<42:45,  2.78it/s]

    Closest words to the center word as: ['as', 'argumentum', 'clavier', 'regarded', 'colloquially', 'agave', 'hominem', 'boutros', 'insofar', 'popularly']


    Epoch 1/2:  79%|███████▉  | 27299/34343 [4:32:43<25:35,  4.59it/s]   

    Epoch 1 Batch 27300 loss: 2.148089647293091


    Epoch 1/2:  79%|███████▉  | 27300/34343 [4:32:43<39:47,  2.95it/s]

    Closest words to the center word rogers: ['comedian', 'hammett', 'actress', 'bandleader', 'lesh', 'mcvie', 'ronnie', 'coach', 'comedienne', 'actor']


    Epoch 1/2:  80%|███████▉  | 27399/34343 [4:33:58<9:54:46,  5.14s/it]

    Epoch 1 Batch 27400 loss: 2.1383790969848633


    Epoch 1/2:  80%|███████▉  | 27400/34343 [4:33:58<7:17:52,  3.78s/it]

    Closest words to the center word material: ['material', 'findable', 'cvd', 'rna', 'amplification', 'solvents', 'dioxide', 'molecules', 'eukaryotic', 'tissues']


    Epoch 1/2:  80%|████████  | 27499/34343 [4:34:57<1:08:43,  1.66it/s]

    Epoch 1 Batch 27500 loss: 2.1313536167144775


    Epoch 1/2:  80%|████████  | 27500/34343 [4:34:57<1:09:04,  1.65it/s]

    Closest words to the center word calls: ['newnode', 'findable', 'calls', 'asks', 'prev', 'asking', 'unido', 'wants', 'thank', 'requests']


    Epoch 1/2:  80%|████████  | 27599/34343 [4:36:03<27:07,  4.14it/s]   

    Epoch 1 Batch 27600 loss: 2.1406517028808594


    Epoch 1/2:  80%|████████  | 27600/34343 [4:36:04<40:19,  2.79it/s]

    Closest words to the center word production: ['kwh', 'gwh', 'production', 'twh', 'commodities', 'textiles', 'imports', 'fuels', 'exports', 'electricity']


    Epoch 1/2:  81%|████████  | 27699/34343 [4:37:07<23:49,  4.65it/s]  

    Epoch 1 Batch 27700 loss: 2.1324963569641113


    Epoch 1/2:  81%|████████  | 27700/34343 [4:37:08<41:34,  2.66it/s]

    Closest words to the center word chronicle: ['chronicle', 'nevi', 'mamre', 'mechon', 'hammadi', 'afc', 'nfc', 'dramatists', 'encyclop', 'tanakh']


    Epoch 1/2:  81%|████████  | 27799/34343 [4:38:15<3:53:54,  2.14s/it]

    Epoch 1 Batch 27800 loss: 2.1638875007629395


    Epoch 1/2:  81%|████████  | 27800/34343 [4:38:16<3:06:28,  1.71s/it]

    Closest words to the center word spoken: ['spoken', 'dialects', 'urdu', 'languages', 'slavic', 'speakers', 'bangla', 'afrikaans', 'dialect', 'hindi']


    Epoch 1/2:  81%|████████  | 27899/34343 [4:39:18<51:57,  2.07it/s]  

    Epoch 1 Batch 27900 loss: 2.159383535385132


    Epoch 1/2:  81%|████████  | 27900/34343 [4:39:18<59:29,  1.80it/s]

    Closest words to the center word animation: ['animation', 'animated', 'anime', 'consoles', 'codecs', 'machinima', 'lossy', 'graphics', 'karaoke', 'lossless']


    Epoch 1/2:  82%|████████▏ | 27999/34343 [4:40:27<24:06,  4.39it/s]  

    Epoch 1 Batch 28000 loss: 2.138845920562744


    Epoch 1/2:  82%|████████▏ | 28000/34343 [4:40:27<38:53,  2.72it/s]

    Closest words to the center word beginning: ['gregorian', 'beginning', 'cretaceous', 'leap', 'end', 'proleptic', 'triassic', 'outbreak', 'vernal', 'nineteenth']


    Epoch 1/2:  82%|████████▏ | 28099/34343 [4:41:37<22:21,  4.66it/s]  

    Epoch 1 Batch 28100 loss: 2.157573699951172


    Epoch 1/2:  82%|████████▏ | 28100/34343 [4:41:38<35:55,  2.90it/s]

    Closest words to the center word k: ['otimes', 'cdot', 'cdots', 'leq', 'rightarrow', 'qquad', 'ldots', 'operatorname', 'infty', 'rangle']


    Epoch 1/2:  82%|████████▏ | 28199/34343 [4:42:49<3:50:50,  2.25s/it]

    Epoch 1 Batch 28200 loss: 2.166411876678467


    Epoch 1/2:  82%|████████▏ | 28200/34343 [4:42:49<3:03:17,  1.79s/it]

    Closest words to the center word super: ['afc', 'nfc', 'super', 'playoffs', 'playoff', 'divisional', 'kart', 'pngimage', 'bowl', 'steelers']


    Epoch 1/2:  82%|████████▏ | 28299/34343 [4:43:49<35:29,  2.84it/s]  

    Epoch 1 Batch 28300 loss: 2.1673877239227295


    Epoch 1/2:  82%|████████▏ | 28300/34343 [4:43:50<45:45,  2.20it/s]

    Closest words to the center word by: ['by', 'ifrcs', 'ifc', 'icrm', 'ifad', 'agave', 'unctad', 'unido', 'ibrd', 'xaver']


    Epoch 1/2:  83%|████████▎ | 28399/34343 [4:45:00<21:53,  4.53it/s]  

    Epoch 1 Batch 28400 loss: 2.122408390045166


    Epoch 1/2:  83%|████████▎ | 28400/34343 [4:45:01<36:13,  2.73it/s]

    Closest words to the center word out: ['out', 'down', 'up', 'off', 'away', 'scrimmage', 'prev', 'onto', 'reins', 'couldn']


    Epoch 1/2:  83%|████████▎ | 28499/34343 [4:46:09<20:33,  4.74it/s]  

    Epoch 1 Batch 28500 loss: 2.1728084087371826


    Epoch 1/2:  83%|████████▎ | 28500/34343 [4:46:09<34:36,  2.81it/s]

    Closest words to the center word death: ['death', 'resurrection', 'expectancy', 'infant', 'tenji', 'imprisonment', 'mortality', 'birth', 'jehoram', 'sentenced']


    Epoch 1/2:  83%|████████▎ | 28599/34343 [4:47:33<2:05:07,  1.31s/it]

    Epoch 1 Batch 28600 loss: 2.1430716514587402


    Epoch 1/2:  83%|████████▎ | 28600/34343 [4:47:34<1:48:30,  1.13s/it]

    Closest words to the center word masters: ['nfc', 'masters', 'championships', 'championship', 'winners', 'champions', 'afc', 'champion', 'finals', 'wimbledon']


    Epoch 1/2:  84%|████████▎ | 28699/34343 [4:48:45<26:15,  3.58it/s]  

    Epoch 1 Batch 28700 loss: 2.115452289581299


    Epoch 1/2:  84%|████████▎ | 28700/34343 [4:48:45<38:24,  2.45it/s]

    Closest words to the center word children: ['children', 'householder', 'couples', 'males', 'females', 'female', 'daughters', 'parents', 'male', 'adults']


    Epoch 1/2:  84%|████████▍ | 28799/34343 [4:49:54<19:54,  4.64it/s]  

    Epoch 1 Batch 28800 loss: 2.215231418609619


    Epoch 1/2:  84%|████████▍ | 28800/34343 [4:49:55<33:19,  2.77it/s]

    Closest words to the center word iraq: ['iraq', 'saudi', 'syria', 'iraqi', 'kuwait', 'dinar', 'peacekeeping', 'gaza', 'hussein', 'lebanon']


    Epoch 1/2:  84%|████████▍ | 28899/34343 [4:51:03<19:08,  4.74it/s]  

    Epoch 1 Batch 28900 loss: 2.143759250640869


    Epoch 1/2:  84%|████████▍ | 28900/34343 [4:51:03<32:34,  2.78it/s]

    Closest words to the center word salt: ['beets', 'broadleaf', 'cassava', 'salt', 'humid', 'limestone', 'subtropical', 'arable', 'citrus', 'gravel']


    Epoch 1/2:  84%|████████▍ | 28999/34343 [4:52:27<1:09:25,  1.28it/s]

    Epoch 1 Batch 29000 loss: 2.1738033294677734


    Epoch 1/2:  84%|████████▍ | 29000/34343 [4:52:28<1:06:59,  1.33it/s]

    Closest words to the center word the: ['the', 'nfc', 'pluriform', 'ifrcs', 'vernal', 'ifc', 'newnode', 'afc', 'scrimmage', 'nicaea']


    Epoch 1/2:  85%|████████▍ | 29099/34343 [4:53:37<19:26,  4.50it/s]  

    Epoch 1 Batch 29100 loss: 2.130866289138794


    Epoch 1/2:  85%|████████▍ | 29100/34343 [4:53:37<31:46,  2.75it/s]

    Closest words to the center word whether: ['whether', 'falsifiable', 'subjective', 'ethical', 'countably', 'causal', 'cardinality', 'verifiable', 'normative', 'disprove']


    Epoch 1/2:  85%|████████▌ | 29199/34343 [4:54:46<18:11,  4.71it/s]  

    Epoch 1 Batch 29200 loss: 2.167146682739258


    Epoch 1/2:  85%|████████▌ | 29200/34343 [4:54:47<29:43,  2.88it/s]

    Closest words to the center word wear: ['wear', 'pants', 'eat', 'wears', 'throw', 'tamarin', 'wore', 'wearing', 'dresses', 'dressing']


    Epoch 1/2:  85%|████████▌ | 29299/34343 [4:56:10<6:56:10,  4.95s/it]

    Epoch 1 Batch 29300 loss: 2.1464898586273193


    Epoch 1/2:  85%|████████▌ | 29300/34343 [4:56:11<5:09:19,  3.68s/it]

    Closest words to the center word canaan: ['assyria', 'canaan', 'persia', 'constantinople', 'ceded', 'conquered', 'judah', 'heraclius', 'sumer', 'flee']


    Epoch 1/2:  86%|████████▌ | 29399/34343 [4:57:19<26:18,  3.13it/s]  

    Epoch 1 Batch 29400 loss: 2.1150240898132324


    Epoch 1/2:  86%|████████▌ | 29400/34343 [4:57:20<34:01,  2.42it/s]

    Closest words to the center word principal: ['principal', 'largest', 'ifad', 'vested', 'administrative', 'gangetic', 'main', 'leeward', 'primary', 'ifc']


    Epoch 1/2:  86%|████████▌ | 29499/34343 [4:58:28<18:08,  4.45it/s]  

    Epoch 1 Batch 29500 loss: 2.1666815280914307


    Epoch 1/2:  86%|████████▌ | 29500/34343 [4:58:28<30:55,  2.61it/s]

    Closest words to the center word of: ['of', 'ifrcs', 'ifad', 'ifc', 'akan', 'icrm', 'wtoo', 'rajonas', 'nazarene', 'iom']


    Epoch 1/2:  86%|████████▌ | 29599/34343 [4:59:38<16:56,  4.67it/s]  

    Epoch 1 Batch 29600 loss: 2.1527047157287598


    Epoch 1/2:  86%|████████▌ | 29600/34343 [4:59:38<28:13,  2.80it/s]

    Closest words to the center word a: ['a', 'pluriform', 'automorphism', 'morphism', 'cardinality', 'every', 'otimes', 'newnode', 'another', 'any']


    Epoch 1/2:  86%|████████▋ | 29699/34343 [5:01:03<1:19:02,  1.02s/it]

    Epoch 1 Batch 29700 loss: 2.139249324798584


    Epoch 1/2:  86%|████████▋ | 29700/34343 [5:01:03<1:09:47,  1.11it/s]

    Closest words to the center word canadian: ['canadian', 'footballer', 'american', 'ifrcs', 'cricketer', 'australian', 'kwh', 'ifc', 'laureate', 'comedian']


    Epoch 1/2:  87%|████████▋ | 29799/34343 [5:02:10<20:47,  3.64it/s]  

    Epoch 1 Batch 29800 loss: 2.1743788719177246


    Epoch 1/2:  87%|████████▋ | 29800/34343 [5:02:11<28:17,  2.68it/s]

    Closest words to the center word metres: ['grt', 'gwh', 'metres', 'inches', 'meters', 'unpaved', 'twh', 'runways', 'ft', 'kilometers']


    Epoch 1/2:  87%|████████▋ | 29899/34343 [5:03:19<17:31,  4.22it/s]  

    Epoch 1 Batch 29900 loss: 2.123094081878662


    Epoch 1/2:  87%|████████▋ | 29900/34343 [5:03:20<26:14,  2.82it/s]

    Closest words to the center word aka: ['pngimage', 'ifrcs', 'icrm', 'mjs', 'ifc', 'ifad', 'agave', 'icftu', 'ibrd', 'tamarin']


    Epoch 1/2:  87%|████████▋ | 29999/34343 [5:04:28<15:29,  4.67it/s]  

    Epoch 1 Batch 30000 loss: 2.1571402549743652


    Epoch 1/2:  87%|████████▋ | 30000/34343 [5:04:45<6:17:40,  5.22s/it]

    Closest words to the center word singapore: ['opcw', 'chungcheong', 'barbuda', 'swaziland', 'icrm', 'busan', 'kwh', 'ifrcs', 'singapore', 'rupee']


    Epoch 1/2:  88%|████████▊ | 30099/34343 [5:05:54<55:07,  1.28it/s]  

    Epoch 1 Batch 30100 loss: 2.1652884483337402


    Epoch 1/2:  88%|████████▊ | 30100/34343 [5:05:55<52:38,  1.34it/s]

    Closest words to the center word a: ['a', 'pluriform', 'automorphism', 'cardinality', 'every', 'morphism', 'otimes', 'monoid', 'another', 'newnode']


    Epoch 1/2:  88%|████████▊ | 30199/34343 [5:07:02<17:32,  3.94it/s]  

    Epoch 1 Batch 30200 loss: 2.124663829803467


    Epoch 1/2:  88%|████████▊ | 30200/34343 [5:07:03<25:40,  2.69it/s]

    Closest words to the center word record: ['record', 'playoffs', 'grammy', 'nfl', 'mvp', 'singles', 'batting', 'playoff', 'championship', 'billboard']


    Epoch 1/2:  88%|████████▊ | 30299/34343 [5:08:11<14:42,  4.58it/s]  

    Epoch 1 Batch 30300 loss: 2.1587517261505127


    Epoch 1/2:  88%|████████▊ | 30300/34343 [5:08:12<23:10,  2.91it/s]

    Closest words to the center word b: ['b', 'd', 'cyg', 'laureate', 'pngimage', 'footballer', 'physiologist', 'pmid', 'gwh', 'cdots']


    Epoch 1/2:  89%|████████▊ | 30399/34343 [5:09:35<5:00:57,  4.58s/it]

    Epoch 1 Batch 30400 loss: 2.1257636547088623


    Epoch 1/2:  89%|████████▊ | 30400/34343 [5:09:35<3:43:47,  3.41s/it]

    Closest words to the center word philo: ['yoannis', 'nevi', 'philo', 'patriarch', 'palaeologus', 'barnabas', 'epiphanius', 'ezekiel', 'isaiah', 'jesu']


    Epoch 1/2:  89%|████████▉ | 30499/34343 [5:10:44<38:26,  1.67it/s]  

    Epoch 1 Batch 30500 loss: 2.1307451725006104


    Epoch 1/2:  89%|████████▉ | 30500/34343 [5:10:45<41:19,  1.55it/s]

    Closest words to the center word began: ['began', 'continued', 'started', 'helped', 'went', 'came', 'attempted', 'saw', 'coincided', 'culminated']


    Epoch 1/2:  89%|████████▉ | 30599/34343 [5:11:52<14:43,  4.24it/s]  

    Epoch 1 Batch 30600 loss: 2.1571314334869385


    Epoch 1/2:  89%|████████▉ | 30600/34343 [5:11:53<22:46,  2.74it/s]

    Closest words to the center word field: ['field', 'magnetic', 'mathbf', 'dipole', 'topological', 'electromagnetic', 'vector', 'gravitational', 'automorphism', 'subfield']


    Epoch 1/2:  89%|████████▉ | 30699/34343 [5:13:00<13:04,  4.65it/s]  

    Epoch 1 Batch 30700 loss: 2.15496826171875


    Epoch 1/2:  89%|████████▉ | 30700/34343 [5:13:01<21:34,  2.81it/s]

    Closest words to the center word fiction: ['fiction', 'fantasy', 'horror', 'novels', 'dystopian', 'fandom', 'science', 'illustrators', 'anthologies', 'mythos']


    Epoch 1/2:  90%|████████▉ | 30799/34343 [5:14:25<3:27:37,  3.52s/it]

    Epoch 1 Batch 30800 loss: 2.1584253311157227


    Epoch 1/2:  90%|████████▉ | 30800/34343 [5:14:26<2:36:34,  2.65s/it]

    Closest words to the center word ii: ['ii', 'iii', 'iv', 'palaeologus', 'yoannis', 'vii', 'anastasius', 'comnenus', 'jumaada', 'viii']


    Epoch 1/2:  90%|████████▉ | 30899/34343 [5:15:33<23:17,  2.46it/s]  

    Epoch 1 Batch 30900 loss: 2.1477136611938477


    Epoch 1/2:  90%|████████▉ | 30900/34343 [5:15:34<27:38,  2.08it/s]

    Closest words to the center word held: ['held', 'elected', 'elections', 'vested', 'referendum', 'election', 'nfc', 'nominated', 'appointed', 'ecumenical']


    Epoch 1/2:  90%|█████████ | 30999/34343 [5:16:41<12:17,  4.53it/s]  

    Epoch 1 Batch 31000 loss: 2.171220302581787


    Epoch 1/2:  90%|█████████ | 31000/34343 [5:16:42<18:59,  2.93it/s]

    Closest words to the center word consisted: ['consisted', 'consists', 'pluriform', 'consisting', 'consist', 'dwt', 'comprised', 'regiments', 'battalions', 'composed']


    Epoch 1/2:  91%|█████████ | 31099/34343 [5:17:49<11:20,  4.76it/s]  

    Epoch 1 Batch 31100 loss: 2.148926258087158


    Epoch 1/2:  91%|█████████ | 31100/34343 [5:17:50<25:37,  2.11it/s]

    Closest words to the center word antonio: ['antonio', 'anh', 'cegep', 'universidad', 'bwv', 'footballer', 'icrm', 'eug', 'lez', 'rovere']


    Epoch 1/2:  91%|█████████ | 31199/34343 [5:19:16<1:36:34,  1.84s/it]

    Epoch 1 Batch 31200 loss: 2.142033338546753


    Epoch 1/2:  91%|█████████ | 31200/34343 [5:19:17<1:18:24,  1.50s/it]

    Closest words to the center word the: ['the', 'pluriform', 'ifrcs', 'vernal', 'nfc', 'newnode', 'ghats', 'outskirts', 'holocene', 'nabla']


    Epoch 1/2:  91%|█████████ | 31299/34343 [5:20:25<16:37,  3.05it/s]  

    Epoch 1 Batch 31300 loss: 2.155806064605713


    Epoch 1/2:  91%|█████████ | 31300/34343 [5:20:25<21:34,  2.35it/s]

    Closest words to the center word is: ['is', 'exists', 'bijective', 'satisfies', 'contains', 'refers', 'depends', 'serves', 'approximates', 'was']


    Epoch 1/2:  91%|█████████▏| 31399/34343 [5:21:33<10:29,  4.67it/s]  

    Epoch 1 Batch 31400 loss: 2.1606945991516113


    Epoch 1/2:  91%|█████████▏| 31400/34343 [5:21:34<16:30,  2.97it/s]

    Closest words to the center word camel: ['callithrix', 'equus', 'stenella', 'ferus', 'camel', 'ursus', 'mentha', 'tamarin', 'hamster', 'leontopithecus']


    Epoch 1/2:  92%|█████████▏| 31499/34343 [5:22:41<09:57,  4.76it/s]  

    Epoch 1 Batch 31500 loss: 2.0854337215423584


    Epoch 1/2:  92%|█████████▏| 31500/34343 [5:22:42<16:08,  2.94it/s]

    Closest words to the center word occurred: ['occurred', 'lasted', 'erupted', 'outbreak', 'eruption', 'lasts', 'resulted', 'happened', 'pleistocene', 'existed']


    Epoch 1/2:  92%|█████████▏| 31599/34343 [5:24:05<46:46,  1.02s/it]  

    Epoch 1 Batch 31600 loss: 2.163994073867798


    Epoch 1/2:  92%|█████████▏| 31600/34343 [5:24:06<41:22,  1.11it/s]

    Closest words to the center word for: ['for', 'ifrcs', 'ifad', 'newnode', 'iom', 'ifc', 'lossy', 'prev', 'lastnode', 'provide']


    Epoch 1/2:  92%|█████████▏| 31699/34343 [5:25:14<10:00,  4.40it/s]  

    Epoch 1 Batch 31700 loss: 2.14137601852417


    Epoch 1/2:  92%|█████████▏| 31700/34343 [5:25:15<15:09,  2.91it/s]

    Closest words to the center word form: ['form', 'adjoint', 'subgroup', 'groupoid', 'pluriform', 'bilinear', 'mathfrak', 'approximant', 'covalent', 'conjugate']


    Epoch 1/2:  93%|█████████▎| 31799/34343 [5:26:22<09:10,  4.62it/s]  

    Epoch 1 Batch 31800 loss: 2.1413497924804688


    Epoch 1/2:  93%|█████████▎| 31800/34343 [5:26:23<14:10,  2.99it/s]

    Closest words to the center word separate: ['separate', 'distinct', 'homomorphism', 'pluriform', 'different', 'functor', 'morphisms', 'homomorphisms', 'grouped', 'subnational']


    Epoch 1/2:  93%|█████████▎| 31899/34343 [5:27:30<08:50,  4.61it/s]  

    Epoch 1 Batch 31900 loss: 2.151773691177368


    Epoch 1/2:  93%|█████████▎| 31900/34343 [5:27:48<3:42:49,  5.47s/it]

    Closest words to the center word ideological: ['ideological', 'egalitarianism', 'leninist', 'authoritarianism', 'authoritarian', 'ideology', 'marxism', 'socio', 'laissez', 'leninism']


    Epoch 1/2:  93%|█████████▎| 31999/34343 [5:28:56<13:27,  2.90it/s]  

    Epoch 1 Batch 32000 loss: 2.17047119140625


    Epoch 1/2:  93%|█████████▎| 32000/34343 [5:28:57<16:36,  2.35it/s]

    Closest words to the center word their: ['their', 'its', 'your', 'our', 'her', 'his', 'respective', 'gesserit', 'themselves', 'my']


    Epoch 1/2:  93%|█████████▎| 32099/34343 [5:30:04<08:23,  4.46it/s]  

    Epoch 1 Batch 32100 loss: 2.138092041015625


    Epoch 1/2:  93%|█████████▎| 32100/34343 [5:30:05<13:01,  2.87it/s]

    Closest words to the center word regime: ['regime', 'bolsheviks', 'coup', 'gorbachev', 'reforms', 'pluriform', 'commissar', 'junta', 'communist', 'perestroika']


    Epoch 1/2:  94%|█████████▍| 32199/34343 [5:31:13<07:55,  4.51it/s]  

    Epoch 1 Batch 32200 loss: 2.1601781845092773


    Epoch 1/2:  94%|█████████▍| 32200/34343 [5:31:13<12:08,  2.94it/s]

    Closest words to the center word time: ['time', 'period', 'utc', 'distances', 'elapsed', 'moment', 'vernal', 'mhz', 'speeds', 'cardinality']


    Epoch 1/2:  94%|█████████▍| 32299/34343 [5:32:37<44:22,  1.30s/it]  

    Epoch 1 Batch 32300 loss: 2.141155242919922


    Epoch 1/2:  94%|█████████▍| 32300/34343 [5:32:37<37:25,  1.10s/it]

    Closest words to the center word he: ['he', 'she', 'clytemnestra', 'gehrig', 'aegisthus', 'remarried', 'bogart', 'tenji', 'ribbentrop', 'bacall']


    Epoch 1/2:  94%|█████████▍| 32399/34343 [5:33:46<09:52,  3.28it/s]  

    Epoch 1 Batch 32400 loss: 2.1982171535491943


    Epoch 1/2:  94%|█████████▍| 32400/34343 [5:33:47<12:53,  2.51it/s]

    Closest words to the center word secret: ['secret', 'wipo', 'unido', 'wmo', 'wtoo', 'wco', 'fbi', 'upu', 'wftu', 'defendant']


    Epoch 1/2:  95%|█████████▍| 32499/34343 [5:34:55<06:45,  4.55it/s]  

    Epoch 1 Batch 32500 loss: 2.177432060241699


    Epoch 1/2:  95%|█████████▍| 32500/34343 [5:34:56<10:21,  2.97it/s]

    Closest words to the center word traditionally: ['traditionally', 'widely', 'popularly', 'historically', 'interchangeably', 'commonly', 'ethnically', 'heretical', 'fundamentalist', 'universally']


    Epoch 1/2:  95%|█████████▍| 32599/34343 [5:36:02<06:10,  4.71it/s]  

    Epoch 1 Batch 32600 loss: 2.1100528240203857


    Epoch 1/2:  95%|█████████▍| 32600/34343 [5:36:03<09:31,  3.05it/s]

    Closest words to the center word immediately: ['immediately', 'shortly', 'resign', 'shuja', 'vowed', 'odrade', 'withdraw', 'aegisthus', 'clytemnestra', 'matres']


    Epoch 1/2:  95%|█████████▌| 32699/34343 [5:37:27<28:55,  1.06s/it]  

    Epoch 1 Batch 32700 loss: 2.1485962867736816


    Epoch 1/2:  95%|█████████▌| 32700/34343 [5:37:28<25:11,  1.09it/s]

    Closest words to the center word isbn: ['isbn', 'gwh', 'kwh', 'icrm', 'pp', 'ibrd', 'hbk', 'pmid', 'twh', 'jul']


    Epoch 1/2:  96%|█████████▌| 32799/34343 [5:38:36<07:21,  3.50it/s]  

    Epoch 1 Batch 32800 loss: 2.1603503227233887


    Epoch 1/2:  96%|█████████▌| 32800/34343 [5:38:36<09:55,  2.59it/s]

    Closest words to the center word cost: ['gwh', 'cost', 'expenditures', 'kwh', 'gdp', 'speeds', 'grt', 'bandwidth', 'tons', 'rate']


    Epoch 1/2:  96%|█████████▌| 32899/34343 [5:39:45<05:07,  4.69it/s]  

    Epoch 1 Batch 32900 loss: 2.1025540828704834


    Epoch 1/2:  96%|█████████▌| 32900/34343 [5:39:45<09:00,  2.67it/s]

    Closest words to the center word john: ['john', 'earl', 'cricketer', 'leiserson', 'george', 'cormen', 'yoannis', 'wadsworth', 'francis', 'viscount']


    Epoch 1/2:  96%|█████████▌| 32999/34343 [5:40:54<04:47,  4.68it/s]  

    Epoch 1 Batch 33000 loss: 2.1741445064544678


    Epoch 1/2:  96%|█████████▌| 33000/34343 [5:41:10<1:55:06,  5.14s/it]

    Closest words to the center word road: ['road', 'nfc', 'afc', 'commuter', 'oakland', 'piccadilly', 'ferry', 'playoffs', 'divisional', 'erie']


    Epoch 1/2:  96%|█████████▋| 33099/34343 [5:42:18<16:19,  1.27it/s]  

    Epoch 1 Batch 33100 loss: 2.1251473426818848


    Epoch 1/2:  96%|█████████▋| 33100/34343 [5:42:18<15:43,  1.32it/s]

    Closest words to the center word that: ['that', 'falsifiable', 'priori', 'empirically', 'testable', 'countably', 'divinely', 'verifiable', 'conclusive', 'hadn']


    Epoch 1/2:  97%|█████████▋| 33199/34343 [5:43:26<04:39,  4.09it/s]  

    Epoch 1 Batch 33200 loss: 2.127079486846924


    Epoch 1/2:  97%|█████████▋| 33200/34343 [5:43:26<07:11,  2.65it/s]

    Closest words to the center word lady: ['lady', 'boleyn', 'duchess', 'wife', 'granddaughter', 'niece', 'princess', 'daughter', 'heiress', 'aunt']


    Epoch 1/2:  97%|█████████▋| 33299/34343 [5:44:35<03:44,  4.65it/s]  

    Epoch 1 Batch 33300 loss: 2.1511149406433105


    Epoch 1/2:  97%|█████████▋| 33300/34343 [5:44:36<06:12,  2.80it/s]

    Closest words to the center word connected: ['connected', 'isomorphic', 'homeomorphic', 'automorphism', 'perpendicular', 'homomorphism', 'connecting', 'bijective', 'bounded', 'normed']


    Epoch 1/2:  97%|█████████▋| 33399/34343 [5:45:59<1:17:31,  4.93s/it]

    Epoch 1 Batch 33400 loss: 2.126316547393799


    Epoch 1/2:  97%|█████████▋| 33400/34343 [5:46:00<57:26,  3.66s/it]  

    Closest words to the center word spanish: ['spanish', 'portuguese', 'french', 'polynesia', 'blica', 'italian', 'argentine', 'austro', 'dutch', 'basque']


    Epoch 1/2:  98%|█████████▊| 33499/34343 [5:47:07<06:49,  2.06it/s]  

    Epoch 1 Batch 33500 loss: 2.1518990993499756


    Epoch 1/2:  98%|█████████▊| 33500/34343 [5:47:08<07:43,  1.82it/s]

    Closest words to the center word split: ['split', 'divided', 'pluriform', 'inducted', 'factions', 'merged', 'subdivided', 'nfc', 'afc', 'coalitions']


    Epoch 1/2:  98%|█████████▊| 33599/34343 [5:48:18<04:14,  2.93it/s]  

    Epoch 1 Batch 33600 loss: 2.1346943378448486


    Epoch 1/2:  98%|█████████▊| 33600/34343 [5:48:19<05:38,  2.19it/s]

    Closest words to the center word basal: ['phosphorylation', 'basal', 'citric', 'cvd', 'secreted', 'hormones', 'pyruvate', 'glycogen', 'anterior', 'clotting']


    Epoch 1/2:  98%|█████████▊| 33699/34343 [5:49:28<02:18,  4.67it/s]

    Epoch 1 Batch 33700 loss: 2.154817819595337


    Epoch 1/2:  98%|█████████▊| 33700/34343 [5:49:29<04:06,  2.60it/s]

    Closest words to the center word land: ['arable', 'land', 'irrigated', 'pastures', 'hydropower', 'km', 'sq', 'runways', 'crops', 'navigable']


    Epoch 1/2:  98%|█████████▊| 33799/34343 [5:50:52<22:53,  2.52s/it]

    Epoch 1 Batch 33800 loss: 2.142815589904785


    Epoch 1/2:  98%|█████████▊| 33800/34343 [5:50:53<17:53,  1.98s/it]

    Closest words to the center word african: ['african', 'saharan', 'chungcheong', 'asian', 'ifrcs', 'oau', 'wtoo', 'wto', 'oecs', 'afdb']


    Epoch 1/2:  99%|█████████▊| 33899/34343 [5:52:00<02:33,  2.90it/s]

    Epoch 1 Batch 33900 loss: 2.1455960273742676


    Epoch 1/2:  99%|█████████▊| 33900/34343 [5:52:01<03:13,  2.28it/s]

    Closest words to the center word chess: ['chess', 'championships', 'finalist', 'badminton', 'tennis', 'handball', 'championship', 'uefa', 'hockey', 'prix']


    Epoch 1/2:  99%|█████████▉| 33999/34343 [5:53:08<01:13,  4.68it/s]

    Epoch 1 Batch 34000 loss: 2.185729503631592


    Epoch 1/2:  99%|█████████▉| 34000/34343 [5:53:08<02:02,  2.80it/s]

    Closest words to the center word known: ['known', 'referred', 'regarded', 'agave', 'described', 'refered', 'classified', 'remembered', 'argumentum', 'understood']


    Epoch 1/2:  99%|█████████▉| 34099/34343 [5:54:18<00:58,  4.19it/s]

    Epoch 1 Batch 34100 loss: 2.1650309562683105


    Epoch 1/2:  99%|█████████▉| 34100/34343 [5:54:18<01:28,  2.76it/s]

    Closest words to the center word html: ['iom', 'ftp', 'pngimage', 'html', 'opcw', 'nonsignatory', 'http', 'webelements', 'icrm', 'www']


    Epoch 1/2: 100%|█████████▉| 34199/34343 [5:55:40<03:06,  1.30s/it]

    Epoch 1 Batch 34200 loss: 2.1328563690185547


    Epoch 1/2: 100%|█████████▉| 34200/34343 [5:55:41<02:38,  1.11s/it]

    Closest words to the center word thus: ['falsifiable', 'causal', 'thus', 'bijective', 'pluriform', 'priori', 'rationality', 'reactivity', 'integrable', 'axiom']


    Epoch 1/2: 100%|█████████▉| 34299/34343 [5:56:48<00:10,  4.28it/s]

    Epoch 1 Batch 34300 loss: 2.1766209602355957


    Epoch 1/2: 100%|█████████▉| 34300/34343 [5:56:49<00:15,  2.73it/s]

    Closest words to the center word animals: ['animals', 'organisms', 'multicellular', 'species', 'mammals', 'fungi', 'cassava', 'invertebrates', 'beets', 'shellfish']


    Epoch 1/2: 100%|██████████| 34343/34343 [5:57:16<00:00,  1.60it/s]


    Epoch 1 loss: 2.234515273133304


    Epoch 2/2:   0%|          | 99/34343 [01:23<9:13:17,  1.03it/s] 

    Epoch 2 Batch 100 loss: 2.1427221298217773


    Epoch 2/2:   0%|          | 100/34343 [01:24<8:27:31,  1.12it/s]

    Closest words to the center word people: ['people', 'refugees', 'albanians', 'births', 'persons', 'americans', 'politicians', 'nationality', 'immigrants', 'lgbt']


    Epoch 2/2:   1%|          | 199/34343 [02:31<2:27:51,  3.85it/s] 

    Epoch 2 Batch 200 loss: 2.13472843170166


    Epoch 2/2:   1%|          | 200/34343 [02:32<3:40:42,  2.58it/s]

    Closest words to the center word been: ['been', 'become', 'fallen', 'arisen', 'be', 'existed', 'harshly', 'recently', 'agave', 'householder']


    Epoch 2/2:   1%|          | 299/34343 [03:39<2:02:12,  4.64it/s] 

    Epoch 2 Batch 300 loss: 2.078275442123413


    Epoch 2/2:   1%|          | 300/34343 [03:40<3:24:26,  2.78it/s]

    Closest words to the center word governor: ['governor', 'governors', 'minister', 'lieutenant', 'bailiff', 'secretary', 'deputy', 'appointed', 'premiers', 'earl']


    Epoch 2/2:   1%|          | 399/34343 [04:47<1:59:58,  4.72it/s] 

    Epoch 2 Batch 400 loss: 2.158493757247925


    Epoch 2/2:   1%|          | 400/34343 [05:04<48:47:35,  5.18s/it]

    Closest words to the center word marry: ['marry', 'remarried', 'married', 'grandmother', 'clytemnestra', 'seduce', 'tenji', 'aegisthus', 'daughter', 'confess']


    Epoch 2/2:   1%|▏         | 499/34343 [06:13<5:49:12,  1.62it/s] 

    Epoch 2 Batch 500 loss: 2.12949538230896


    Epoch 2/2:   1%|▏         | 500/34343 [06:13<6:04:33,  1.55it/s]

    Closest words to the center word with: ['with', 'ifrcs', 'between', 'close', 'ifad', 'insubstantial', 'tamarin', 'icrm', 'ifc', 'positively']


    Epoch 2/2:   2%|▏         | 599/34343 [07:21<2:04:04,  4.53it/s] 

    Epoch 2 Batch 600 loss: 2.130528450012207


    Epoch 2/2:   2%|▏         | 600/34343 [07:22<3:16:58,  2.85it/s]

    Closest words to the center word bj: ['bj', 'finalist', 'wagoner', 'ulvaeus', 'discography', 'rgen', 'rk', 'composer', 'householder', 'elke']


    Epoch 2/2:   2%|▏         | 699/34343 [08:29<1:58:28,  4.73it/s] 

    Epoch 2 Batch 700 loss: 2.119719982147217


    Epoch 2/2:   2%|▏         | 700/34343 [08:30<3:14:13,  2.89it/s]

    Closest words to the center word lowlands: ['lowlands', 'subtropical', 'humid', 'semiarid', 'mountainous', 'uplands', 'cordillera', 'hilly', 'arid', 'broadleaf']


    Epoch 2/2:   2%|▏         | 799/34343 [09:58<33:12:40,  3.56s/it]

    Epoch 2 Batch 800 loss: 2.059316396713257


    Epoch 2/2:   2%|▏         | 800/34343 [09:58<25:10:51,  2.70s/it]

    Closest words to the center word simple: ['mathfrak', 'deterministic', 'bilinear', 'countably', 'bijective', 'adjoint', 'boolean', 'functor', 'newnode', 'convolution']


    Epoch 2/2:   3%|▎         | 899/34343 [11:07<2:34:24,  3.61it/s] 

    Epoch 2 Batch 900 loss: 2.1261239051818848


    Epoch 2/2:   3%|▎         | 900/34343 [11:07<3:34:47,  2.60it/s]

    Closest words to the center word robert: ['leiserson', 'rivest', 'cormen', 'elke', 'physiologist', 'laureate', 'robert', 'geneticist', 'wadsworth', 'archibald']


    Epoch 2/2:   3%|▎         | 999/34343 [12:15<2:07:05,  4.37it/s] 

    Epoch 2 Batch 1000 loss: 2.14149808883667


    Epoch 2/2:   3%|▎         | 1000/34343 [12:16<3:22:52,  2.74it/s]

    Closest words to the center word ascribed: ['ascribed', 'attributed', 'epistle', 'ordained', 'referred', 'anh', 'gnostic', 'apocryphal', 'authorship', 'epistles']


    Epoch 2/2:   3%|▎         | 1099/34343 [13:23<1:57:10,  4.73it/s] 

    Epoch 2 Batch 1100 loss: 2.1283271312713623


    Epoch 2/2:   3%|▎         | 1100/34343 [13:24<3:13:19,  2.87it/s]

    Closest words to the center word alabama: ['township', 'abet', 'devry', 'chungcheong', 'alabama', 'ssn', 'busan', 'bangor', 'missouri', 'county']


    Epoch 2/2:   3%|▎         | 1199/34343 [14:49<7:18:34,  1.26it/s] 

    Epoch 2 Batch 1200 loss: 2.108813762664795


    Epoch 2/2:   3%|▎         | 1200/34343 [14:50<6:59:21,  1.32it/s]

    Closest words to the center word divers: ['divers', 'diving', 'competitions', 'curling', 'restaurants', 'grt', 'bowls', 'golf', 'handball', 'skiing']


    Epoch 2/2:   4%|▍         | 1299/34343 [15:58<2:20:52,  3.91it/s] 

    Epoch 2 Batch 1300 loss: 2.11685848236084


    Epoch 2/2:   4%|▍         | 1300/34343 [15:59<3:27:35,  2.65it/s]

    Closest words to the center word denominational: ['denominational', 'denominations', 'fundamentalist', 'evangelicalism', 'haredi', 'churches', 'reconstructionist', 'masorti', 'hutterites', 'unido']


    Epoch 2/2:   4%|▍         | 1399/34343 [17:07<1:58:24,  4.64it/s] 

    Epoch 2 Batch 1400 loss: 2.136810779571533


    Epoch 2/2:   4%|▍         | 1400/34343 [17:08<4:46:10,  1.92it/s]

    Closest words to the center word due: ['due', 'owing', 'reactivity', 'induce', 'desertification', 'inversely', 'cloudy', 'hyperinsulinism', 'mortality', 'contributes']


    Epoch 2/2:   4%|▍         | 1499/34343 [18:31<46:13:55,  5.07s/it]

    Epoch 2 Batch 1500 loss: 2.115029811859131


    Epoch 2/2:   4%|▍         | 1500/34343 [18:32<34:14:56,  3.75s/it]

    Closest words to the center word establish: ['unido', 'establish', 'wco', 'unctad', 'upu', 'ifrcs', 'wftu', 'enact', 'unmibh', 'wcl']


    Epoch 2/2:   5%|▍         | 1599/34343 [19:39<5:19:40,  1.71it/s] 

    Epoch 2 Batch 1600 loss: 2.08738374710083


    Epoch 2/2:   5%|▍         | 1600/34343 [19:40<5:37:05,  1.62it/s]

    Closest words to the center word alliance: ['alliance', 'ifrcs', 'unido', 'unctad', 'wco', 'opcw', 'wftu', 'ifc', 'iom', 'frud']


    Epoch 2/2:   5%|▍         | 1699/34343 [20:48<2:19:00,  3.91it/s] 

    Epoch 2 Batch 1700 loss: 2.10058856010437


    Epoch 2/2:   5%|▍         | 1700/34343 [20:48<3:27:46,  2.62it/s]

    Closest words to the center word second: ['second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'ninth', 'rd', 'tenth', 'nd']


    Epoch 2/2:   5%|▌         | 1799/34343 [21:59<1:54:44,  4.73it/s] 

    Epoch 2 Batch 1800 loss: 2.1364266872406006


    Epoch 2/2:   5%|▌         | 1800/34343 [21:59<3:02:41,  2.97it/s]

    Closest words to the center word science: ['science', 'fiction', 'cognitivism', 'sciences', 'informatics', 'psychology', 'cognitive', 'engineering', 'epistemology', 'sociology']


    Epoch 2/2:   6%|▌         | 1899/34343 [23:23<29:52:31,  3.31s/it]

    Epoch 2 Batch 1900 loss: 2.1308605670928955


    Epoch 2/2:   6%|▌         | 1900/34343 [23:24<22:43:45,  2.52s/it]

    Closest words to the center word pragmatic: ['pragmatic', 'normative', 'egoism', 'falsifiable', 'cognitivism', 'libertarianism', 'agnosticism', 'rationality', 'consequentialism', 'relativism']


    Epoch 2/2:   6%|▌         | 1999/34343 [24:35<4:27:08,  2.02it/s] 

    Epoch 2 Batch 2000 loss: 2.123274564743042


    Epoch 2/2:   6%|▌         | 2000/34343 [24:35<4:57:27,  1.81it/s]

    Closest words to the center word also: ['also', 'wmo', 'disambiguation', 'wtoo', 'ifad', 'wftu', 'opcw', 'unido', 'ifrcs', 'wcl']


    Epoch 2/2:   6%|▌         | 2099/34343 [25:43<2:01:03,  4.44it/s] 

    Epoch 2 Batch 2100 loss: 2.0830719470977783


    Epoch 2/2:   6%|▌         | 2100/34343 [25:44<3:11:09,  2.81it/s]

    Closest words to the center word ingredients: ['sorghum', 'ingredients', 'spices', 'cassava', 'solvents', 'soybeans', 'beets', 'soy', 'alkaloids', 'dishes']


    Epoch 2/2:   6%|▋         | 2199/34343 [26:52<1:53:15,  4.73it/s] 

    Epoch 2 Batch 2200 loss: 2.109307289123535


    Epoch 2/2:   6%|▋         | 2200/34343 [26:52<3:03:44,  2.92it/s]

    Closest words to the center word york: ['york', 'giroux', 'schuster', 'knopf', 'ny', 'newark', 'scribner', 'berkley', 'schenectady', 'devry']


    Epoch 2/2:   7%|▋         | 2299/34343 [28:16<22:40:05,  2.55s/it]

    Epoch 2 Batch 2300 loss: 2.0878589153289795


    Epoch 2/2:   7%|▋         | 2300/34343 [28:17<17:35:18,  1.98s/it]

    Closest words to the center word france: ['france', 'belgium', 'portugal', 'spain', 'partement', 'luxembourg', 'netherlands', 'alsace', 'castile', 'nazaire']


    Epoch 2/2:   7%|▋         | 2399/34343 [29:23<3:03:09,  2.91it/s] 

    Epoch 2 Batch 2400 loss: 2.139266014099121


    Epoch 2/2:   7%|▋         | 2400/34343 [29:24<3:58:10,  2.24it/s]

    Closest words to the center word fire: ['fire', 'tank', 'projectiles', 'ammunition', 'bolt', 'dwt', 'combustion', 'projectile', 'torpedo', 'propelled']


    Epoch 2/2:   7%|▋         | 2499/34343 [30:33<1:56:40,  4.55it/s] 

    Epoch 2 Batch 2500 loss: 2.1110358238220215


    Epoch 2/2:   7%|▋         | 2500/34343 [30:33<3:04:56,  2.87it/s]

    Closest words to the center word a: ['a', 'pluriform', 'every', 'automorphism', 'any', 'another', 'monoid', 'agave', 'otimes', 'newnode']


    Epoch 2/2:   8%|▊         | 2599/34343 [31:42<1:50:59,  4.77it/s] 

    Epoch 2 Batch 2600 loss: 2.0861809253692627


    Epoch 2/2:   8%|▊         | 2600/34343 [31:43<3:06:06,  2.84it/s]

    Closest words to the center word or: ['or', 'ifrcs', 'grt', 'newnode', 'rajonas', 'chordata', 'insubstantial', 'conjugated', 'prev', 'ifad']


    Epoch 2/2:   8%|▊         | 2699/34343 [33:04<11:38:27,  1.32s/it]

    Epoch 2 Batch 2700 loss: 2.1319422721862793


    Epoch 2/2:   8%|▊         | 2700/34343 [33:05<9:59:22,  1.14s/it] 

    Closest words to the center word edited: ['edited', 'elke', 'hlich', 'rivest', 'leiserson', 'anh', 'routledge', 'pratchett', 'reprinted', 'cormen']


    Epoch 2/2:   8%|▊         | 2799/34343 [34:12<2:25:22,  3.62it/s] 

    Epoch 2 Batch 2800 loss: 2.1186113357543945


    Epoch 2/2:   8%|▊         | 2800/34343 [34:13<3:27:22,  2.54it/s]

    Closest words to the center word wealth: ['wealth', 'earnings', 'gdp', 'prosperity', 'glasnost', 'overgrazing', 'vitality', 'subsistence', 'income', 'urbanization']


    Epoch 2/2:   8%|▊         | 2899/34343 [35:20<1:50:49,  4.73it/s] 

    Epoch 2 Batch 2900 loss: 2.0913403034210205


    Epoch 2/2:   8%|▊         | 2900/34343 [35:20<3:00:28,  2.90it/s]

    Closest words to the center word has: ['has', 'had', 'have', 'ifrcs', 'hasn', 'pluriform', 'having', 'ifad', 'enjoys', 'possesses']


    Epoch 2/2:   9%|▊         | 2999/34343 [36:29<1:49:42,  4.76it/s] 

    Epoch 2 Batch 3000 loss: 2.0676462650299072


    Epoch 2/2:   9%|▊         | 3000/34343 [36:30<2:59:12,  2.91it/s]

    Closest words to the center word verb: ['verb', 'infinitive', 'verbs', 'perfective', 'participle', 'approximant', 'consonant', 'fricative', 'nouns', 'nominative']


    Epoch 2/2:   9%|▉         | 3099/34343 [37:55<6:49:00,  1.27it/s] 

    Epoch 2 Batch 3100 loss: 2.0898971557617188


    Epoch 2/2:   9%|▉         | 3100/34343 [37:55<6:28:44,  1.34it/s]

    Closest words to the center word into: ['into', 'agave', 'subdivided', 'onto', 'divided', 'diploid', 'across', 'northward', 'together', 'through']


    Epoch 2/2:   9%|▉         | 3199/34343 [39:03<1:56:54,  4.44it/s] 

    Epoch 2 Batch 3200 loss: 2.125943899154663


    Epoch 2/2:   9%|▉         | 3200/34343 [39:04<3:09:01,  2.75it/s]

    Closest words to the center word french: ['french', 'ois', 'fran', 'dutch', 'spanish', 'auguste', 'comte', 'belgian', 'portuguese', 'outre']


    Epoch 2/2:  10%|▉         | 3299/34343 [40:10<1:50:56,  4.66it/s] 

    Epoch 2 Batch 3300 loss: 2.0982842445373535


    Epoch 2/2:  10%|▉         | 3300/34343 [40:11<3:12:07,  2.69it/s]

    Closest words to the center word expected: ['expected', 'able', 'lifes', 'expectancy', 'estimated', 'grt', 'willing', 'calculate', 'homeomorphic', 'adjusted']


    Epoch 2/2:  10%|▉         | 3399/34343 [41:36<44:11:25,  5.14s/it]

    Epoch 2 Batch 3400 loss: 2.088789463043213


    Epoch 2/2:  10%|▉         | 3400/34343 [41:36<32:38:19,  3.80s/it]

    Closest words to the center word interventionist: ['interventionist', 'laissez', 'collectivism', 'pluriform', 'conservatism', 'totalitarian', 'faire', 'egoism', 'egalitarianism', 'authoritarianism']


    Epoch 2/2:  10%|█         | 3499/34343 [42:44<2:39:00,  3.23it/s] 

    Epoch 2 Batch 3500 loss: 2.083583354949951


    Epoch 2/2:  10%|█         | 3500/34343 [42:45<3:42:12,  2.31it/s]

    Closest words to the center word of: ['of', 'ifrcs', 'ifad', 'rajonas', 'icrm', 'ifc', 'wtoo', 'akan', 'wtro', 'wmo']


    Epoch 2/2:  10%|█         | 3599/34343 [43:53<1:52:29,  4.55it/s] 

    Epoch 2 Batch 3600 loss: 2.1149137020111084


    Epoch 2/2:  10%|█         | 3600/34343 [43:54<3:05:02,  2.77it/s]

    Closest words to the center word sugarcubes: ['allman', 'wagoner', 'sugarcubes', 'rockabilly', 'django', 'vocals', 'albums', 'beatles', 'singles', 'everly']


    Epoch 2/2:  11%|█         | 3699/34343 [45:02<1:51:27,  4.58it/s] 

    Epoch 2 Batch 3700 loss: 2.111382007598877


    Epoch 2/2:  11%|█         | 3700/34343 [45:03<3:07:22,  2.73it/s]

    Closest words to the center word mg: ['gwh', 'grt', 'twh', 'mg', 'cooh', 'kwh', 'pmid', 'kcal', 'dwt', 'pngimage']


    Epoch 2/2:  11%|█         | 3799/34343 [46:27<8:47:37,  1.04s/it] 

    Epoch 2 Batch 3800 loss: 2.096947431564331


    Epoch 2/2:  11%|█         | 3800/34343 [46:27<7:52:46,  1.08it/s]

    Closest words to the center word social: ['social', 'pluriform', 'socio', 'egalitarianism', 'marxian', 'conservatism', 'welfare', 'libertarian', 'libertarianism', 'collectivism']


    Epoch 2/2:  11%|█▏        | 3899/34343 [47:35<2:21:06,  3.60it/s] 

    Epoch 2 Batch 3900 loss: 2.1126112937927246


    Epoch 2/2:  11%|█▏        | 3900/34343 [47:36<3:24:41,  2.48it/s]

    Closest words to the center word massive: ['massive', 'pluriform', 'rapid', 'overgrazing', 'huge', 'shortage', 'sudden', 'catastrophic', 'recession', 'drought']


    Epoch 2/2:  12%|█▏        | 3999/34343 [48:43<1:48:51,  4.65it/s] 

    Epoch 2 Batch 4000 loss: 2.1012675762176514


    Epoch 2/2:  12%|█▏        | 4000/34343 [48:44<3:03:07,  2.76it/s]

    Closest words to the center word generally: ['generally', 'widely', 'commonly', 'universally', 'often', 'usually', 'hotly', 'broadly', 'inherently', 'humid']


    Epoch 2/2:  12%|█▏        | 4099/34343 [49:53<1:45:54,  4.76it/s] 

    Epoch 2 Batch 4100 loss: 2.126585006713867


    Epoch 2/2:  12%|█▏        | 4100/34343 [50:09<42:11:39,  5.02s/it]

    Closest words to the center word signed: ['signed', 'ratified', 'treaty', 'wetlands', 'runways', 'ratification', 'agreement', 'agreements', 'signing', 'accords']


    Epoch 2/2:  12%|█▏        | 4199/34343 [51:18<6:24:49,  1.31it/s] 

    Epoch 2 Batch 4200 loss: 2.1400294303894043


    Epoch 2/2:  12%|█▏        | 4200/34343 [51:19<6:10:36,  1.36it/s]

    Closest words to the center word by: ['by', 'ifrcs', 'icrm', 'ifc', 'agave', 'rajonas', 'unido', 'leiserson', 'ifad', 'iom']


    Epoch 2/2:  13%|█▎        | 4299/34343 [52:27<2:07:40,  3.92it/s] 

    Epoch 2 Batch 4300 loss: 2.1312642097473145


    Epoch 2/2:  13%|█▎        | 4300/34343 [52:28<3:12:47,  2.60it/s]

    Closest words to the center word most: ['most', 'lifes', 'poorest', 'earliest', 'highly', 'extremely', 'more', 'largest', 'greatest', 'less']


    Epoch 2/2:  13%|█▎        | 4399/34343 [53:37<1:47:25,  4.65it/s] 

    Epoch 2 Batch 4400 loss: 2.139698028564453


    Epoch 2/2:  13%|█▎        | 4400/34343 [53:38<2:55:23,  2.85it/s]

    Closest words to the center word is: ['is', 'satisfies', 'bijective', 'refers', 'consists', 'was', 'behaves', 'contains', 'corresponds', 'exists']


    Epoch 2/2:  13%|█▎        | 4499/34343 [55:03<40:19:51,  4.87s/it]

    Epoch 2 Batch 4500 loss: 2.091897964477539


    Epoch 2/2:  13%|█▎        | 4500/34343 [55:03<29:56:05,  3.61s/it]

    Closest words to the center word chosen: ['chosen', 'elected', 'appointed', 'ordained', 'vested', 'anointed', 'sworn', 'overridden', 'nominated', 'proportional']


    Epoch 2/2:  13%|█▎        | 4599/34343 [56:12<5:04:27,  1.63it/s] 

    Epoch 2 Batch 4600 loss: 2.1326465606689453


    Epoch 2/2:  13%|█▎        | 4600/34343 [56:12<5:06:40,  1.62it/s]

    Closest words to the center word praise: ['praise', 'everlasting', 'preach', 'forgiveness', 'almighty', 'nomination', 'myself', 'thee', 'amnon', 'sins']


    Epoch 2/2:  14%|█▎        | 4699/34343 [57:22<1:56:12,  4.25it/s] 

    Epoch 2 Batch 4700 loss: 2.07889461517334


    Epoch 2/2:  14%|█▎        | 4700/34343 [57:23<2:54:00,  2.84it/s]

    Closest words to the center word the: ['the', 'ifrcs', 'pluriform', 'icrm', 'nonsignatory', 'intelsat', 'nfc', 'ifc', 'ladoga', 'vernal']


    Epoch 2/2:  14%|█▍        | 4799/34343 [58:31<1:44:50,  4.70it/s] 

    Epoch 2 Batch 4800 loss: 2.122645616531372


    Epoch 2/2:  14%|█▍        | 4800/34343 [58:32<2:45:25,  2.98it/s]

    Closest words to the center word earliest: ['earliest', 'oldest', 'masoretic', 'finest', 'cretaceous', 'cambrian', 'hellenistic', 'antiquity', 'prehistoric', 'carboniferous']


    Epoch 2/2:  14%|█▍        | 4899/34343 [59:55<28:35:38,  3.50s/it]

    Epoch 2 Batch 4900 loss: 2.0987653732299805


    Epoch 2/2:  14%|█▍        | 4900/34343 [59:55<21:32:21,  2.63s/it]

    Closest words to the center word after: ['after', 'before', 'lasted', 'afterwards', 'shortly', 'thereafter', 'months', 'during', 'weeks', 'afterward']


    Epoch 2/2:  15%|█▍        | 4999/34343 [1:01:05<3:23:53,  2.40it/s] 

    Epoch 2 Batch 5000 loss: 2.0908498764038086


    Epoch 2/2:  15%|█▍        | 5000/34343 [1:01:05<3:55:48,  2.07it/s]

    Closest words to the center word australia: ['australia', 'tasmania', 'chungcheong', 'zealand', 'busan', 'canada', 'premiers', 'kitts', 'uruguay', 'queensland']


    Epoch 2/2:  15%|█▍        | 5099/34343 [1:02:13<1:47:56,  4.52it/s] 

    Epoch 2 Batch 5100 loss: 2.123389720916748


    Epoch 2/2:  15%|█▍        | 5100/34343 [1:02:14<2:49:46,  2.87it/s]

    Closest words to the center word early: ['early', 'late', 'mid', 'kwh', 'twh', 'gwh', 'availabilitymales', 'cretaceous', 'nineteenth', 'eighteenth']


    Epoch 2/2:  15%|█▌        | 5199/34343 [1:03:22<1:44:56,  4.63it/s] 

    Epoch 2 Batch 5200 loss: 2.087289333343506


    Epoch 2/2:  15%|█▌        | 5200/34343 [1:03:23<2:47:42,  2.90it/s]

    Closest words to the center word le: ['le', 'cegep', 'linguistique', 'estudios', 'serpento', 'quijote', 'dicis', 'histoire', 'esas', 'sur']


    Epoch 2/2:  15%|█▌        | 5299/34343 [1:04:47<15:04:07,  1.87s/it]

    Epoch 2 Batch 5300 loss: 2.09371018409729


    Epoch 2/2:  15%|█▌        | 5300/34343 [1:04:48<12:06:57,  1.50s/it]

    Closest words to the center word residing: ['residing', 'grt', 'ukrainians', 'populous', 'villages', 'wealthiest', 'sq', 'emigrated', 'bosniaks', 'poorest']


    Epoch 2/2:  16%|█▌        | 5399/34343 [1:05:55<2:23:08,  3.37it/s] 

    Epoch 2 Batch 5400 loss: 2.153334856033325


    Epoch 2/2:  16%|█▌        | 5400/34343 [1:05:55<3:10:54,  2.53it/s]

    Closest words to the center word thus: ['falsifiable', 'thus', 'integrable', 'countably', 'bijective', 'infty', 'adjoint', 'enthalpy', 'causal', 'cardinality']


    Epoch 2/2:  16%|█▌        | 5499/34343 [1:07:05<1:44:24,  4.60it/s] 

    Epoch 2 Batch 5500 loss: 2.08534574508667


    Epoch 2/2:  16%|█▌        | 5500/34343 [1:07:06<2:40:43,  2.99it/s]

    Closest words to the center word invention: ['invention', 'advent', 'inventor', 'microcomputer', 'ipf', 'synthesizers', 'combustion', 'luminiferous', 'synthesizer', 'introduction']


    Epoch 2/2:  16%|█▋        | 5599/34343 [1:08:13<1:44:56,  4.57it/s] 

    Epoch 2 Batch 5600 loss: 2.1243724822998047


    Epoch 2/2:  16%|█▋        | 5600/34343 [1:08:14<2:48:03,  2.85it/s]

    Closest words to the center word charles: ['charles', 'leiserson', 'cormen', 'earl', 'duchess', 'louis', 'ois', 'emmanuel', 'valois', 'henry']


    Epoch 2/2:  17%|█▋        | 5699/34343 [1:09:39<8:10:31,  1.03s/it] 

    Epoch 2 Batch 5700 loss: 2.0910654067993164


    Epoch 2/2:  17%|█▋        | 5700/34343 [1:09:39<7:33:53,  1.05it/s]

    Closest words to the center word terre: ['icrm', 'universidade', 'rajonas', 'ribeira', 'ouest', 'partement', 'noma', 'terre', 'classis', 'cegep']


    Epoch 2/2:  17%|█▋        | 5799/34343 [1:10:48<1:48:30,  4.38it/s] 

    Epoch 2 Batch 5800 loss: 2.112058162689209


    Epoch 2/2:  17%|█▋        | 5800/34343 [1:10:48<2:52:03,  2.76it/s]

    Closest words to the center word church: ['church', 'episcopal', 'churches', 'orthodox', 'communion', 'catholic', 'methodist', 'lutheran', 'anglican', 'anglicanism']


    Epoch 2/2:  17%|█▋        | 5899/34343 [1:11:57<1:40:13,  4.73it/s] 

    Epoch 2 Batch 5900 loss: 2.1148324012756348


    Epoch 2/2:  17%|█▋        | 5900/34343 [1:11:58<2:45:46,  2.86it/s]

    Closest words to the center word preferably: ['beets', 'cooked', 'stewed', 'waterborne', 'boiled', 'soy', 'grilled', 'dried', 'fried', 'cassava']


    Epoch 2/2:  17%|█▋        | 5999/34343 [1:13:05<1:40:52,  4.68it/s] 

    Epoch 2 Batch 6000 loss: 2.1314735412597656


    Epoch 2/2:  17%|█▋        | 6000/34343 [1:13:20<38:13:06,  4.85s/it]

    Closest words to the center word led: ['led', 'unido', 'contributed', 'ifrcs', 'resulted', 'helped', 'caused', 'unctad', 'attempted', 'prompted']


    Epoch 2/2:  18%|█▊        | 6099/34343 [1:14:28<2:40:24,  2.93it/s] 

    Epoch 2 Batch 6100 loss: 2.0925450325012207


    Epoch 2/2:  18%|█▊        | 6100/34343 [1:14:29<3:28:39,  2.26it/s]

    Closest words to the center word has: ['has', 'have', 'had', 'ifrcs', 'pluriform', 'hasn', 'having', 'hadn', 'enjoys', 'possesses']


    Epoch 2/2:  18%|█▊        | 6199/34343 [1:15:39<1:44:02,  4.51it/s] 

    Epoch 2 Batch 6200 loss: 2.115598440170288


    Epoch 2/2:  18%|█▊        | 6200/34343 [1:15:40<2:46:35,  2.82it/s]

    Closest words to the center word gravitational: ['gravitational', 'dipole', 'electromagnetic', 'mathbf', 'acceleration', 'velocity', 'rotational', 'inertial', 'angular', 'spacetime']


    Epoch 2/2:  18%|█▊        | 6299/34343 [1:16:48<1:39:59,  4.67it/s] 

    Epoch 2 Batch 6300 loss: 2.150808572769165


    Epoch 2/2:  18%|█▊        | 6300/34343 [1:16:49<2:46:19,  2.81it/s]

    Closest words to the center word rating: ['rating', 'gwh', 'grt', 'pluriform', 'kwh', 'twh', 'expenditures', 'mbit', 'dwt', 'mjs']


    Epoch 2/2:  19%|█▊        | 6399/34343 [1:18:13<10:37:16,  1.37s/it]

    Epoch 2 Batch 6400 loss: 2.1091442108154297


    Epoch 2/2:  19%|█▊        | 6400/34343 [1:18:14<9:02:21,  1.16s/it] 

    Closest words to the center word broadcast: ['broadcast', 'intelsat', 'shortwave', 'isps', 'intersputnik', 'stations', 'inmarsat', 'radio', 'aired', 'eutelsat']


    Epoch 2/2:  19%|█▉        | 6499/34343 [1:19:21<2:18:38,  3.35it/s] 

    Epoch 2 Batch 6500 loss: 2.0898923873901367


    Epoch 2/2:  19%|█▉        | 6500/34343 [1:19:22<3:10:13,  2.44it/s]

    Closest words to the center word forces: ['forces', 'troops', 'army', 'gendarmerie', 'marshal', 'armies', 'peacekeeping', 'armed', 'squadrons', 'force']


    Epoch 2/2:  19%|█▉        | 6599/34343 [1:20:29<1:40:31,  4.60it/s] 

    Epoch 2 Batch 6600 loss: 2.1494300365448


    Epoch 2/2:  19%|█▉        | 6600/34343 [1:20:29<2:41:06,  2.87it/s]

    Closest words to the center word done: ['done', 'insubstantial', 'import', 'duplicate', 'irrelevent', 'subnet', 'nonsignatory', 'redirect', 'jargon', 'iom']


    Epoch 2/2:  20%|█▉        | 6699/34343 [1:21:40<1:38:13,  4.69it/s] 

    Epoch 2 Batch 6700 loss: 2.068402051925659


    Epoch 2/2:  20%|█▉        | 6700/34343 [1:21:40<2:43:11,  2.82it/s]

    Closest words to the center word poverty: ['poverty', 'expectancy', 'est', 'income', 'unemployment', 'median', 'expenditures', 'mortality', 'population', 'wages']


    Epoch 2/2:  20%|█▉        | 6799/34343 [1:23:04<7:46:08,  1.02s/it] 

    Epoch 2 Batch 6800 loss: 2.1448311805725098


    Epoch 2/2:  20%|█▉        | 6800/34343 [1:23:05<6:59:13,  1.10it/s]

    Closest words to the center word ahmet: ['ebne', 'footballer', 'grt', 'kwh', 'abol', 'gwh', 'rovere', 'maaouya', 'violinist', 'finalist']


    Epoch 2/2:  20%|██        | 6899/34343 [1:24:15<2:09:37,  3.53it/s] 

    Epoch 2 Batch 6900 loss: 2.1025197505950928


    Epoch 2/2:  20%|██        | 6900/34343 [1:24:15<2:58:44,  2.56it/s]

    Closest words to the center word british: ['british', 'canadian', 'scottish', 'australian', 'american', 'naturalized', 'autodidacts', 'french', 'dutch', 'portuguese']


    Epoch 2/2:  20%|██        | 6999/34343 [1:25:23<1:36:58,  4.70it/s] 

    Epoch 2 Batch 7000 loss: 2.095928430557251


    Epoch 2/2:  20%|██        | 7000/34343 [1:25:23<2:33:05,  2.98it/s]

    Closest words to the center word america: ['america', 'chungcheong', 'mycological', 'busan', 'gyeonggi', 'nfc', 'gyeongsang', 'africa', 'saharan', 'caribbean']


    Epoch 2/2:  21%|██        | 7099/34343 [1:26:31<1:36:07,  4.72it/s] 

    Epoch 2 Batch 7100 loss: 2.082852602005005


    Epoch 2/2:  21%|██        | 7100/34343 [1:26:47<36:38:03,  4.84s/it]

    Closest words to the center word various: ['various', 'several', 'numerous', 'different', 'variety', 'differing', 'varying', 'multiple', 'many', 'ifad']


    Epoch 2/2:  21%|██        | 7199/34343 [1:27:54<5:37:45,  1.34it/s] 

    Epoch 2 Batch 7200 loss: 2.112398862838745


    Epoch 2/2:  21%|██        | 7200/34343 [1:27:55<5:23:27,  1.40it/s]

    Closest words to the center word liu: ['unido', 'liu', 'unctad', 'wco', 'wftu', 'unmibh', 'opcw', 'ifrcs', 'kapoor', 'shek']


    Epoch 2/2:  21%|██▏       | 7299/34343 [1:29:04<1:53:12,  3.98it/s] 

    Epoch 2 Batch 7300 loss: 2.092010021209717


    Epoch 2/2:  21%|██▏       | 7300/34343 [1:29:04<2:51:32,  2.63it/s]

    Closest words to the center word surgeon: ['surgeon', 'chemist', 'unido', 'physicist', 'philanthropist', 'physician', 'physiologist', 'zoologist', 'psychiatrist', 'mineralogist']


    Epoch 2/2:  22%|██▏       | 7399/34343 [1:30:13<1:35:07,  4.72it/s] 

    Epoch 2 Batch 7400 loss: 2.088667392730713


    Epoch 2/2:  22%|██▏       | 7400/34343 [1:30:14<2:41:24,  2.78it/s]

    Closest words to the center word one: ['gwh', 'kwh', 'twh', 'grt', 'hbk', 'pngimage', 'sfg', 'cyg', 'pmid', 'jul']


    Epoch 2/2:  22%|██▏       | 7499/34343 [1:31:38<36:02:00,  4.83s/it]

    Epoch 2 Batch 7500 loss: 2.1404001712799072


    Epoch 2/2:  22%|██▏       | 7500/34343 [1:31:39<26:51:55,  3.60s/it]

    Closest words to the center word wonderful: ['wonderful', 'stevie', 'austen', 'frankenstein', 'poirot', 'swaim', 'sherlock', 'marple', 'diddley', 'loves']


    Epoch 2/2:  22%|██▏       | 7599/34343 [1:32:46<3:28:59,  2.13it/s] 

    Epoch 2 Batch 7600 loss: 2.103727340698242


    Epoch 2/2:  22%|██▏       | 7600/34343 [1:32:47<3:49:00,  1.95it/s]

    Closest words to the center word them: ['them', 'findable', 'matres', 'him', 'prev', 'thy', 'confess', 'preach', 'oneself', 'baptize']


    Epoch 2/2:  22%|██▏       | 7699/34343 [1:33:56<1:41:04,  4.39it/s] 

    Epoch 2 Batch 7700 loss: 2.111868381500244


    Epoch 2/2:  22%|██▏       | 7700/34343 [1:33:57<2:44:57,  2.69it/s]

    Closest words to the center word which: ['which', 'cytosol', 'countably', 'ifrcs', 'endothermic', 'insoluble', 'enthalpy', 'symplectic', 'circularly', 'gangetic']


    Epoch 2/2:  23%|██▎       | 7799/34343 [1:35:05<1:35:31,  4.63it/s] 

    Epoch 2 Batch 7800 loss: 2.120771884918213


    Epoch 2/2:  23%|██▎       | 7800/34343 [1:35:06<2:30:37,  2.94it/s]

    Closest words to the center word latex: ['latex', 'findable', 'iom', 'cassava', 'soy', 'lossy', 'codecs', 'ifad', 'scsi', 'ipf']


    Epoch 2/2:  23%|██▎       | 7899/34343 [1:36:31<19:07:55,  2.60s/it]

    Epoch 2 Batch 7900 loss: 2.0719943046569824


    Epoch 2/2:  23%|██▎       | 7900/34343 [1:36:31<15:06:32,  2.06s/it]

    Closest words to the center word mongoloids: ['mongoloids', 'elevations', 'lifes', 'incomes', 'humid', 'reactivity', 'rainfall', 'humidity', 'differentiable', 'estimators']


    Epoch 2/2:  23%|██▎       | 7999/34343 [1:37:39<2:32:21,  2.88it/s] 

    Epoch 2 Batch 8000 loss: 2.151247024536133


    Epoch 2/2:  23%|██▎       | 8000/34343 [1:37:40<3:17:14,  2.23it/s]

    Closest words to the center word even: ['even', 'lifes', 'aesthetically', 'cheaper', 'weren', 'quicker', 'forgiving', 'falsifiable', 'worse', 'computationally']


    Epoch 2/2:  24%|██▎       | 8099/34343 [1:38:47<1:33:29,  4.68it/s] 

    Epoch 2 Batch 8100 loss: 2.1032638549804688


    Epoch 2/2:  24%|██▎       | 8100/34343 [1:38:48<2:33:56,  2.84it/s]

    Closest words to the center word standard: ['standard', 'iec', 'cccc', 'bbbb', 'ansi', 'tiberian', 'iso', 'utf', 'phonetic', 'standards']


    Epoch 2/2:  24%|██▍       | 8199/34343 [1:39:56<1:31:29,  4.76it/s] 

    Epoch 2 Batch 8200 loss: 2.1376395225524902


    Epoch 2/2:  24%|██▍       | 8200/34343 [1:39:57<2:33:47,  2.83it/s]

    Closest words to the center word il: ['il', 'bwv', 'anh', 'ebne', 'linguistique', 'stosunku', 'cegep', 'landsmannschaft', 'leiserson', 'studi']


    Epoch 2/2:  24%|██▍       | 8299/34343 [1:41:20<9:20:42,  1.29s/it] 

    Epoch 2 Batch 8300 loss: 2.1120400428771973


    Epoch 2/2:  24%|██▍       | 8300/34343 [1:41:21<8:02:19,  1.11s/it]

    Closest words to the center word headache: ['headache', 'nausea', 'dizziness', 'diarrhea', 'vectorborne', 'vomiting', 'congenital', 'inhibitors', 'symptoms', 'hypothyroidism']


    Epoch 2/2:  24%|██▍       | 8399/34343 [1:42:42<3:56:36,  1.83it/s] 

    Epoch 2 Batch 8400 loss: 2.1168150901794434


    Epoch 2/2:  24%|██▍       | 8400/34343 [1:42:42<4:34:22,  1.58it/s]

    Closest words to the center word aforementioned: ['aforementioned', 'privy', 'ecumenical', 'mormon', 'epistle', 'immaculate', 'curia', 'nevi', 'cluetrain', 'justices']


    Epoch 2/2:  25%|██▍       | 8499/34343 [1:44:04<1:45:12,  4.09it/s] 

    Epoch 2 Batch 8500 loss: 2.105393409729004


    Epoch 2/2:  25%|██▍       | 8500/34343 [1:44:05<2:48:06,  2.56it/s]

    Closest words to the center word coal: ['coal', 'bauxite', 'hydropower', 'potash', 'fuels', 'kwh', 'petroleum', 'zinc', 'beets', 'gwh']


    Epoch 2/2:  25%|██▌       | 8599/34343 [1:45:23<1:33:03,  4.61it/s] 

    Epoch 2 Batch 8600 loss: 2.0791122913360596


    Epoch 2/2:  25%|██▌       | 8600/34343 [1:45:24<2:43:14,  2.63it/s]

    Closest words to the center word graphics: ['graphics', 'raster', 'ipf', 'opengl', 'lossy', 'bitmap', 'usb', 'lossless', 'codecs', 'programmable']


    Epoch 2/2:  25%|██▌       | 8699/34343 [1:46:51<2:55:29,  2.44it/s] 

    Epoch 2 Batch 8700 loss: 2.095363140106201


    Epoch 2/2:  25%|██▌       | 8700/34343 [1:46:51<3:23:12,  2.10it/s]

    Closest words to the center word if: ['if', 'prev', 'bijective', 'newnode', 'forall', 'refutable', 'otimes', 'bijection', 'countably', 'ldots']


    Epoch 2/2:  26%|██▌       | 8799/34343 [1:48:03<1:39:47,  4.27it/s] 

    Epoch 2 Batch 8800 loss: 2.0790231227874756


    Epoch 2/2:  26%|██▌       | 8800/34343 [1:48:03<2:29:34,  2.85it/s]

    Closest words to the center word j: ['leiserson', 'elke', 'rivest', 'polskiej', 'cormen', 'stosunku', 'otimes', 'leq', 'landsmannschaft', 'hlich']


    Epoch 2/2:  26%|██▌       | 8899/34343 [1:49:12<1:29:55,  4.72it/s] 

    Epoch 2 Batch 8900 loss: 2.159534215927124


    Epoch 2/2:  26%|██▌       | 8900/34343 [1:49:13<2:21:50,  2.99it/s]

    Closest words to the center word it: ['it', 'findable', 'prev', 'bijective', 'testable', 'falsifiable', 'empirically', 'metrizable', 'refutable', 'unwise']


    Epoch 2/2:  26%|██▌       | 8999/34343 [1:50:38<12:44:57,  1.81s/it]

    Epoch 2 Batch 9000 loss: 2.1130452156066895


    Epoch 2/2:  26%|██▌       | 9000/34343 [1:50:39<10:16:01,  1.46s/it]

    Closest words to the center word see: ['see', 'disambiguation', 'yird', 'redirects', 'list', 'newnode', 'topics', 'terrier', 'wtoo', 'wtro']


    Epoch 2/2:  26%|██▋       | 9099/34343 [1:51:48<2:24:48,  2.91it/s] 

    Epoch 2 Batch 9100 loss: 2.0795629024505615


    Epoch 2/2:  26%|██▋       | 9100/34343 [1:51:49<3:02:15,  2.31it/s]

    Closest words to the center word final: ['final', 'finals', 'nfc', 'playoff', 'championship', 'divisional', 'fantasy', 'afc', 'champions', 'redskins']


    Epoch 2/2:  27%|██▋       | 9199/34343 [1:53:00<1:34:52,  4.42it/s] 

    Epoch 2 Batch 9200 loss: 2.129692554473877


    Epoch 2/2:  27%|██▋       | 9200/34343 [1:53:00<2:23:57,  2.91it/s]

    Closest words to the center word a: ['a', 'pluriform', 'groupoid', 'prev', 'automorphism', 'bijective', 'newnode', 'descriptor', 'agave', 'lastnode']


    Epoch 2/2:  27%|██▋       | 9299/34343 [1:54:11<1:27:59,  4.74it/s] 

    Epoch 2 Batch 9300 loss: 2.0988025665283203


    Epoch 2/2:  27%|██▋       | 9300/34343 [1:54:11<2:33:43,  2.72it/s]

    Closest words to the center word late: ['late', 'mid', 'kwh', 'gwh', 'nfc', 'early', 'nineteenth', 'twh', 'seventeenth', 'afc']


    Epoch 2/2:  27%|██▋       | 9399/34343 [1:55:35<8:56:45,  1.29s/it] 

    Epoch 2 Batch 9400 loss: 2.0639731884002686


    Epoch 2/2:  27%|██▋       | 9400/34343 [1:55:36<7:35:19,  1.10s/it]

    Closest words to the center word placed: ['placed', 'worn', 'interred', 'inscribed', 'buried', 'inserted', 'touches', 'mounted', 'located', 'stacked']


    Epoch 2/2:  28%|██▊       | 9499/34343 [1:56:43<1:55:07,  3.60it/s] 

    Epoch 2 Batch 9500 loss: 2.1140642166137695


    Epoch 2/2:  28%|██▊       | 9500/34343 [1:56:44<2:42:05,  2.55it/s]

    Closest words to the center word success: ['success', 'acclaim', 'popularity', 'airplay', 'downturn', 'successes', 'garnered', 'notoriety', 'comeback', 'postseason']


    Epoch 2/2:  28%|██▊       | 9599/34343 [1:57:53<1:28:22,  4.67it/s] 

    Epoch 2 Batch 9600 loss: 2.0933616161346436


    Epoch 2/2:  28%|██▊       | 9600/34343 [1:57:54<2:18:16,  2.98it/s]

    Closest words to the center word divorced: ['divorced', 'remarried', 'married', 'caesaris', 'wed', 'householder', 'fathered', 'raped', 'agrippina', 'antonia']


    Epoch 2/2:  28%|██▊       | 9699/34343 [1:59:02<1:30:31,  4.54it/s] 

    Epoch 2 Batch 9700 loss: 2.0646674633026123


    Epoch 2/2:  28%|██▊       | 9700/34343 [1:59:02<2:24:05,  2.85it/s]

    Closest words to the center word opponents: ['opponents', 'masculists', 'ideologies', 'collectivism', 'oppose', 'capitalists', 'supporters', 'detractors', 'unwillingness', 'egalitarianism']


    Epoch 2/2:  29%|██▊       | 9799/34343 [2:00:25<5:09:20,  1.32it/s] 

    Epoch 2 Batch 9800 loss: 2.098212242126465


    Epoch 2/2:  29%|██▊       | 9800/34343 [2:00:26<4:54:13,  1.39it/s]

    Closest words to the center word erasmus: ['yoannis', 'erasmus', 'stwertka', 'palaeologus', 'mamre', 'judaica', 'dramatists', 'nevi', 'duchess', 'friedrich']


    Epoch 2/2:  29%|██▉       | 9899/34343 [2:01:36<1:40:16,  4.06it/s] 

    Epoch 2 Batch 9900 loss: 2.0927066802978516


    Epoch 2/2:  29%|██▉       | 9900/34343 [2:01:36<2:26:44,  2.78it/s]

    Closest words to the center word replicating: ['replicating', 'eukaryotic', 'oxidizing', 'polymers', 'adjoint', 'deterministic', 'probabilistic', 'glycogen', 'multicellular', 'positron']


    Epoch 2/2:  29%|██▉       | 9999/34343 [2:02:45<1:26:33,  4.69it/s] 

    Epoch 2 Batch 10000 loss: 2.116197347640991


    Epoch 2/2:  29%|██▉       | 10000/34343 [2:02:46<2:17:33,  2.95it/s]

    Closest words to the center word they: ['they', 'we', 'you', 'matres', 'gesserit', 'liars', 'lifes', 'microtubules', 'themselves', 'findable']


    Epoch 2/2:  29%|██▉       | 10099/34343 [2:04:10<32:24:59,  4.81s/it]

    Epoch 2 Batch 10100 loss: 2.0954537391662598


    Epoch 2/2:  29%|██▉       | 10100/34343 [2:04:11<23:56:45,  3.56s/it]

    Closest words to the center word international: ['ifrcs', 'ifc', 'iom', 'icrm', 'ifad', 'international', 'opcw', 'icftu', 'acct', 'ibrd']


    Epoch 2/2:  30%|██▉       | 10199/34343 [2:05:18<3:08:10,  2.14it/s] 

    Epoch 2 Batch 10200 loss: 2.101100206375122


    Epoch 2/2:  30%|██▉       | 10200/34343 [2:05:19<3:37:36,  1.85it/s]

    Closest words to the center word led: ['led', 'unido', 'contributed', 'helped', 'ifrcs', 'caused', 'unctad', 'resulted', 'icrm', 'maaouya']


    Epoch 2/2:  30%|██▉       | 10299/34343 [2:06:28<1:27:50,  4.56it/s] 

    Epoch 2 Batch 10300 loss: 2.0999958515167236


    Epoch 2/2:  30%|██▉       | 10300/34343 [2:06:28<2:17:49,  2.91it/s]

    Closest words to the center word parker: ['parker', 'clooney', 'bandleader', 'giroux', 'stevie', 'holliday', 'astaire', 'saxophonist', 'gillespie', 'zant']


    Epoch 2/2:  30%|███       | 10399/34343 [2:07:39<1:27:27,  4.56it/s] 

    Epoch 2 Batch 10400 loss: 2.1005687713623047


    Epoch 2/2:  30%|███       | 10400/34343 [2:07:40<2:29:13,  2.67it/s]

    Closest words to the center word addresses: ['findable', 'addresses', 'rfc', 'dhcp', 'ipv', 'dns', 'address', 'servers', 'ip', 'iec']


    Epoch 2/2:  31%|███       | 10499/34343 [2:09:04<16:37:46,  2.51s/it]

    Epoch 2 Batch 10500 loss: 2.1011013984680176


    Epoch 2/2:  31%|███       | 10500/34343 [2:09:05<13:04:12,  1.97s/it]

    Closest words to the center word thought: ['thought', 'monistic', 'falsifiable', 'rationalism', 'materialism', 'panentheism', 'pantheism', 'leninism', 'empirically', 'testable']


    Epoch 2/2:  31%|███       | 10599/34343 [2:10:14<1:58:40,  3.33it/s] 

    Epoch 2 Batch 10600 loss: 2.0845556259155273


    Epoch 2/2:  31%|███       | 10600/34343 [2:10:15<2:47:37,  2.36it/s]

    Closest words to the center word while: ['while', 'gangetic', 'although', 'inflicted', 'haliotis', 'though', 'hand', 'whilst', 'tropics', 'callithrix']


    Epoch 2/2:  31%|███       | 10699/34343 [2:11:21<1:24:27,  4.67it/s] 

    Epoch 2 Batch 10700 loss: 2.097136974334717


    Epoch 2/2:  31%|███       | 10700/34343 [2:11:22<2:16:52,  2.88it/s]

    Closest words to the center word one: ['gwh', 'hbk', 'pngimage', 'kwh', 'grt', 'twh', 'cyg', 'sfg', 'jul', 'mjs']


    Epoch 2/2:  31%|███▏      | 10799/34343 [2:12:30<1:22:57,  4.73it/s] 

    Epoch 2 Batch 10800 loss: 2.103605270385742


    Epoch 2/2:  31%|███▏      | 10800/34343 [2:12:31<2:19:16,  2.82it/s]

    Closest words to the center word security: ['opcw', 'unido', 'unctad', 'security', 'iom', 'ifad', 'icrm', 'ifrcs', 'ifc', 'pluriform']


    Epoch 2/2:  32%|███▏      | 10899/34343 [2:13:57<4:02:28,  1.61it/s] 

    Epoch 2 Batch 10900 loss: 2.095186710357666


    Epoch 2/2:  32%|███▏      | 10900/34343 [2:13:57<4:09:25,  1.57it/s]

    Closest words to the center word character: ['character', 'villain', 'characters', 'protagonist', 'scrooge', 'superhero', 'animated', 'toriyama', 'elric', 'hitchhiker']


    Epoch 2/2:  32%|███▏      | 10999/34343 [2:15:06<1:36:48,  4.02it/s] 

    Epoch 2 Batch 11000 loss: 2.1248416900634766


    Epoch 2/2:  32%|███▏      | 11000/34343 [2:15:07<2:32:50,  2.55it/s]

    Closest words to the center word mobil: ['kwh', 'mobil', 'gwh', 'twh', 'dwt', 'pluriform', 'intelsat', 'exxon', 'nasdaq', 'motors']


    Epoch 2/2:  32%|███▏      | 11099/34343 [2:16:47<2:00:34,  3.21it/s] 

    Epoch 2 Batch 11100 loss: 2.1157004833221436


    Epoch 2/2:  32%|███▏      | 11100/34343 [2:16:48<2:50:25,  2.27it/s]

    Closest words to the center word logical: ['logical', 'adjoint', 'axiom', 'propositional', 'mathfrak', 'ponens', 'axiomatic', 'cognitivism', 'deterministic', 'countable']


    Epoch 2/2:  33%|███▎      | 11199/34343 [2:18:38<29:11:27,  4.54s/it]

    Epoch 2 Batch 11200 loss: 2.076874256134033


    Epoch 2/2:  33%|███▎      | 11200/34343 [2:18:40<22:29:43,  3.50s/it]

    Closest words to the center word players: ['players', 'playoffs', 'linemen', 'orioles', 'browns', 'astros', 'nhl', 'bengals', 'broncos', 'teams']


    Epoch 2/2:  33%|███▎      | 11299/34343 [2:20:07<4:00:56,  1.59it/s] 

    Epoch 2 Batch 11300 loss: 2.1086976528167725


    Epoch 2/2:  33%|███▎      | 11300/34343 [2:20:08<4:26:09,  1.44it/s]

    Closest words to the center word facilitate: ['facilitate', 'nonsignatory', 'ifrcs', 'iom', 'ifad', 'opcw', 'stimulate', 'motivate', 'induce', 'ifc']


    Epoch 2/2:  33%|███▎      | 11399/34343 [2:21:36<1:44:33,  3.66it/s] 

    Epoch 2 Batch 11400 loss: 2.0916242599487305


    Epoch 2/2:  33%|███▎      | 11400/34343 [2:21:37<2:46:34,  2.30it/s]

    Closest words to the center word minister: ['minister', 'prime', 'csu', 'ministers', 'cdu', 'succeeds', 'fdp', 'boutros', 'chancellor', 'strau']


    Epoch 2/2:  33%|███▎      | 11499/34343 [2:23:01<1:36:37,  3.94it/s] 

    Epoch 2 Batch 11500 loss: 2.126551866531372


    Epoch 2/2:  33%|███▎      | 11500/34343 [2:23:02<2:31:42,  2.51it/s]

    Closest words to the center word mind: ['mind', 'phenomenology', 'cognition', 'egoism', 'cognitivism', 'objectivism', 'consciousness', 'metaphysics', 'intellect', 'brahman']


    Epoch 2/2:  34%|███▍      | 11599/34343 [2:24:55<22:51:01,  3.62s/it]

    Epoch 2 Batch 11600 loss: 2.094184637069702


    Epoch 2/2:  34%|███▍      | 11600/34343 [2:24:56<17:26:12,  2.76s/it]

    Closest words to the center word performance: ['performance', 'throughput', 'lossy', 'compression', 'efficiency', 'macroeconomic', 'paced', 'superscalar', 'airplay', 'lossless']


    Epoch 2/2:  34%|███▍      | 11699/34343 [2:26:50<5:50:58,  1.08it/s] 

    Epoch 2 Batch 11700 loss: 2.1219887733459473


    Epoch 2/2:  34%|███▍      | 11700/34343 [2:26:52<7:47:46,  1.24s/it]

    Closest words to the center word has: ['has', 'hasn', 'had', 'have', 'ifrcs', 'pluriform', 'having', 'possesses', 'hadn', 'ifad']


    Epoch 2/2:  34%|███▍      | 11799/34343 [2:28:21<1:34:59,  3.96it/s] 

    Epoch 2 Batch 11800 loss: 2.106555938720703


    Epoch 2/2:  34%|███▍      | 11800/34343 [2:28:22<2:15:25,  2.77it/s]

    Closest words to the center word which: ['which', 'enthalpy', 'cytosol', 'insoluble', 'countably', 'piezoelectric', 'centripetal', 'halide', 'symplectic', 'endothermic']


    Epoch 2/2:  35%|███▍      | 11899/34343 [2:30:03<1:39:25,  3.76it/s] 

    Epoch 2 Batch 11900 loss: 2.116100549697876


    Epoch 2/2:  35%|███▍      | 11900/34343 [2:30:03<2:22:48,  2.62it/s]

    Closest words to the center word south: ['south', 'chungcheong', 'busan', 'irian', 'gyeonggi', 'sulawesi', 'north', 'tenggara', 'savannas', 'southwest']


    Epoch 2/2:  35%|███▍      | 11999/34343 [2:31:54<13:58:33,  2.25s/it]

    Epoch 2 Batch 12000 loss: 2.086256504058838


    Epoch 2/2:  35%|███▍      | 12000/34343 [2:31:54<10:59:20,  1.77s/it]

    Closest words to the center word possessed: ['possessed', 'transcendent', 'misled', 'lifes', 'immanent', 'harkonnen', 'peleus', 'knew', 'exercised', 'deified']


    Epoch 2/2:  35%|███▌      | 12099/34343 [2:33:26<2:26:01,  2.54it/s] 

    Epoch 2 Batch 12100 loss: 2.1063146591186523


    Epoch 2/2:  35%|███▌      | 12100/34343 [2:33:27<2:58:50,  2.07it/s]

    Closest words to the center word collaboration: ['collaboration', 'wagoner', 'collaborated', 'interviews', 'friendship', 'collaborator', 'finalist', 'diffie', 'interview', 'emmylou']


    Epoch 2/2:  36%|███▌      | 12199/34343 [2:35:50<2:06:20,  2.92it/s] 

    Epoch 2 Batch 12200 loss: 2.123460531234741


    Epoch 2/2:  36%|███▌      | 12200/34343 [2:35:51<3:11:04,  1.93it/s]

    Closest words to the center word any: ['any', 'anything', 'newnode', 'lastnode', 'prev', 'every', 'baryonic', 'whatsoever', 'differentiable', 'antiderivatives']


    Epoch 2/2:  36%|███▌      | 12299/34343 [2:38:04<1:54:01,  3.22it/s] 

    Epoch 2 Batch 12300 loss: 2.0970840454101562


    Epoch 2/2:  36%|███▌      | 12300/34343 [2:38:05<3:06:39,  1.97it/s]

    Closest words to the center word ad: ['ad', 'argumentum', 'hominem', 'bc', 'bce', 'reductio', 'ce', 'hoc', 'yoannis', 'propter']


    Epoch 2/2:  36%|███▌      | 12399/34343 [2:40:50<9:16:06,  1.52s/it] 

    Epoch 2 Batch 12400 loss: 2.1157491207122803


    Epoch 2/2:  36%|███▌      | 12400/34343 [2:40:51<7:46:29,  1.28s/it]

    Closest words to the center word twice: ['twice', 'innings', 'lifes', 'playoffs', 'strikeouts', 'batted', 'grt', 'intercalary', 'inning', 'centimeters']


    Epoch 2/2:  36%|███▋      | 12499/34343 [2:42:33<2:11:32,  2.77it/s] 

    Epoch 2 Batch 12500 loss: 2.1198954582214355


    Epoch 2/2:  36%|███▋      | 12500/34343 [2:42:34<2:57:29,  2.05it/s]

    Closest words to the center word five: ['gwh', 'grt', 'kwh', 'hbk', 'twh', 'cyg', 'sfg', 'pngimage', 'mjs', 'jul']


    Epoch 2/2:  37%|███▋      | 12599/34343 [2:45:08<1:47:23,  3.37it/s] 

    Epoch 2 Batch 12600 loss: 2.142111301422119


    Epoch 2/2:  37%|███▋      | 12600/34343 [2:45:09<2:43:01,  2.22it/s]

    Closest words to the center word compilers: ['compilers', 'lisp', 'macros', 'opengl', 'bytecode', 'applets', 'implementations', 'brainfuck', 'debugger', 'compiler']


    Epoch 2/2:  37%|███▋      | 12699/34343 [2:47:06<1:36:24,  3.74it/s] 

    Epoch 2 Batch 12700 loss: 2.0801544189453125


    Epoch 2/2:  37%|███▋      | 12700/34343 [2:47:33<48:57:58,  8.14s/it]

    Closest words to the center word approved: ['approved', 'ratified', 'iom', 'amended', 'fda', 'unicameral', 'kwh', 'impeached', 'ifad', 'ratification']


    Epoch 2/2:  37%|███▋      | 12799/34343 [2:49:07<4:52:12,  1.23it/s] 

    Epoch 2 Batch 12800 loss: 2.1307029724121094


    Epoch 2/2:  37%|███▋      | 12800/34343 [2:49:07<4:51:26,  1.23it/s]

    Closest words to the center word auger: ['auger', 'icrm', 'lorentz', 'physicist', 'coulomb', 'ibrd', 'leiserson', 'krebs', 'nsted', 'ment']


    Epoch 2/2:  38%|███▊      | 12899/34343 [2:50:40<1:33:24,  3.83it/s] 

    Epoch 2 Batch 12900 loss: 2.14245343208313


    Epoch 2/2:  38%|███▊      | 12900/34343 [2:50:41<2:32:01,  2.35it/s]

    Closest words to the center word stadium: ['stadium', 'chargers', 'playoffs', 'texans', 'afc', 'nfc', 'oakland', 'playoff', 'broncos', 'steelers']


    Epoch 2/2:  38%|███▊      | 12999/34343 [2:52:15<1:43:25,  3.44it/s] 

    Epoch 2 Batch 13000 loss: 2.105532646179199


    Epoch 2/2:  38%|███▊      | 13000/34343 [2:52:16<3:24:16,  1.74it/s]

    Closest words to the center word protected: ['protected', 'regulated', 'unido', 'wastes', 'dwt', 'administered', 'vesicles', 'ifrcs', 'dumping', 'pluriform']


    Epoch 2/2:  38%|███▊      | 13099/34343 [2:54:17<34:21:46,  5.82s/it]

    Epoch 2 Batch 13100 loss: 2.138997793197632


    Epoch 2/2:  38%|███▊      | 13100/34343 [2:54:18<25:29:28,  4.32s/it]

    Closest words to the center word subsequent: ['subsequent', 'culminated', 'perestroika', 'amanullah', 'abbasid', 'glasnost', 'ensuing', 'hastened', 'mauryan', 'resulted']


    Epoch 2/2:  38%|███▊      | 13199/34343 [2:55:59<2:08:13,  2.75it/s] 

    Epoch 2 Batch 13200 loss: 2.1364693641662598


    Epoch 2/2:  38%|███▊      | 13200/34343 [2:56:00<2:47:41,  2.10it/s]

    Closest words to the center word competitively: ['competitively', 'kart', 'soybeans', 'aroma', 'firmware', 'skiing', 'bikes', 'gba', 'ductile', 'harmonicas']


    Epoch 2/2:  39%|███▊      | 13299/34343 [2:57:41<1:52:28,  3.12it/s] 

    Epoch 2 Batch 13300 loss: 2.0943915843963623


    Epoch 2/2:  39%|███▊      | 13300/34343 [2:57:42<3:12:14,  1.82it/s]

    Closest words to the center word clairvoyance: ['clairvoyance', 'cognitivism', 'testable', 'falsifiable', 'therapies', 'memetics', 'deconstructive', 'paranormal', 'anecdotal', 'anthropic']


    Epoch 2/2:  39%|███▉      | 13399/34343 [2:59:12<1:37:28,  3.58it/s] 

    Epoch 2 Batch 13400 loss: 2.108501434326172


    Epoch 2/2:  39%|███▉      | 13400/34343 [2:59:13<2:34:05,  2.27it/s]

    Closest words to the center word and: ['ifrcs', 'ifad', 'ifc', 'icrm', 'unmibh', 'opcw', 'unctad', 'rajonas', 'classis', 'icftu']


    Epoch 2/2:  39%|███▉      | 13499/34343 [3:01:10<5:41:02,  1.02it/s] 

    Epoch 2 Batch 13500 loss: 2.14046573638916


    Epoch 2/2:  39%|███▉      | 13500/34343 [3:01:11<5:51:50,  1.01s/it]

    Closest words to the center word premier: ['premier', 'premiers', 'finalist', 'famers', 'chungcheong', 'coach', 'diamondbacks', 'nfc', 'football', 'footballers']


    Epoch 2/2:  40%|███▉      | 13599/34343 [3:02:46<1:55:13,  3.00it/s] 

    Epoch 2 Batch 13600 loss: 2.1126320362091064


    Epoch 2/2:  40%|███▉      | 13600/34343 [3:02:47<2:40:46,  2.15it/s]

    Closest words to the center word sperm: ['sperm', 'diploid', 'eukaryotic', 'haploid', 'secrete', 'homologous', 'hormones', 'gamete', 'chromosomes', 'mitochondria']


    Epoch 2/2:  40%|███▉      | 13699/34343 [3:04:21<1:34:50,  3.63it/s] 

    Epoch 2 Batch 13700 loss: 2.0808300971984863


    Epoch 2/2:  40%|███▉      | 13700/34343 [3:04:22<2:29:12,  2.31it/s]

    Closest words to the center word narrow: ['narrow', 'steep', 'sloping', 'humid', 'mountainous', 'gauge', 'kilometer', 'km', 'elevations', 'westerly']


    Epoch 2/2:  40%|████      | 13799/34343 [3:06:09<33:13:50,  5.82s/it]

    Epoch 2 Batch 13800 loss: 2.062546730041504


    Epoch 2/2:  40%|████      | 13800/34343 [3:06:10<24:34:07,  4.31s/it]

    Closest words to the center word were: ['were', 'are', 'persecuted', 'massacred', 'outnumbered', 'raids', 'aediles', 'serbs', 'ukrainians', 'byzantines']


    Epoch 2/2:  40%|████      | 13899/34343 [3:07:41<4:24:37,  1.29it/s] 

    Epoch 2 Batch 13900 loss: 2.090404987335205


    Epoch 2/2:  40%|████      | 13900/34343 [3:07:42<4:24:32,  1.29it/s]

    Closest words to the center word chicago: ['chicago', 'devry', 'abet', 'culver', 'kansas', 'illinois', 'urbana', 'detroit', 'moines', 'tacoma']


    Epoch 2/2:  41%|████      | 13999/34343 [3:09:10<1:45:37,  3.21it/s] 

    Epoch 2 Batch 14000 loss: 2.1093571186065674


    Epoch 2/2:  41%|████      | 14000/34343 [3:09:11<2:32:37,  2.22it/s]

    Closest words to the center word breeding: ['breeding', 'ferus', 'shrubs', 'crocodiles', 'habitat', 'cassava', 'prey', 'tusks', 'larvae', 'vectorborne']


    Epoch 2/2:  41%|████      | 14099/34343 [3:10:53<1:39:30,  3.39it/s] 

    Epoch 2 Batch 14100 loss: 2.117936134338379


    Epoch 2/2:  41%|████      | 14100/34343 [3:10:54<2:35:55,  2.16it/s]

    Closest words to the center word revolution: ['revolution', 'bolsheviks', 'revolutionary', 'uprising', 'socialist', 'rsdlp', 'kuomintang', 'anarchism', 'sino', 'maoist']


    Epoch 2/2:  41%|████▏     | 14199/34343 [3:12:49<26:48:52,  4.79s/it]

    Epoch 2 Batch 14200 loss: 2.1263201236724854


    Epoch 2/2:  41%|████▏     | 14200/34343 [3:12:50<19:57:05,  3.57s/it]

    Closest words to the center word jeong: ['jeong', 'tiberian', 'hunmin', 'polskiej', 'cccc', 'cooh', 'eum', 'stosunku', 'rzeczypospolitej', 'ttt']


    Epoch 2/2:  42%|████▏     | 14299/34343 [3:14:17<3:23:05,  1.64it/s] 

    Epoch 2 Batch 14300 loss: 2.089411497116089


    Epoch 2/2:  42%|████▏     | 14300/34343 [3:14:18<4:14:42,  1.31it/s]

    Closest words to the center word morning: ['morning', 'evening', 'friday', 'saturday', 'afternoon', 'selamat', 'night', 'tonight', 'thursday', 'solstice']


    Epoch 2/2:  42%|████▏     | 14399/34343 [3:15:56<1:38:47,  3.36it/s] 

    Epoch 2 Batch 14400 loss: 2.0882961750030518


    Epoch 2/2:  42%|████▏     | 14400/34343 [3:15:57<2:24:45,  2.30it/s]

    Closest words to the center word does: ['does', 'findable', 'did', 'doesn', 'didn', 'factum', 'wouldn', 'opcw', 'do', 'iom']


    Epoch 2/2:  42%|████▏     | 14499/34343 [3:17:39<1:27:26,  3.78it/s] 

    Epoch 2 Batch 14500 loss: 2.1101858615875244


    Epoch 2/2:  42%|████▏     | 14500/34343 [3:17:39<2:32:22,  2.17it/s]

    Closest words to the center word m: ['elke', 'm', 'cdots', 'ldots', 'stosunku', 'otimes', 'cdot', 'nchen', 'ttt', 'leq']


    Epoch 2/2:  43%|████▎     | 14599/34343 [3:19:45<25:34:12,  4.66s/it]

    Epoch 2 Batch 14600 loss: 2.106454849243164


    Epoch 2/2:  43%|████▎     | 14600/34343 [3:19:45<19:05:12,  3.48s/it]

    Closest words to the center word ouen: ['yoannis', 'basilica', 'churchyard', 'methodius', 'grenadines', 'sistine', 'kitts', 'chapel', 'hagia', 'benedictine']


    Epoch 2/2:  43%|████▎     | 14699/34343 [3:21:29<2:48:26,  1.94it/s] 

    Epoch 2 Batch 14700 loss: 2.0788698196411133


    Epoch 2/2:  43%|████▎     | 14700/34343 [3:21:30<3:19:19,  1.64it/s]

    Closest words to the center word terrorist: ['terrorist', 'qaeda', 'opcw', 'ifrcs', 'hezbollah', 'unido', 'terrorism', 'paramilitary', 'islamist', 'peacekeeping']


    Epoch 2/2:  43%|████▎     | 14799/34343 [3:23:04<1:33:37,  3.48it/s] 

    Epoch 2 Batch 14800 loss: 2.0820975303649902


    Epoch 2/2:  43%|████▎     | 14800/34343 [3:23:05<2:20:17,  2.32it/s]

    Closest words to the center word republicans: ['republicans', 'bolsheviks', 'conservatives', 'wingers', 'democrats', 'voters', 'whigs', 'liberals', 'militias', 'communists']


    Epoch 2/2:  43%|████▎     | 14899/34343 [3:24:44<1:40:07,  3.24it/s] 

    Epoch 2 Batch 14900 loss: 2.08479642868042


    Epoch 2/2:  43%|████▎     | 14900/34343 [3:24:44<2:29:13,  2.17it/s]

    Closest words to the center word replacing: ['pluriform', 'replacing', 'imac', 'dwt', 'replaced', 'ipf', 'powerbook', 'otimes', 'cccc', 'bbbb']


    Epoch 2/2:  44%|████▎     | 14999/34343 [3:26:59<11:40:42,  2.17s/it]

    Epoch 2 Batch 15000 loss: 2.081838607788086


    Epoch 2/2:  44%|████▎     | 15000/34343 [3:26:59<9:28:21,  1.76s/it] 

    Closest words to the center word this: ['this', 'enthalpy', 'epimenides', 'empirically', 'findable', 'falsifiability', 'minimax', 'strictest', 'casuistry', 'irreducibly']


    Epoch 2/2:  44%|████▍     | 15099/34343 [3:28:29<1:48:28,  2.96it/s] 

    Epoch 2 Batch 15100 loss: 2.1021761894226074


    Epoch 2/2:  44%|████▍     | 15100/34343 [3:28:30<2:20:39,  2.28it/s]

    Closest words to the center word but: ['but', 'though', 'unless', 'liars', 'countably', 'lifes', 'nor', 'preregular', 'because', 'consummated']


    Epoch 2/2:  44%|████▍     | 15199/34343 [3:30:04<1:24:06,  3.79it/s] 

    Epoch 2 Batch 15200 loss: 2.1122500896453857


    Epoch 2/2:  44%|████▍     | 15200/34343 [3:30:04<2:07:30,  2.50it/s]

    Closest words to the center word called: ['called', 'agave', 'termed', 'referred', 'labiodental', 'leontopithecus', 'endomorphism', 'denoted', 'disambiguation', 'paracompact']


    Epoch 2/2:  45%|████▍     | 15299/34343 [3:31:48<1:58:21,  2.68it/s] 

    Epoch 2 Batch 15300 loss: 2.0835440158843994


    Epoch 2/2:  45%|████▍     | 15300/34343 [3:31:49<2:46:14,  1.91it/s]

    Closest words to the center word or: ['or', 'alkoxide', 'clonic', 'ifrcs', 'waterborne', 'ketone', 'tachycardia', 'unsaturated', 'newnode', 'sibilant']


    Epoch 2/2:  45%|████▍     | 15399/34343 [3:34:04<5:51:17,  1.11s/it] 

    Epoch 2 Batch 15400 loss: 2.1030282974243164


    Epoch 2/2:  45%|████▍     | 15400/34343 [3:34:05<5:24:45,  1.03s/it]

    Closest words to the center word refer: ['refer', 'describe', 'refers', 'referred', 'relate', 'adhere', 'refered', 'belong', 'referring', 'classify']


    Epoch 2/2:  45%|████▌     | 15499/34343 [3:35:51<1:42:49,  3.05it/s] 

    Epoch 2 Batch 15500 loss: 2.0975685119628906


    Epoch 2/2:  45%|████▌     | 15500/34343 [3:35:52<2:29:14,  2.10it/s]

    Closest words to the center word civil: ['civil', 'liberties', 'pluriform', 'criminal', 'secedes', 'judicial', 'suffrage', 'icao', 'williamite', 'confederate']


    Epoch 2/2:  45%|████▌     | 15599/34343 [3:37:24<1:17:05,  4.05it/s] 

    Epoch 2 Batch 15600 loss: 2.0705533027648926


    Epoch 2/2:  45%|████▌     | 15600/34343 [3:37:24<1:57:43,  2.65it/s]

    Closest words to the center word not: ['not', 'liars', 'unable', 'refutable', 'repent', 'empirically', 'disprove', 'necessarily', 'wouldn', 'nor']


    Epoch 2/2:  46%|████▌     | 15699/34343 [3:39:30<37:02:00,  7.15s/it]

    Epoch 2 Batch 15700 loss: 2.1161015033721924


    Epoch 2/2:  46%|████▌     | 15700/34343 [3:39:31<26:55:43,  5.20s/it]

    Closest words to the center word less: ['less', 'lifes', 'more', 'cheaper', 'thicker', 'softer', 'viscous', 'than', 'hotter', 'denser']


    Epoch 2/2:  46%|████▌     | 15799/34343 [3:41:09<2:42:23,  1.90it/s] 

    Epoch 2 Batch 15800 loss: 2.139280080795288


    Epoch 2/2:  46%|████▌     | 15800/34343 [3:41:09<2:57:06,  1.74it/s]

    Closest words to the center word to: ['to', 'findable', 'inability', 'able', 'unable', 'obliged', 'prev', 'willing', 'stimulate', 'attempt']


    Epoch 2/2:  46%|████▋     | 15899/34343 [3:43:03<1:25:23,  3.60it/s] 

    Epoch 2 Batch 15900 loss: 2.098975658416748


    Epoch 2/2:  46%|████▋     | 15900/34343 [3:43:04<2:06:48,  2.42it/s]

    Closest words to the center word screen: ['screen', 'bitmap', 'ipf', 'backlit', 'scrolling', 'playstation', 'gba', 'raster', 'clicking', 'rgb']


    Epoch 2/2:  47%|████▋     | 15999/34343 [3:44:43<1:49:17,  2.80it/s] 

    Epoch 2 Batch 16000 loss: 2.151726722717285


    Epoch 2/2:  47%|████▋     | 16000/34343 [3:44:44<2:29:53,  2.04it/s]

    Closest words to the center word speed: ['speed', 'speeds', 'voltage', 'velocity', 'throughput', 'torque', 'subsonic', 'mbit', 'mhz', 'voltages']


    Epoch 2/2:  47%|████▋     | 16099/34343 [3:46:49<7:16:01,  1.43s/it] 

    Epoch 2 Batch 16100 loss: 2.0663247108459473


    Epoch 2/2:  47%|████▋     | 16100/34343 [3:46:50<6:08:33,  1.21s/it]

    Closest words to the center word libraries: ['libraries', 'cygwin', 'compilers', 'netbsd', 'findable', 'gpled', 'browsers', 'directories', 'recompilation', 'gnu']


    Epoch 2/2:  47%|████▋     | 16199/34343 [3:48:33<1:56:08,  2.60it/s] 

    Epoch 2 Batch 16200 loss: 2.0803465843200684


    Epoch 2/2:  47%|████▋     | 16200/34343 [3:48:34<2:36:48,  1.93it/s]

    Closest words to the center word its: ['its', 'ifrcs', 'ifad', 'their', 'icrm', 'ifc', 'iom', 'findable', 'nonsignatory', 'iho']


    Epoch 2/2:  47%|████▋     | 16299/34343 [3:50:22<1:54:51,  2.62it/s] 

    Epoch 2 Batch 16300 loss: 2.09684681892395


    Epoch 2/2:  47%|████▋     | 16300/34343 [3:50:24<4:55:49,  1.02it/s]

    Closest words to the center word decided: ['decided', 'refused', 'vowed', 'persuaded', 'wanted', 'opted', 'agreed', 'didn', 'announced', 'obliged']


    Epoch 2/2:  48%|████▊     | 16399/34343 [3:52:12<1:27:55,  3.40it/s] 

    Epoch 2 Batch 16400 loss: 2.0894036293029785


    Epoch 2/2:  48%|████▊     | 16400/34343 [3:52:40<42:56:31,  8.62s/it]

    Closest words to the center word cathedral: ['cathedral', 'basilica', 'churchyard', 'sistine', 'yoannis', 'chapel', 'abbey', 'sepulchre', 'convent', 'hagia']


    Epoch 2/2:  48%|████▊     | 16499/34343 [3:54:18<5:18:15,  1.07s/it] 

    Epoch 2 Batch 16500 loss: 2.095620632171631


    Epoch 2/2:  48%|████▊     | 16500/34343 [3:54:19<5:20:31,  1.08s/it]

    Closest words to the center word atoms: ['atoms', 'protons', 'covalent', 'electrons', 'orbitals', 'ions', 'neutrons', 'particles', 'hydrogen', 'molecules']


    Epoch 2/2:  48%|████▊     | 16599/34343 [3:56:04<1:55:28,  2.56it/s] 

    Epoch 2 Batch 16600 loss: 2.1025757789611816


    Epoch 2/2:  48%|████▊     | 16600/34343 [3:56:06<3:16:47,  1.50it/s]

    Closest words to the center word an: ['an', 'iom', 'ifrcs', 'ifc', 'nonsignatory', 'unido', 'argumentum', 'laia', 'wmo', 'wtro']


    Epoch 2/2:  49%|████▊     | 16699/34343 [3:57:51<1:32:37,  3.17it/s] 

    Epoch 2 Batch 16700 loss: 2.0925910472869873


    Epoch 2/2:  49%|████▊     | 16700/34343 [3:57:51<2:09:48,  2.27it/s]

    Closest words to the center word being: ['being', 'having', 'acutely', 'orally', 'divinely', 'universally', 'heretical', 'oxidizing', 'insoluble', 'amplify']


    Epoch 2/2:  49%|████▉     | 16799/34343 [3:59:55<32:53:01,  6.75s/it]

    Epoch 2 Batch 16800 loss: 2.0983142852783203


    Epoch 2/2:  49%|████▉     | 16800/34343 [3:59:56<24:09:14,  4.96s/it]

    Closest words to the center word a: ['a', 'pluriform', 'agave', 'newnode', 'insubstantial', 'keying', 'unital', 'prev', 'globicephala', 'enthalpy']


    Epoch 2/2:  49%|████▉     | 16899/34343 [4:01:36<4:09:39,  1.16it/s] 

    Epoch 2 Batch 16900 loss: 2.0604052543640137


    Epoch 2/2:  49%|████▉     | 16900/34343 [4:01:37<4:08:10,  1.17it/s]

    Closest words to the center word of: ['of', 'ifrcs', 'ifad', 'rajonas', 'wtoo', 'unido', 'ifc', 'icrm', 'nazarene', 'wftu']


    Epoch 2/2:  49%|████▉     | 16999/34343 [4:03:11<1:22:44,  3.49it/s] 

    Epoch 2 Batch 17000 loss: 2.0964388847351074


    Epoch 2/2:  50%|████▉     | 17000/34343 [4:03:12<2:02:30,  2.36it/s]

    Closest words to the center word area: ['area', 'sq', 'unpaved', 'runways', 'ecoregion', 'irrigated', 'subtropical', 'isthmus', 'km', 'kilometers']


    Epoch 2/2:  50%|████▉     | 17099/34343 [4:04:46<1:10:31,  4.08it/s] 

    Epoch 2 Batch 17100 loss: 2.1100945472717285


    Epoch 2/2:  50%|████▉     | 17100/34343 [4:04:47<1:53:03,  2.54it/s]

    Closest words to the center word days: ['days', 'months', 'grt', 'weeks', 'hours', 'minutes', 'lifes', 'lasts', 'gregorian', 'friday']


    Epoch 2/2:  50%|█████     | 17199/34343 [4:06:41<23:06:24,  4.85s/it]

    Epoch 2 Batch 17200 loss: 2.0566067695617676


    Epoch 2/2:  50%|█████     | 17200/34343 [4:06:42<17:16:30,  3.63s/it]

    Closest words to the center word years: ['years', 'grt', 'decades', 'months', 'weeks', 'servicemales', 'availabilitymales', 'lifes', 'gwh', 'males']


    Epoch 2/2:  50%|█████     | 17299/34343 [4:08:10<2:30:49,  1.88it/s] 

    Epoch 2 Batch 17300 loss: 2.074434280395508


    Epoch 2/2:  50%|█████     | 17300/34343 [4:08:11<2:42:48,  1.74it/s]

    Closest words to the center word core: ['core', 'cisc', 'zseries', 'xt', 'graphical', 'desktop', 'sparc', 'athlon', 'risc', 'microprocessor']


    Epoch 2/2:  51%|█████     | 17399/34343 [4:09:55<5:27:25,  1.16s/it] 

    Epoch 2 Batch 17400 loss: 2.108968734741211


    Epoch 2/2:  51%|█████     | 17400/34343 [4:09:55<4:51:27,  1.03s/it]

    Closest words to the center word both: ['both', 'bilabial', 'mutually', 'mizrahi', 'respective', 'maronites', 'trigonometric', 'fricatives', 'subtraction', 'laminal']


    Epoch 2/2:  51%|█████     | 17499/34343 [4:11:48<2:47:20,  1.68it/s] 

    Epoch 2 Batch 17500 loss: 2.0946757793426514


    Epoch 2/2:  51%|█████     | 17500/34343 [4:11:49<2:59:41,  1.56it/s]

    Closest words to the center word of: ['of', 'ifrcs', 'ifad', 'rajonas', 'wtoo', 'icrm', 'ifc', 'unido', 'akan', 'nazarene']


    Epoch 2/2:  51%|█████     | 17599/34343 [4:13:39<12:51:04,  2.76s/it]

    Epoch 2 Batch 17600 loss: 2.121558666229248


    Epoch 2/2:  51%|█████     | 17600/34343 [4:13:40<10:11:19,  2.19s/it]

    Closest words to the center word tag: ['tag', 'pngimage', 'mjs', 'mjd', 'cccc', 'bbbb', 'komm', 'findable', 'meine', 'jesu']


    Epoch 2/2:  52%|█████▏    | 17699/34343 [4:15:14<1:47:47,  2.57it/s] 

    Epoch 2 Batch 17700 loss: 2.097524642944336


    Epoch 2/2:  52%|█████▏    | 17700/34343 [4:15:15<2:19:00,  2.00it/s]

    Closest words to the center word club: ['club', 'bruins', 'cyclopedia', 'texans', 'broncos', 'diamondbacks', 'stadium', 'vsl', 'sox', 'coppa']


    Epoch 2/2:  52%|█████▏    | 17799/34343 [4:16:47<59:40,  4.62it/s]   

    Epoch 2 Batch 17800 loss: 2.089198589324951


    Epoch 2/2:  52%|█████▏    | 17800/34343 [4:16:48<1:35:35,  2.88it/s]

    Closest words to the center word those: ['those', 'lifes', 'felonies', 'consenting', 'neopagans', 'disapprove', 'upu', 'unido', 'nontrinitarian', 'denominations']


    Epoch 2/2:  52%|█████▏    | 17899/34343 [4:18:00<59:40,  4.59it/s]   

    Epoch 2 Batch 17900 loss: 2.1561672687530518


    Epoch 2/2:  52%|█████▏    | 17900/34343 [4:18:01<1:56:20,  2.36it/s]

    Closest words to the center word moment: ['moment', 'dipole', 'newnode', 'tangent', 'eccentricity', 'velocity', 'prev', 'penumbra', 'rotational', 'hyperfocal']


    Epoch 2/2:  52%|█████▏    | 17999/34343 [4:19:30<5:07:31,  1.13s/it] 

    Epoch 2 Batch 18000 loss: 2.089468002319336


    Epoch 2/2:  52%|█████▏    | 18000/34343 [4:19:31<4:30:23,  1.01it/s]

    Closest words to the center word mode: ['mode', 'bitmap', 'debugger', 'ipf', 'usb', 'cpu', 'processor', 'xt', 'scsi', 'asynchronous']


    Epoch 2/2:  53%|█████▎    | 18099/34343 [4:20:44<2:01:01,  2.24it/s] 

    Epoch 2 Batch 18100 loss: 2.1045188903808594


    Epoch 2/2:  53%|█████▎    | 18100/34343 [4:20:45<2:18:03,  1.96it/s]

    Closest words to the center word created: ['created', 'pluriform', 'coined', 'unido', 'nonsignatory', 'popularized', 'invented', 'supplanted', 'founded', 'formed']


    Epoch 2/2:  53%|█████▎    | 18199/34343 [4:21:56<59:45,  4.50it/s]   

    Epoch 2 Batch 18200 loss: 2.0475542545318604


    Epoch 2/2:  53%|█████▎    | 18200/34343 [4:21:57<1:34:06,  2.86it/s]

    Closest words to the center word is: ['is', 'refers', 'differentiable', 'satisfies', 'diffeomorphism', 'bijective', 'metrizable', 'converges', 'endomorphism', 'subgroup']


    Epoch 2/2:  53%|█████▎    | 18299/34343 [4:23:07<56:31,  4.73it/s]   

    Epoch 2 Batch 18300 loss: 2.0971145629882812


    Epoch 2/2:  53%|█████▎    | 18300/34343 [4:23:23<22:11:16,  4.98s/it]

    Closest words to the center word furigana: ['furigana', 'hiragana', 'kanji', 'katakana', 'diacritic', 'perfective', 'digraphs', 'approximant', 'phonemes', 'devanagari']


    Epoch 2/2:  54%|█████▎    | 18399/34343 [4:24:36<1:44:15,  2.55it/s] 

    Epoch 2 Batch 18400 loss: 2.1299948692321777


    Epoch 2/2:  54%|█████▎    | 18400/34343 [4:24:37<2:34:48,  1.72it/s]

    Closest words to the center word tel: ['tel', 'allafrica', 'icrm', 'rajons', 'coquitlam', 'universidade', 'ifad', 'unmibh', 'opcw', 'stadtbahn']


    Epoch 2/2:  54%|█████▍    | 18499/34343 [4:25:56<1:21:16,  3.25it/s] 

    Epoch 2 Batch 18500 loss: 2.0718226432800293


    Epoch 2/2:  54%|█████▍    | 18500/34343 [4:25:57<1:49:22,  2.41it/s]

    Closest words to the center word against: ['against', 'ifrcs', 'unido', 'unmibh', 'prosecute', 'mughals', 'slobodan', 'libel', 'wco', 'invading']


    Epoch 2/2:  54%|█████▍    | 18599/34343 [4:27:23<59:59,  4.37it/s]   

    Epoch 2 Batch 18600 loss: 2.0688459873199463


    Epoch 2/2:  54%|█████▍    | 18600/34343 [4:27:24<1:37:17,  2.70it/s]

    Closest words to the center word after: ['after', 'before', 'shortly', 'lasted', 'afterwards', 'afterward', 'thereafter', 'rafik', 'aegisthus', 'abruptly']


    Epoch 2/2:  54%|█████▍    | 18699/34343 [4:29:09<6:09:38,  1.42s/it] 

    Epoch 2 Batch 18700 loss: 2.076733112335205


    Epoch 2/2:  54%|█████▍    | 18700/34343 [4:29:10<5:07:04,  1.18s/it]

    Closest words to the center word than: ['than', 'lifes', 'less', 'heavier', 'cheaper', 'sweeter', 'denser', 'thicker', 'considerably', 'faster']


    Epoch 2/2:  55%|█████▍    | 18799/34343 [4:30:21<1:20:14,  3.23it/s] 

    Epoch 2 Batch 18800 loss: 2.106260061264038


    Epoch 2/2:  55%|█████▍    | 18800/34343 [4:30:22<1:43:31,  2.50it/s]

    Closest words to the center word has: ['has', 'hasn', 'ifrcs', 'have', 'had', 'pluriform', 'hadn', 'ifad', 'ifc', 'enjoys']


    Epoch 2/2:  55%|█████▌    | 18899/34343 [4:31:32<57:20,  4.49it/s]   

    Epoch 2 Batch 18900 loss: 2.090738296508789


    Epoch 2/2:  55%|█████▌    | 18900/34343 [4:31:32<1:28:23,  2.91it/s]

    Closest words to the center word actor: ['actor', 'actress', 'footballer', 'comedian', 'cricketer', 'pngimage', 'comedienne', 'singer', 'dramatist', 'laureate']


    Epoch 2/2:  55%|█████▌    | 18999/34343 [4:32:42<54:13,  4.72it/s]   

    Epoch 2 Batch 19000 loss: 2.079223394393921


    Epoch 2/2:  55%|█████▌    | 19000/34343 [4:32:43<1:27:56,  2.91it/s]

    Closest words to the center word employed: ['employed', 'dwt', 'used', 'trained', 'practiced', 'conscripted', 'invented', 'appointed', 'oxidizing', 'exploited']


    Epoch 2/2:  56%|█████▌    | 19099/34343 [4:34:08<4:29:40,  1.06s/it] 

    Epoch 2 Batch 19100 loss: 2.087118625640869


    Epoch 2/2:  56%|█████▌    | 19100/34343 [4:34:09<4:03:16,  1.04it/s]

    Closest words to the center word primer: ['primer', 'ifrcs', 'findable', 'ifad', 'nonsignatory', 'ifc', 'erowid', 'rajonas', 'tutorial', 'radiology']


    Epoch 2/2:  56%|█████▌    | 19199/34343 [4:35:18<1:11:42,  3.52it/s] 

    Epoch 2 Batch 19200 loss: 2.094693183898926


    Epoch 2/2:  56%|█████▌    | 19200/34343 [4:35:19<1:39:26,  2.54it/s]

    Closest words to the center word also: ['also', 'disambiguation', 'wmo', 'wftu', 'wtoo', 'yird', 'ifrcs', 'wcl', 'unido', 'ifad']


    Epoch 2/2:  56%|█████▌    | 19299/34343 [4:36:27<54:34,  4.59it/s]   

    Epoch 2 Batch 19300 loss: 2.088292121887207


    Epoch 2/2:  56%|█████▌    | 19300/34343 [4:36:28<1:26:57,  2.88it/s]

    Closest words to the center word words: ['words', 'pronouns', 'nouns', 'verbs', 'cognates', 'adjectives', 'morphemes', 'digraphs', 'phonemes', 'consonants']


    Epoch 2/2:  56%|█████▋    | 19399/34343 [4:37:40<53:26,  4.66it/s]   

    Epoch 2 Batch 19400 loss: 2.1207566261291504


    Epoch 2/2:  56%|█████▋    | 19400/34343 [4:37:56<20:54:01,  5.04s/it]

    Closest words to the center word usually: ['usually', 'typically', 'calcite', 'intravenous', 'circularly', 'hydride', 'often', 'unicellular', 'normally', 'waterborne']


    Epoch 2/2:  57%|█████▋    | 19499/34343 [4:39:05<3:13:23,  1.28it/s] 

    Epoch 2 Batch 19500 loss: 2.0922622680664062


    Epoch 2/2:  57%|█████▋    | 19500/34343 [4:39:05<3:06:51,  1.32it/s]

    Closest words to the center word considerable: ['considerable', 'substantial', 'tremendous', 'significant', 'pluriform', 'decreased', 'huge', 'enormous', 'disproportionate', 'greater']


    Epoch 2/2:  57%|█████▋    | 19599/34343 [4:40:16<1:02:09,  3.95it/s] 

    Epoch 2 Batch 19600 loss: 2.075504779815674


    Epoch 2/2:  57%|█████▋    | 19600/34343 [4:40:17<1:48:26,  2.27it/s]

    Closest words to the center word groups: ['groups', 'ethnic', 'abelian', 'homomorphisms', 'bamar', 'denominations', 'mestizo', 'amerindian', 'malayo', 'group']


    Epoch 2/2:  57%|█████▋    | 19699/34343 [4:41:26<52:01,  4.69it/s]   

    Epoch 2 Batch 19700 loss: 2.104104518890381


    Epoch 2/2:  57%|█████▋    | 19700/34343 [4:41:27<1:27:50,  2.78it/s]

    Closest words to the center word trademark: ['trademark', 'license', 'copyleft', 'infringement', 'copyright', 'abandonware', 'gpl', 'kazaa', 'netbsd', 'fsf']


    Epoch 2/2:  58%|█████▊    | 19799/34343 [4:42:54<19:40:54,  4.87s/it]

    Epoch 2 Batch 19800 loss: 2.1101269721984863


    Epoch 2/2:  58%|█████▊    | 19800/34343 [4:42:55<14:32:43,  3.60s/it]

    Closest words to the center word sword: ['sword', 'tamarin', 'spear', 'excalibur', 'diomedes', 'dresses', 'lyre', 'hilt', 'wakizashi', 'uther']


    Epoch 2/2:  58%|█████▊    | 19899/34343 [4:44:04<2:03:31,  1.95it/s] 

    Epoch 2 Batch 19900 loss: 2.1287410259246826


    Epoch 2/2:  58%|█████▊    | 19900/34343 [4:44:04<2:16:28,  1.76it/s]

    Closest words to the center word three: ['gwh', 'grt', 'pngimage', 'twh', 'kwh', 'cyg', 'mjs', 'hbk', 'sfg', 'lup']


    Epoch 2/2:  58%|█████▊    | 19999/34343 [4:45:15<54:44,  4.37it/s]   

    Epoch 2 Batch 20000 loss: 2.0664329528808594


    Epoch 2/2:  58%|█████▊    | 20000/34343 [4:45:16<1:26:05,  2.78it/s]

    Closest words to the center word ironic: ['ironic', 'epistolary', 'antithesis', 'sarcastic', 'deprecating', 'falsifiability', 'monistic', 'bhagavad', 'arguable', 'retelling']


    Epoch 2/2:  59%|█████▊    | 20099/34343 [4:46:28<50:25,  4.71it/s]   

    Epoch 2 Batch 20100 loss: 2.0766873359680176


    Epoch 2/2:  59%|█████▊    | 20100/34343 [4:46:29<1:19:26,  2.99it/s]

    Closest words to the center word following: ['following', 'gregorian', 'leap', 'rafik', 'fao', 'bloodiest', 'williamite', 'listing', 'preceding', 'pseudocode']


    Epoch 2/2:  59%|█████▉    | 20199/34343 [4:47:57<10:08:52,  2.58s/it]

    Epoch 2 Batch 20200 loss: 2.061518669128418


    Epoch 2/2:  59%|█████▉    | 20200/34343 [4:47:57<7:51:39,  2.00s/it] 

    Closest words to the center word links: ['links', 'webelements', 'wmo', 'icrm', 'external', 'allafrica', 'nonsignatory', 'ifrcs', 'wtoo', 'ifc']


    Epoch 2/2:  59%|█████▉    | 20299/34343 [4:49:07<1:24:28,  2.77it/s] 

    Epoch 2 Batch 20300 loss: 2.0836377143859863


    Epoch 2/2:  59%|█████▉    | 20300/34343 [4:49:08<1:45:04,  2.23it/s]

    Closest words to the center word malaysia: ['malaysia', 'tuvalu', 'allafrica', 'rupee', 'nepal', 'dinar', 'bissau', 'swaziland', 'tajikistan', 'laos']


    Epoch 2/2:  59%|█████▉    | 20399/34343 [4:50:18<49:53,  4.66it/s]   

    Epoch 2 Batch 20400 loss: 2.1162683963775635


    Epoch 2/2:  59%|█████▉    | 20400/34343 [4:50:19<1:24:02,  2.77it/s]

    Closest words to the center word ten: ['ten', 'grt', 'sixty', 'forty', 'fifteen', 'twelve', 'thirty', 'totaling', 'hundred', 'eleven']


    Epoch 2/2:  60%|█████▉    | 20499/34343 [4:51:28<48:52,  4.72it/s]   

    Epoch 2 Batch 20500 loss: 2.0981311798095703


    Epoch 2/2:  60%|█████▉    | 20500/34343 [4:51:29<1:22:32,  2.79it/s]

    Closest words to the center word external: ['webelements', 'icrm', 'external', 'links', 'ifc', 'ifrcs', 'nonsignatory', 'ibrd', 'wmo', 'iom']


    Epoch 2/2:  60%|█████▉    | 20599/34343 [4:52:57<5:29:22,  1.44s/it] 

    Epoch 2 Batch 20600 loss: 2.108396053314209


    Epoch 2/2:  60%|█████▉    | 20600/34343 [4:52:58<4:41:32,  1.23s/it]

    Closest words to the center word u: ['u', 'poz', 'polskiej', 'codepoint', 'rzeczypospolitej', 'rightarrow', 'forall', 'textrm', 'dz', 'ustawa']


    Epoch 2/2:  60%|██████    | 20699/34343 [4:54:06<53:51,  4.22it/s]   

    Epoch 2 Batch 20700 loss: 2.1024155616760254


    Epoch 2/2:  60%|██████    | 20700/34343 [4:54:07<1:31:43,  2.48it/s]

    Closest words to the center word version: ['version', 'ipf', 'versions', 'bwv', 'edition', 'imac', 'remastered', 'athlon', 'cccc', 'wikisource']


    Epoch 2/2:  61%|██████    | 20799/34343 [4:55:16<50:02,  4.51it/s]   

    Epoch 2 Batch 20800 loss: 2.1139543056488037


    Epoch 2/2:  61%|██████    | 20800/34343 [4:55:17<1:18:53,  2.86it/s]

    Closest words to the center word new: ['new', 'york', 'schuster', 'ny', 'giroux', 'farrar', 'ticker', 'bangor', 'dunedin', 'straus']


    Epoch 2/2:  61%|██████    | 20899/34343 [4:56:26<47:12,  4.75it/s]   

    Epoch 2 Batch 20900 loss: 2.1296777725219727


    Epoch 2/2:  61%|██████    | 20900/34343 [4:56:26<1:18:38,  2.85it/s]

    Closest words to the center word allegedly: ['unido', 'wco', 'ifrcs', 'unmibh', 'allegedly', 'hazmi', 'unctad', 'wftu', 'zarqawi', 'aegisthus']


    Epoch 2/2:  61%|██████    | 20999/34343 [4:57:52<1:30:38,  2.45it/s] 

    Epoch 2 Batch 21000 loss: 2.129845142364502


    Epoch 2/2:  61%|██████    | 21000/34343 [4:57:52<1:47:01,  2.08it/s]

    Closest words to the center word black: ['callithrix', 'black', 'leontopithecus', 'tamarin', 'marmoset', 'saguinus', 'eulemur', 'dasyprocta', 'mico', 'capuchin']


    Epoch 2/2:  61%|██████▏   | 21099/34343 [4:59:03<51:58,  4.25it/s]   

    Epoch 2 Batch 21100 loss: 2.1261391639709473


    Epoch 2/2:  61%|██████▏   | 21100/34343 [4:59:03<1:19:53,  2.76it/s]

    Closest words to the center word to: ['to', 'findable', 'inability', 'triadic', 'prev', 'unable', 'able', 'suspend', 'willing', 'unwilling']


    Epoch 2/2:  62%|██████▏   | 21199/34343 [5:00:12<47:33,  4.61it/s]   

    Epoch 2 Batch 21200 loss: 2.128490686416626


    Epoch 2/2:  62%|██████▏   | 21200/34343 [5:00:13<1:19:14,  2.76it/s]

    Closest words to the center word of: ['of', 'ifrcs', 'wtoo', 'nazarene', 'rajonas', 'ifad', 'akan', 'bentheim', 'mycological', 'icrm']


    Epoch 2/2:  62%|██████▏   | 21299/34343 [5:01:38<6:39:27,  1.84s/it] 

    Epoch 2 Batch 21300 loss: 2.0650970935821533


    Epoch 2/2:  62%|██████▏   | 21300/34343 [5:01:39<5:25:22,  1.50s/it]

    Closest words to the center word participated: ['participated', 'starred', 'competed', 'culminated', 'fought', 'embroiled', 'resulted', 'engaged', 'campaigned', 'wco']


    Epoch 2/2:  62%|██████▏   | 21399/34343 [5:02:47<1:16:45,  2.81it/s] 

    Epoch 2 Batch 21400 loss: 2.1159958839416504


    Epoch 2/2:  62%|██████▏   | 21400/34343 [5:02:48<1:37:51,  2.20it/s]

    Closest words to the center word lively: ['lively', 'tikka', 'nightlife', 'natured', 'flavoured', 'morbid', 'thriving', 'sentimental', 'vibrant', 'twinned']


    Epoch 2/2:  63%|██████▎   | 21499/34343 [5:03:57<47:25,  4.51it/s]   

    Epoch 2 Batch 21500 loss: 2.096832513809204


    Epoch 2/2:  63%|██████▎   | 21500/34343 [5:03:58<1:17:58,  2.75it/s]

    Closest words to the center word work: ['work', 'rediscovery', 'doctorate', 'goethe', 'phenomenology', 'virtuosity', 'insights', 'husserl', 'contribution', 'dissertation']


    Epoch 2/2:  63%|██████▎   | 21599/34343 [5:05:09<45:46,  4.64it/s]   

    Epoch 2 Batch 21600 loss: 2.1141245365142822


    Epoch 2/2:  63%|██████▎   | 21600/34343 [5:05:10<1:14:38,  2.85it/s]

    Closest words to the center word celebrated: ['celebrated', 'feast', 'commemorated', 'solstice', 'canonized', 'beltane', 'liturgics', 'imbolc', 'tishri', 'midsummer']


    Epoch 2/2:  63%|██████▎   | 21699/34343 [5:06:36<5:01:08,  1.43s/it] 

    Epoch 2 Batch 21700 loss: 2.095954418182373


    Epoch 2/2:  63%|██████▎   | 21700/34343 [5:06:37<4:11:05,  1.19s/it]

    Closest words to the center word magnificent: ['magnificent', 'borghese', 'yoannis', 'sistine', 'piazza', 'basilica', 'piccadilly', 'sarcophagus', 'erected', 'cathedral']


    Epoch 2/2:  63%|██████▎   | 21799/34343 [5:07:47<58:40,  3.56it/s]   

    Epoch 2 Batch 21800 loss: 2.1025989055633545


    Epoch 2/2:  63%|██████▎   | 21800/34343 [5:07:48<1:19:41,  2.62it/s]

    Closest words to the center word tied: ['tied', 'dealt', 'homeomorphic', 'aligned', 'connected', 'isomorphic', 'awarded', 'reconcile', 'replayed', 'rewarded']


    Epoch 2/2:  64%|██████▍   | 21899/34343 [5:08:57<45:32,  4.55it/s]   

    Epoch 2 Batch 21900 loss: 2.114682912826538


    Epoch 2/2:  64%|██████▍   | 21900/34343 [5:08:58<1:12:55,  2.84it/s]

    Closest words to the center word narrowly: ['narrowly', 'landslide', 'bingu', 'bolsheviks', 'kamenev', 'mwai', 'karzai', 'reelected', 'votes', 'fianna']


    Epoch 2/2:  64%|██████▍   | 21999/34343 [5:10:11<44:19,  4.64it/s]   

    Epoch 2 Batch 22000 loss: 2.0741190910339355


    Epoch 2/2:  64%|██████▍   | 22000/34343 [5:10:12<1:10:52,  2.90it/s]

    Closest words to the center word cost: ['cost', 'kwh', 'gwh', 'throughput', 'expenditures', 'costs', 'bandwidth', 'speeds', 'twh', 'latency']


    Epoch 2/2:  64%|██████▍   | 22099/34343 [5:11:39<2:40:51,  1.27it/s] 

    Epoch 2 Batch 22100 loss: 2.089301586151123


    Epoch 2/2:  64%|██████▍   | 22100/34343 [5:11:39<2:34:29,  1.32it/s]

    Closest words to the center word received: ['received', 'honorary', 'graduated', 'earned', 'doctorate', 'garnered', 'awarded', 'won', 'gained', 'attended']


    Epoch 2/2:  65%|██████▍   | 22199/34343 [5:12:50<49:29,  4.09it/s]   

    Epoch 2 Batch 22200 loss: 2.1115972995758057


    Epoch 2/2:  65%|██████▍   | 22200/34343 [5:12:50<1:17:49,  2.60it/s]

    Closest words to the center word feller: ['leiserson', 'cormen', 'cricketer', 'elke', 'bckgr', 'peckinpah', 'astaire', 'footballer', 'cullen', 'feller']


    Epoch 2/2:  65%|██████▍   | 22299/34343 [5:14:05<44:22,  4.52it/s]   

    Epoch 2 Batch 22300 loss: 2.0672473907470703


    Epoch 2/2:  65%|██████▍   | 22300/34343 [5:14:05<1:12:57,  2.75it/s]

    Closest words to the center word love: ['love', 'totoro', 'lust', 'nnhilde', 'passionate', 'loves', 'vanity', 'thy', 'unrequited', 'wagoner']


    Epoch 2/2:  65%|██████▌   | 22399/34343 [5:15:32<17:31:29,  5.28s/it]

    Epoch 2 Batch 22400 loss: 2.09804368019104


    Epoch 2/2:  65%|██████▌   | 22400/34343 [5:15:33<13:00:50,  3.92s/it]

    Closest words to the center word one: ['gwh', 'kwh', 'pngimage', 'hbk', 'grt', 'mjw', 'twh', 'mjs', 'cyg', 'sfg']


    Epoch 2/2:  66%|██████▌   | 22499/34343 [5:16:43<1:35:35,  2.07it/s] 

    Epoch 2 Batch 22500 loss: 2.1084938049316406


    Epoch 2/2:  66%|██████▌   | 22500/34343 [5:16:44<1:54:48,  1.72it/s]

    Closest words to the center word suspicious: ['suspicious', 'wary', 'intellectually', 'underestimated', 'consummated', 'exacerbated', 'harshly', 'overdose', 'aware', 'incapable']


    Epoch 2/2:  66%|██████▌   | 22599/34343 [5:18:21<59:02,  3.32it/s]   

    Epoch 2 Batch 22600 loss: 2.0844850540161133


    Epoch 2/2:  66%|██████▌   | 22600/34343 [5:18:21<1:25:20,  2.29it/s]

    Closest words to the center word change: ['change', 'desertification', 'enthalpy', 'changes', 'hazardous', 'variability', 'biodiversity', 'findable', 'pollutants', 'drift']


    Epoch 2/2:  66%|██████▌   | 22699/34343 [5:19:58<51:54,  3.74it/s]   

    Epoch 2 Batch 22700 loss: 2.1036622524261475


    Epoch 2/2:  66%|██████▌   | 22700/34343 [5:19:58<1:17:43,  2.50it/s]

    Closest words to the center word said: ['said', 'replied', 'rumoured', 'moneo', 'quipped', 'glad', 'hafsa', 'told', 'remarried', 'believed']


    Epoch 2/2:  66%|██████▋   | 22799/34343 [5:22:12<10:56:27,  3.41s/it]

    Epoch 2 Batch 22800 loss: 2.102323293685913


    Epoch 2/2:  66%|██████▋   | 22800/34343 [5:22:13<8:22:27,  2.61s/it] 

    Closest words to the center word heretic: ['heretic', 'divinely', 'patriarch', 'habakkuk', 'forgiveness', 'prophet', 'absolution', 'marcion', 'arius', 'pius']


    Epoch 2/2:  67%|██████▋   | 22899/34343 [5:23:48<57:46,  3.30it/s]   

    Epoch 2 Batch 22900 loss: 2.1062870025634766


    Epoch 2/2:  67%|██████▋   | 22900/34343 [5:23:49<1:28:42,  2.15it/s]

    Closest words to the center word by: ['by', 'ifrcs', 'leiserson', 'maaouya', 'icrm', 'unido', 'ifc', 'ifad', 'rajonas', 'nonsignatory']


    Epoch 2/2:  67%|██████▋   | 22999/34343 [5:25:26<56:11,  3.37it/s]   

    Epoch 2 Batch 23000 loss: 2.136038303375244


    Epoch 2/2:  67%|██████▋   | 23000/34343 [5:25:27<1:20:00,  2.36it/s]

    Closest words to the center word computers: ['computers', 'cpus', 'microcomputer', 'consoles', 'xt', 'macs', 'peripherals', 'scsi', 'microprocessors', 'powerpc']


    Epoch 2/2:  67%|██████▋   | 23099/34343 [5:27:11<55:20,  3.39it/s]   

    Epoch 2 Batch 23100 loss: 2.081343173980713


    Epoch 2/2:  67%|██████▋   | 23100/34343 [5:27:11<1:20:10,  2.34it/s]

    Closest words to the center word accuses: ['accuses', 'wco', 'unido', 'wftu', 'unmibh', 'wmo', 'unctad', 'weu', 'wtoo', 'wcl']


    Epoch 2/2:  68%|██████▊   | 23199/34343 [5:29:22<2:39:13,  1.17it/s] 

    Epoch 2 Batch 23200 loss: 2.132946491241455


    Epoch 2/2:  68%|██████▊   | 23200/34343 [5:29:23<2:33:21,  1.21it/s]

    Closest words to the center word in: ['in', 'kwh', 'gwh', 'annexes', 'nfc', 'rajonas', 'sfg', 'wct', 'births', 'anh']


    Epoch 2/2:  68%|██████▊   | 23299/34343 [5:31:01<1:22:42,  2.23it/s] 

    Epoch 2 Batch 23300 loss: 2.139177083969116


    Epoch 2/2:  68%|██████▊   | 23300/34343 [5:31:02<1:44:21,  1.76it/s]

    Closest words to the center word graded: ['dolomite', 'beets', 'cassava', 'montane', 'graded', 'lifes', 'broadleaf', 'humid', 'oxides', 'soluble']


    Epoch 2/2:  68%|██████▊   | 23399/34343 [5:32:48<44:29,  4.10it/s]   

    Epoch 2 Batch 23400 loss: 2.080181121826172


    Epoch 2/2:  68%|██████▊   | 23400/34343 [5:32:49<1:17:34,  2.35it/s]

    Closest words to the center word god: ['god', 'yahweh', 'yhwh', 'brahman', 'allah', 'omnipotent', 'vishnu', 'almighty', 'forgiveness', 'incarnate']


    Epoch 2/2:  68%|██████▊   | 23499/34343 [5:34:53<15:37:54,  5.19s/it]

    Epoch 2 Batch 23500 loss: 2.0523595809936523


    Epoch 2/2:  68%|██████▊   | 23500/34343 [5:34:54<11:53:46,  3.95s/it]

    Closest words to the center word spots: ['spots', 'leontopithecus', 'isosceles', 'mantled', 'saguinus', 'tamarin', 'tailed', 'callithrix', 'grt', 'hexagonal']


    Epoch 2/2:  69%|██████▊   | 23599/34343 [5:36:42<2:13:40,  1.34it/s] 

    Epoch 2 Batch 23600 loss: 2.092559814453125


    Epoch 2/2:  69%|██████▊   | 23600/34343 [5:36:43<2:14:33,  1.33it/s]

    Closest words to the center word need: ['need', 'intend', 'pluperfect', 'newnode', 'incentive', 'posteriori', 'want', 'factum', 'require', 'disprove']


    Epoch 2/2:  69%|██████▉   | 23699/34343 [5:38:25<1:01:12,  2.90it/s] 

    Epoch 2 Batch 23700 loss: 2.08243465423584


    Epoch 2/2:  69%|██████▉   | 23700/34343 [5:38:26<1:51:52,  1.59it/s]

    Closest words to the center word electrolysis: ['electrolysis', 'halides', 'hydride', 'alkoxide', 'anode', 'nitric', 'phosphorylation', 'cathode', 'pyruvate', 'carbonyl']


    Epoch 2/2:  69%|██████▉   | 23799/34343 [5:40:05<52:40,  3.34it/s]   

    Epoch 2 Batch 23800 loss: 2.127439498901367


    Epoch 2/2:  69%|██████▉   | 23800/34343 [5:40:05<1:14:04,  2.37it/s]

    Closest words to the center word benz: ['benz', 'daimler', 'audi', 'kwh', 'mercedes', 'gwh', 'bugatti', 'bmw', 'aston', 'twh']


    Epoch 2/2:  70%|██████▉   | 23899/34343 [5:42:18<9:17:21,  3.20s/it] 

    Epoch 2 Batch 23900 loss: 2.15342116355896


    Epoch 2/2:  70%|██████▉   | 23900/34343 [5:42:18<7:08:55,  2.46s/it]

    Closest words to the center word bhutan: ['bhutan', 'allafrica', 'rupee', 'kitts', 'tuvalu', 'swaziland', 'dinar', 'escudo', 'tajikistan', 'chungcheong']


    Epoch 2/2:  70%|██████▉   | 23999/34343 [5:44:02<1:31:50,  1.88it/s] 

    Epoch 2 Batch 24000 loss: 2.0733962059020996


    Epoch 2/2:  70%|██████▉   | 24000/34343 [5:44:03<1:42:34,  1.68it/s]

    Closest words to the center word flutes: ['flutes', 'tremolo', 'basses', 'cellos', 'harmonicas', 'woodwind', 'clarinets', 'clarinet', 'diatonic', 'contrabass']


    Epoch 2/2:  70%|███████   | 24099/34343 [5:45:36<46:28,  3.67it/s]   

    Epoch 2 Batch 24100 loss: 2.1080574989318848


    Epoch 2/2:  70%|███████   | 24100/34343 [5:45:36<1:17:58,  2.19it/s]

    Closest words to the center word weaknesses: ['weaknesses', 'irritability', 'strengths', 'eyesight', 'melee', 'flaws', 'assumptions', 'tardive', 'inhibitors', 'myocardial']


    Epoch 2/2:  70%|███████   | 24199/34343 [5:47:17<42:58,  3.93it/s]   

    Epoch 2 Batch 24200 loss: 2.142861843109131


    Epoch 2/2:  70%|███████   | 24200/34343 [5:47:18<1:38:53,  1.71it/s]

    Closest words to the center word scroll: ['scroll', 'polycarbonate', 'fretboard', 'platters', 'socket', 'rotor', 'cylindrical', 'keypad', 'fret', 'tubing']


    Epoch 2/2:  71%|███████   | 24299/34343 [5:49:17<6:47:51,  2.44s/it] 

    Epoch 2 Batch 24300 loss: 2.1393821239471436


    Epoch 2/2:  71%|███████   | 24300/34343 [5:49:18<5:29:12,  1.97s/it]

    Closest words to the center word states: ['states', 'federated', 'emirates', 'micronesia', 'united', 'tuvalu', 'nations', 'seceded', 'oecs', 'eapc']


    Epoch 2/2:  71%|███████   | 24399/34343 [5:50:51<1:09:01,  2.40it/s] 

    Epoch 2 Batch 24400 loss: 2.1135880947113037


    Epoch 2/2:  71%|███████   | 24400/34343 [5:50:52<1:25:38,  1.94it/s]

    Closest words to the center word under: ['under', 'auspices', 'gpl', 'mamluk', 'durrani', 'mughals', 'fatimid', 'salih', 'maaouya', 'tokugawa']


    Epoch 2/2:  71%|███████▏  | 24499/34343 [5:52:26<49:12,  3.33it/s]   

    Epoch 2 Batch 24500 loss: 2.0778377056121826


    Epoch 2/2:  71%|███████▏  | 24500/34343 [5:52:27<1:10:20,  2.33it/s]

    Closest words to the center word changed: ['changed', 'shifted', 'reverted', 'findable', 'faded', 'altered', 'ratified', 'traced', 'renamed', 'recovered']


    Epoch 2/2:  72%|███████▏  | 24599/34343 [5:54:01<48:12,  3.37it/s]   

    Epoch 2 Batch 24600 loss: 2.0834269523620605


    Epoch 2/2:  72%|███████▏  | 24600/34343 [5:54:02<1:11:41,  2.27it/s]

    Closest words to the center word brown: ['callithrix', 'eulemur', 'brown', 'leontopithecus', 'tamias', 'tamarin', 'aquilegia', 'mesoplodon', 'haliotis', 'beaked']


    Epoch 2/2:  72%|███████▏  | 24699/34343 [5:56:09<4:14:31,  1.58s/it] 

    Epoch 2 Batch 24700 loss: 2.1234207153320312


    Epoch 2/2:  72%|███████▏  | 24700/34343 [5:56:09<3:39:53,  1.37s/it]

    Closest words to the center word had: ['had', 'hadn', 'fathered', 'aegisthus', 'thyestes', 'hath', 'hasn', 'clytemnestra', 'has', 'betrothed']


    Epoch 2/2:  72%|███████▏  | 24799/34343 [5:57:42<55:55,  2.84it/s]   

    Epoch 2 Batch 24800 loss: 2.1185073852539062


    Epoch 2/2:  72%|███████▏  | 24800/34343 [5:57:42<1:10:05,  2.27it/s]

    Closest words to the center word card: ['card', 'cards', 'nfc', 'debit', 'afc', 'betting', 'playoffs', 'bobble', 'newnode', 'gba']


    Epoch 2/2:  73%|███████▎  | 24899/34343 [5:59:12<42:06,  3.74it/s]   

    Epoch 2 Batch 24900 loss: 2.102382183074951


    Epoch 2/2:  73%|███████▎  | 24900/34343 [5:59:13<1:05:01,  2.42it/s]

    Closest words to the center word in: ['in', 'kwh', 'gwh', 'anh', 'annexes', 'births', 'throughout', 'anterselva', 'twh', 'availabilitymales']


    Epoch 2/2:  73%|███████▎  | 24999/34343 [6:00:43<1:32:59,  1.67it/s] 

    Epoch 2 Batch 25000 loss: 2.1095993518829346


    Epoch 2/2:  73%|███████▎  | 25000/34343 [6:01:05<17:56:32,  6.91s/it]

    Closest words to the center word years: ['years', 'grt', 'decades', 'months', 'servicemales', 'weeks', 'totaling', 'availabilitymales', 'injures', 'lifes']


    Epoch 2/2:  73%|███████▎  | 25099/34343 [6:02:33<1:52:50,  1.37it/s] 

    Epoch 2 Batch 25100 loss: 2.1300365924835205


    Epoch 2/2:  73%|███████▎  | 25100/34343 [6:02:34<1:52:17,  1.37it/s]

    Closest words to the center word hscsd: ['bbbb', 'nonsignatory', 'iom', 'wavelet', 'cccc', 'asynchronous', 'ifrcs', 'opcw', 'bilinear', 'sys']


    Epoch 2/2:  73%|███████▎  | 25199/34343 [6:04:00<43:30,  3.50it/s]   

    Epoch 2 Batch 25200 loss: 2.0810372829437256


    Epoch 2/2:  73%|███████▎  | 25200/34343 [6:04:02<1:36:25,  1.58it/s]

    Closest words to the center word are: ['are', 'were', 'aren', 'unicellular', 'protists', 'celled', 'differ', 'rectangles', 'consist', 'solutes']


    Epoch 2/2:  74%|███████▎  | 25299/34343 [6:05:35<43:32,  3.46it/s]   

    Epoch 2 Batch 25300 loss: 2.08467960357666


    Epoch 2/2:  74%|███████▎  | 25300/34343 [6:05:37<1:24:27,  1.78it/s]

    Closest words to the center word checking: ['checking', 'chaining', 'newnode', 'prev', 'asynchronous', 'firstnode', 'findable', 'avl', 'pointers', 'macros']


    Epoch 2/2:  74%|███████▍  | 25399/34343 [6:07:31<13:00:05,  5.23s/it]

    Epoch 2 Batch 25400 loss: 2.134082794189453


    Epoch 2/2:  74%|███████▍  | 25400/34343 [6:07:32<9:37:47,  3.88s/it] 

    Closest words to the center word constitution: ['constitution', 'bicameral', 'ratified', 'unicameral', 'legislature', 'referendum', 'constitutional', 'amendment', 'monarchy', 'eldr']


    Epoch 2/2:  74%|███████▍  | 25499/34343 [6:09:05<1:06:59,  2.20it/s] 

    Epoch 2 Batch 25500 loss: 2.1233327388763428


    Epoch 2/2:  74%|███████▍  | 25500/34343 [6:09:06<1:57:29,  1.25it/s]

    Closest words to the center word project: ['nonsignatory', 'ifrcs', 'ifad', 'project', 'ifc', 'iom', 'sourceforge', 'idb', 'looksmart', 'gnat']


    Epoch 2/2:  75%|███████▍  | 25599/34343 [6:10:40<44:31,  3.27it/s]   

    Epoch 2 Batch 25600 loss: 2.1135940551757812


    Epoch 2/2:  75%|███████▍  | 25600/34343 [6:10:40<1:05:59,  2.21it/s]

    Closest words to the center word valid: ['valid', 'bijective', 'provable', 'metrizable', 'ponens', 'refutable', 'unital', 'differentiable', 'surjective', 'enumerable']


    Epoch 2/2:  75%|███████▍  | 25699/34343 [6:12:17<39:27,  3.65it/s]   

    Epoch 2 Batch 25700 loss: 2.1290810108184814


    Epoch 2/2:  75%|███████▍  | 25700/34343 [6:12:18<1:14:39,  1.93it/s]

    Closest words to the center word bobby: ['finalist', 'bobby', 'kreutzmann', 'satchel', 'bruins', 'footballer', 'bourque', 'lesh', 'foxx', 'darrell']


    Epoch 2/2:  75%|███████▌  | 25799/34343 [6:14:11<2:23:53,  1.01s/it] 

    Epoch 2 Batch 25800 loss: 2.092529535293579


    Epoch 2/2:  75%|███████▌  | 25800/34343 [6:14:11<2:12:11,  1.08it/s]

    Closest words to the center word a: ['a', 'agave', 'pluriform', 'globicephala', 'lemur', 'keying', 'insubstantial', 'groupoid', 'newnode', 'callithrix']


    Epoch 2/2:  75%|███████▌  | 25899/34343 [6:15:40<43:13,  3.26it/s]   

    Epoch 2 Batch 25900 loss: 2.114318370819092


    Epoch 2/2:  75%|███████▌  | 25900/34343 [6:15:41<1:00:17,  2.33it/s]

    Closest words to the center word they: ['they', 'capybaras', 'we', 'you', 'liars', 'aediles', 'gesserit', 'feeders', 'matres', 'findable']


    Epoch 2/2:  76%|███████▌  | 25999/34343 [6:17:15<35:20,  3.94it/s]   

    Epoch 2 Batch 26000 loss: 2.0930535793304443


    Epoch 2/2:  76%|███████▌  | 26000/34343 [6:17:16<52:30,  2.65it/s]

    Closest words to the center word ethics: ['ethics', 'cognitivism', 'egoism', 'intuitionism', 'ethical', 'jurisprudence', 'epistemology', 'reductionism', 'objectivism', 'relativism']


    Epoch 2/2:  76%|███████▌  | 26099/34343 [6:19:10<15:07:11,  6.60s/it]

    Epoch 2 Batch 26100 loss: 2.076605796813965


    Epoch 2/2:  76%|███████▌  | 26100/34343 [6:19:11<11:09:00,  4.87s/it]

    Closest words to the center word june: ['june', 'gwh', 'january', 'april', 'july', 'february', 'august', 'november', 'kwh', 'december']


    Epoch 2/2:  76%|███████▋  | 26199/34343 [6:20:58<2:05:28,  1.08it/s] 

    Epoch 2 Batch 26200 loss: 2.129329204559326


    Epoch 2/2:  76%|███████▋  | 26200/34343 [6:20:59<1:57:37,  1.15it/s]

    Closest words to the center word studies: ['studies', 'ethologists', 'therapies', 'maxillofacial', 'subfields', 'anthropology', 'psychologists', 'informatics', 'critiques', 'physiology']


    Epoch 2/2:  77%|███████▋  | 26299/34343 [6:22:31<47:39,  2.81it/s]   

    Epoch 2 Batch 26300 loss: 2.129953384399414


    Epoch 2/2:  77%|███████▋  | 26300/34343 [6:22:32<1:04:33,  2.08it/s]

    Closest words to the center word the: ['ifrcs', 'pluriform', 'nonsignatory', 'tamarin', 'leontopithecus', 'ifad', 'the', 'ifc', 'iom', 'saguinus']


    Epoch 2/2:  77%|███████▋  | 26399/34343 [6:23:59<33:27,  3.96it/s]   

    Epoch 2 Batch 26400 loss: 2.112628936767578


    Epoch 2/2:  77%|███████▋  | 26400/34343 [6:23:59<52:50,  2.51it/s]

    Closest words to the center word told: ['told', 'swaim', 'tells', 'moneo', 'replied', 'complains', 'distraught', 'joked', 'afraid', 'hafsa']


    Epoch 2/2:  77%|███████▋  | 26499/34343 [6:26:01<11:06:50,  5.10s/it]

    Epoch 2 Batch 26500 loss: 2.110123872756958


    Epoch 2/2:  77%|███████▋  | 26500/34343 [6:26:02<8:16:45,  3.80s/it] 

    Closest words to the center word brigades: ['brigades', 'battalions', 'battalion', 'brigade', 'ifrcs', 'einsatzgruppen', 'waffen', 'regiment', 'peacekeeping', 'iom']


    Epoch 2/2:  77%|███████▋  | 26599/34343 [6:27:37<1:26:41,  1.49it/s] 

    Epoch 2 Batch 26600 loss: 2.069441795349121


    Epoch 2/2:  77%|███████▋  | 26600/34343 [6:27:38<1:34:16,  1.37it/s]

    Closest words to the center word mathematician: ['mathematician', 'physicist', 'astronomer', 'mineralogist', 'physiologist', 'zoologist', 'philosopher', 'xaver', 'neurologist', 'philologist']


    Epoch 2/2:  78%|███████▊  | 26699/34343 [6:29:27<42:04,  3.03it/s]   

    Epoch 2 Batch 26700 loss: 2.099580764770508


    Epoch 2/2:  78%|███████▊  | 26700/34343 [6:29:27<57:48,  2.20it/s]

    Closest words to the center word probability: ['probability', 'cardinality', 'polynomial', 'mathfrak', 'hamiltonian', 'polynomials', 'entropy', 'lebesgue', 'integrable', 'widehat']


    Epoch 2/2:  78%|███████▊  | 26799/34343 [6:30:58<36:36,  3.44it/s]   

    Epoch 2 Batch 26800 loss: 2.114670515060425


    Epoch 2/2:  78%|███████▊  | 26800/34343 [6:30:59<54:35,  2.30it/s]

    Closest words to the center word fact: ['fact', 'provable', 'epimenides', 'unsurprising', 'rooted', 'unital', 'testable', 'eukaryotes', 'believe', 'misconception']


    Epoch 2/2:  78%|███████▊  | 26899/34343 [6:33:00<6:26:43,  3.12s/it] 

    Epoch 2 Batch 26900 loss: 2.118748188018799


    Epoch 2/2:  78%|███████▊  | 26900/34343 [6:33:01<5:10:20,  2.50s/it]

    Closest words to the center word the: ['pluriform', 'ifrcs', 'leontopithecus', 'tamarin', 'the', 'nonsignatory', 'gangetic', 'saguinus', 'ifad', 'callithrix']


    Epoch 2/2:  79%|███████▊  | 26999/34343 [6:34:20<44:27,  2.75it/s]   

    Epoch 2 Batch 27000 loss: 2.0952489376068115


    Epoch 2/2:  79%|███████▊  | 27000/34343 [6:34:21<54:40,  2.24it/s]

    Closest words to the center word own: ['own', 'findable', 'respective', 'dealings', 'abilities', 'humility', 'egoism', 'countrymen', 'tireless', 'teg']


    Epoch 2/2:  79%|███████▉  | 27099/34343 [6:35:32<27:25,  4.40it/s]   

    Epoch 2 Batch 27100 loss: 2.113372802734375


    Epoch 2/2:  79%|███████▉  | 27100/34343 [6:35:33<42:06,  2.87it/s]

    Closest words to the center word left: ['left', 'otimes', 'arctan', 'right', 'cdot', 'vec', 'rangle', 'kx', 'cdots', 'operatorname']


    Epoch 2/2:  79%|███████▉  | 27199/34343 [6:36:44<25:21,  4.69it/s]   

    Epoch 2 Batch 27200 loss: 2.0929782390594482


    Epoch 2/2:  79%|███████▉  | 27200/34343 [6:36:45<43:00,  2.77it/s]

    Closest words to the center word bit: ['bit', 'kbit', 'pngimage', 'xt', 'mbit', 'megabyte', 'kb', 'bytes', 'kilobytes', 'mjs']


    Epoch 2/2:  79%|███████▉  | 27299/34343 [6:38:12<2:47:55,  1.43s/it] 

    Epoch 2 Batch 27300 loss: 2.0733766555786133


    Epoch 2/2:  79%|███████▉  | 27300/34343 [6:38:13<2:30:21,  1.28s/it]

    Closest words to the center word suitable: ['suitable', 'optimized', 'feedstock', 'computationally', 'unsuitable', 'transshipment', 'consuming', 'suited', 'glycogen', 'metrizable']


    Epoch 2/2:  80%|███████▉  | 27399/34343 [6:39:52<50:22,  2.30it/s]   

    Epoch 2 Batch 27400 loss: 2.10170841217041


    Epoch 2/2:  80%|███████▉  | 27400/34343 [6:39:53<59:28,  1.95it/s]

    Closest words to the center word cuttlefish: ['cuttlefish', 'beets', 'cassava', 'infraorder', 'lepidoptera', 'waterborne', 'secrete', 'lemurs', 'suborder', 'soybeans']


    Epoch 2/2:  80%|████████  | 27499/34343 [6:41:49<30:21,  3.76it/s]   

    Epoch 2 Batch 27500 loss: 2.0607285499572754


    Epoch 2/2:  80%|████████  | 27500/34343 [6:41:50<46:29,  2.45it/s]

    Closest words to the center word equatorial: ['equatorial', 'bissau', 'allafrica', 'verde', 'faso', 'gambia', 'landlocked', 'ivoire', 'escudo', 'brazzaville']


    Epoch 2/2:  80%|████████  | 27599/34343 [6:43:28<29:33,  3.80it/s]   

    Epoch 2 Batch 27600 loss: 2.1012046337127686


    Epoch 2/2:  80%|████████  | 27600/34343 [6:43:29<45:18,  2.48it/s]

    Closest words to the center word reliability: ['reliability', 'throughput', 'flexibility', 'usability', 'latency', 'efficiency', 'recompilation', 'compressive', 'bandwidth', 'tensile']


    Epoch 2/2:  81%|████████  | 27699/34343 [6:45:40<2:37:38,  1.42s/it] 

    Epoch 2 Batch 27700 loss: 2.1068646907806396


    Epoch 2/2:  81%|████████  | 27700/34343 [6:45:40<2:13:49,  1.21s/it]

    Closest words to the center word scientific: ['scientific', 'cognitivism', 'anthropological', 'empirical', 'anthropic', 'nativist', 'epistemological', 'falsifiable', 'psychoanalytic', 'pseudoscience']


    Epoch 2/2:  81%|████████  | 27799/34343 [6:47:19<26:54,  4.05it/s]   

    Epoch 2 Batch 27800 loss: 2.085338592529297


    Epoch 2/2:  81%|████████  | 27800/34343 [6:47:20<42:33,  2.56it/s]

    Closest words to the center word gospel: ['gospel', 'gospels', 'epistle', 'barnabas', 'epistles', 'ephesians', 'apostles', 'hebrews', 'jude', 'philippians']


    Epoch 2/2:  81%|████████  | 27899/34343 [6:48:51<41:51,  2.57it/s]   

    Epoch 2 Batch 27900 loss: 2.1243529319763184


    Epoch 2/2:  81%|████████  | 27900/34343 [6:48:52<59:54,  1.79it/s]

    Closest words to the center word centres: ['centres', 'iom', 'rajonas', 'centers', 'destinations', 'outskirts', 'tourist', 'shopping', 'wangfujing', 'ifrcs']


    Epoch 2/2:  82%|████████▏ | 27999/34343 [6:50:48<14:27:43,  8.21s/it]

    Epoch 2 Batch 28000 loss: 2.1093854904174805


    Epoch 2/2:  82%|████████▏ | 28000/34343 [6:50:50<11:08:05,  6.32s/it]

    Closest words to the center word sense: ['sense', 'cognitivism', 'physicalism', 'prescriptive', 'priori', 'soundness', 'reductionism', 'formalization', 'propositional', 'posteriori']


    Epoch 2/2:  82%|████████▏ | 28099/34343 [6:52:21<38:21,  2.71it/s]   

    Epoch 2 Batch 28100 loss: 2.108693838119507


    Epoch 2/2:  82%|████████▏ | 28100/34343 [6:52:22<49:26,  2.10it/s]

    Closest words to the center word rather: ['rather', 'thicker', 'fairer', 'semivowel', 'nonexistence', 'less', 'aesthetically', 'overly', 'sweeter', 'relying']


    Epoch 2/2:  82%|████████▏ | 28199/34343 [6:53:54<27:50,  3.68it/s]   

    Epoch 2 Batch 28200 loss: 2.105180501937866


    Epoch 2/2:  82%|████████▏ | 28200/34343 [6:53:55<42:41,  2.40it/s]

    Closest words to the center word two: ['gwh', 'grt', 'twh', 'kwh', 'hbk', 'cyg', 'sfg', 'pngimage', 'pmid', 'mjs']


    Epoch 2/2:  82%|████████▏ | 28299/34343 [6:55:27<22:54,  4.40it/s]   

    Epoch 2 Batch 28300 loss: 2.079437255859375


    Epoch 2/2:  82%|████████▏ | 28300/34343 [6:55:28<37:24,  2.69it/s]

    Closest words to the center word rejected: ['rejected', 'denounced', 'contradicted', 'persecuted', 'affirmed', 'ratified', 'criticized', 'dismissed', 'rejects', 'accepted']


    Epoch 2/2:  83%|████████▎ | 28399/34343 [6:57:26<2:08:37,  1.30s/it] 

    Epoch 2 Batch 28400 loss: 2.10756516456604


    Epoch 2/2:  83%|████████▎ | 28400/34343 [6:57:27<2:09:23,  1.31s/it]

    Closest words to the center word changed: ['changed', 'reverted', 'shifted', 'findable', 'altered', 'renamed', 'ported', 'abolished', 'faded', 'ratified']


    Epoch 2/2:  83%|████████▎ | 28499/34343 [6:59:01<34:51,  2.79it/s]   

    Epoch 2 Batch 28500 loss: 2.1471188068389893


    Epoch 2/2:  83%|████████▎ | 28500/34343 [6:59:01<46:10,  2.11it/s]

    Closest words to the center word republican: ['republican', 'pluriform', 'sinn', 'presidential', 'bingu', 'democrat', 'eldr', 'reelected', 'senator', 'sandinista']


    Epoch 2/2:  83%|████████▎ | 28599/34343 [7:00:23<27:12,  3.52it/s]  

    Epoch 2 Batch 28600 loss: 2.1108620166778564


    Epoch 2/2:  83%|████████▎ | 28600/34343 [7:00:24<41:03,  2.33it/s]

    Closest words to the center word george: ['leiserson', 'george', 'earl', 'cormen', 'marquess', 'eldridge', 'custis', 'rivest', 'frideric', 'hervey']


    Epoch 2/2:  83%|████████▎ | 28671/34343 [7:01:28<26:28,  3.57it/s]   Exception ignored in: <bound method IPythonKernel._clean_thread_parent_frames of <ipykernel.ipkernel.IPythonKernel object at 0x104ea8110>>
    Traceback (most recent call last):
      File "/Users/ravi.mandliya/.pyenv/versions/3.11.7/envs/blog/lib/python3.11/site-packages/ipykernel/ipkernel.py", line 775, in _clean_thread_parent_frames
        def _clean_thread_parent_frames(
    
    KeyboardInterrupt: 
    Epoch 2/2:  84%|████████▎ | 28691/34343 [7:01:56<1:23:07,  1.13it/s] 



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[69], line 1
    ----> 1 word2vec.train(text, batch_size=2048)


    Cell In[67], line 62, in Word2Vec.train(self, text, batch_size)
         60 # Create tqdm wrapper inside the loop for each epoch
         61 tqdm_dataloader = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{self.num_epochs}")
    ---> 62 for idx, (target, context, negs) in enumerate(tqdm_dataloader):
         63   self.optimizer.zero_grad()
         64   pos_score, neg_score = self.model(target, context, negs)


    File ~/.pyenv/versions/3.11.7/envs/blog/lib/python3.11/site-packages/tqdm/std.py:1181, in tqdm.__iter__(self)
       1178 time = self._time
       1180 try:
    -> 1181     for obj in iterable:
       1182         yield obj
       1183         # Update and possibly print the progressbar.
       1184         # Note: does not call self.update(1) for speed optimisation.


    File ~/.pyenv/versions/3.11.7/envs/blog/lib/python3.11/site-packages/torch/utils/data/dataloader.py:631, in _BaseDataLoaderIter.__next__(self)
        628 if self._sampler_iter is None:
        629     # TODO(https://github.com/pytorch/pytorch/issues/76750)
        630     self._reset()  # type: ignore[call-arg]
    --> 631 data = self._next_data()
        632 self._num_yielded += 1
        633 if self._dataset_kind == _DatasetKind.Iterable and \
        634         self._IterableDataset_len_called is not None and \
        635         self._num_yielded > self._IterableDataset_len_called:


    File ~/.pyenv/versions/3.11.7/envs/blog/lib/python3.11/site-packages/torch/utils/data/dataloader.py:675, in _SingleProcessDataLoaderIter._next_data(self)
        673 def _next_data(self):
        674     index = self._next_index()  # may raise StopIteration
    --> 675     data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        676     if self._pin_memory:
        677         data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)


    File ~/.pyenv/versions/3.11.7/envs/blog/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py:51, in _MapDatasetFetcher.fetch(self, possibly_batched_index)
         49         data = self.dataset.__getitems__(possibly_batched_index)
         50     else:
    ---> 51         data = [self.dataset[idx] for idx in possibly_batched_index]
         52 else:
         53     data = self.dataset[possibly_batched_index]


    File ~/.pyenv/versions/3.11.7/envs/blog/lib/python3.11/site-packages/torch/utils/data/_utils/fetch.py:51, in <listcomp>(.0)
         49         data = self.dataset.__getitems__(possibly_batched_index)
         50     else:
    ---> 51         data = [self.dataset[idx] for idx in possibly_batched_index]
         52 else:
         53     data = self.dataset[possibly_batched_index]


    Cell In[63], line 91, in Word2VecDataset.__getitem__(self, idx)
         88 target, context = self.pairs[idx]
         90 # Generate negative samples on-the-fly
    ---> 91 negs = torch.multinomial(
         92   self.negative_sample_weights,
         93   self.negative_sample_counts,
         94   replacement=True
         95 )
         97 return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long), negs


    KeyboardInterrupt: 


```python
# print random words in the vocabulary
random.sample(words, 10)

```




    ['after',
     'took',
     'asserts',
     'russian',
     'in',
     'performed',
     'day',
     'two',
     'serving',
     'conference']



```python
word2vec.most_similar("happiness")
```




    ['happiness',
     'goodness',
     'repent',
     'compassion',
     'righteousness',
     'atonement',
     'forgiveness',
     'sinful',
     'immanent',
     'rationality']


