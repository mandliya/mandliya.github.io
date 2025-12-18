---
layout: post
title: 'From Words to Meaning: Implementing Word2Vec from Scratch'
date: '2025-12-17 22:14:57 '
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
- The first term $$\log \sigma((v'_{w_O})^{\top} v_{w_I})$$ maximizes the probability that the actual context word $w_O$ appears with $w_I$
- The second term $$\sum_{i=1}^{k} \log \sigma(-(v'_{w_i})^{\top} v_{w_I})$$ minimizes the probability that $k$ randomly sampled words $w_i$ appear with $w_I$

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
\mathcal{L} = -\log(\text{pos_score}) - \sum_{i=1}^{k} \log(1 - \text{neg_score}_i)
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

    ....

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

## Interpreting the Learned Embeddings

We now have a full Word2Vec pipeline: we built a training dataset from raw text, implemented the Skip‑Gram model with negative sampling, and trained it on the `text8` corpus. The most important question is: **did we actually learn something meaningful?**

One simple way to probe the embedding space is to look at nearest neighbours. For example, when we query:

word2vec.most_similar("happiness")we get neighbours like:

- `goodness`
- `compassion`
- `righteousness`
- `forgiveness`

These are not just random words: they live in a similar semantic region of the corpus, often appearing in religious, ethical, or emotional contexts. That’s exactly the kind of structure Word2Vec is designed to uncover: words that occur in similar contexts end up close together in the embedding space.

In practice, you would typically:

- **Inspect multiple probes** (e.g., `["dog", "king", "economy", "algorithm"]`) to build intuition for what the model has learned.
- **Use dimensionality reduction** (PCA or t‑SNE) to visualize clusters of related words (animals, countries, professions, etc.).
- **Plug the embeddings into downstream models** (classifiers, sequence models, recommendation systems) as dense semantic features.

## Limitations and Practical Considerations

Our implementation is faithful to the original Word2Vec formulation, but there are many practical details that affect quality:

- **Training time and corpus size**: We trained for only a couple of epochs on `text8`. Larger corpora and longer training generally yield much better embeddings.
- **Hyperparameters**: Window size, embedding dimension, number of negative samples, subsampling threshold, and learning rate can all significantly change the geometry of the embedding space.
- **Vocabulary trimming**: Dropping very rare words (via `min_count`) reduces noise and keeps training efficient, at the cost of losing some tail vocabulary.

In modern NLP pipelines, you would rarely train Word2Vec from scratch in production. Instead, you might:

- Use **pretrained static embeddings** (e.g., GloVe, fastText).
- Or, more commonly, rely on **contextual embeddings** from transformer models (BERT, GPT, etc.).

Still, Word2Vec remains a powerful and interpretable starting point for understanding how neural networks can discover structure in language.

## Summary

In this post, we:

1. Motivated the need for dense word representations and introduced the distributional hypothesis.
2. Derived the Skip‑Gram objective and showed how the softmax leads to a computational bottleneck.
3. Introduced **negative sampling** as an efficient approximation, and implemented it from scratch in PyTorch.
4. Built a dataset pipeline that performs subsampling, generates context pairs, and draws negative examples on‑the‑fly.
5. Trained a Word2Vec model on real text and inspected the resulting embeddings.

If you’d like to go further, good next steps include:

- Visualizing embeddings for different word groups.
- Experimenting with hyperparameters and corpus size.
- Comparing static Word2Vec embeddings with contextual embeddings from modern transformer models.

By the end of this journey, you should have a solid, end‑to‑end understanding of how Word2Vec works under the hood—and a working implementation you can adapt to your own datasets.
