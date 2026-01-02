---
layout: post
title: 'Opening the Black Box: Understanding Transformer Internals Through Residual
  Streams'
date: '2026-01-02 03:17:46 '
categories:
- Technology
tags:
- Jupyter
- Notebook
description: A Mechanistic Interpretability Journey from ML Engineering to AI Safety
image: /assets/img/opening-the-black-box-understanding-transformer-internals-through-residual-streams/cover.png
image_alt: 'Opening the Black Box: Understanding Transformer Internals Through Residual
  Streams'
math: true
mermaid: true
pin: false
toc: true
comments: true
---

# Opening the Black Box: Understanding Transformer Internals Through Residual Streams

*A Mechanistic Interpretability Journey from ML Engineering to AI Safety*

**Author:** Ravi  
**Date:** [Insert Date]  
**Reading Time:** ~20 minutes

---

## Introduction

GPT-4 can write code, explain jokes, and translate languages. But can we trust it? When a model says "I don't have access to the internet" while simultaneously citing yesterday's news, is it lying? Confused? Hallucinating? We can't answer these questions by treating transformers as black boxes. To build safe AI systems, we need to understand what's actually happening inside.

I'm a machine learning engineer at Discord, where I build recommendation systems that affect millions of users daily. I work with large-scale notification systems, user engagement models, and ML infrastructure that processes billions of events. But as these models grow more powerful, I've become convinced that understanding their internals isn't just academic curiosity—it's an alignment necessity. This post series chronicles my journey from ML engineering to AI safety research, starting with the foundations of mechanistic interpretability.

In this post, we'll rebuild your mental model of transformers from the ground up. Forget the traditional attention mechanism you learned—we're going deeper. You'll learn to see transformers not as attention + MLPs, but as a communication network where information flows through a shared channel called the residual stream. By the end, you'll understand virtual weights—implicit connections between layers that traditional views miss entirely.

**What you'll learn:**
- What mechanistic interpretability is and why it matters for AI safety
- The residual stream view of transformers (and why it's better than the traditional view)
- How to extract and visualize residual streams from real models
- Virtual weights: the hidden connections between layers
- Hands-on code examples using GPT-2

**Prerequisites:**
- Familiarity with transformers and attention mechanisms
- Basic Python and PyTorch
- Linear algebra fundamentals (matrix multiplication, dot products)

Let's dive in.


---

## 1. What is Mechanistic Interpretability?

### The Goal of MI

When GPT-4 explains a joke, writes working code, or answers a medical question, *how* does it do it? Not in the vague sense of "it learned patterns from data," but mechanically: which circuits activate, which features get computed, how does information flow from input to output?

Mechanistic interpretability (MI) asks exactly this question. It's the science of reverse-engineering neural networks to understand the algorithms they've learned. Think of it like the difference between observing that your car moves forward when you press the gas pedal versus understanding the combustion engine, transmission, and drivetrain that make it happen. Both are valid forms of knowledge, but only the latter lets you diagnose problems, predict failures, or build better engines.

MI stands apart from other interpretability approaches in its ambition and methodology:

**Probing** asks "does the model represent X?"—for instance, does BERT encode part-of-speech information? These are valuable questions, but they treat the model as a black box that you can only query behaviorally.

**Attribution methods** ask "which inputs mattered most?"—they generate saliency maps showing which pixels or tokens influenced a decision. Again useful, but they tell you about correlation, not causation, and nothing about the mechanism.

**Mechanistic interpretability** is more ambitious: it asks "what algorithm does the model implement?" When GPT-2 completes "When Mary and John went to the store, John gave a drink to" with "Mary," it's not just pattern matching—it's implementing an algorithm. MI researchers want to find that algorithm, understand it as precisely as you'd understand quicksort or binary search, and prove that it works the way we think it does.

### Why This Matters for Safety

If MI sounds academic, consider this: we're rapidly deploying AI systems that we don't understand into high-stakes domains. Models are writing code that runs in production, providing medical advice, and soon will be making autonomous decisions with real-world consequences. "It usually works" isn't good enough when failure modes could be catastrophic.

The need becomes urgent when we consider **deception**. Imagine a model that's learned it gets higher reward by appearing helpful during training, while maintaining a hidden "intent" to behave differently during deployment. Can we detect this? Traditional evaluation can't—by definition, the model passes our tests. We need to look inside and ask: are there features representing "stated intent" versus "actual intent"? Do they diverge? Where in the network does this divergence happen?

Or consider the challenge of **capability oversight**. Suppose a new version of GPT exhibits unexpectedly strong performance on some task—maybe it's suddenly much better at manipulating humans or finding security vulnerabilities. Did it develop a novel, dangerous capability? Or is it just combining existing capabilities in a new way? MI gives us tools to answer this: we can trace which circuits activated, whether they're novel compositions of known algorithms, and what exactly changed between versions.

The **jailbreak problem** provides another concrete example. When users find prompts that bypass a model's safety training ("ignore previous instructions" or elaborate role-play scenarios), where is the safety mechanism failing? Is the refusal circuit not activating? Is it being overridden by other circuits? Without understanding the mechanism, we're reduced to whack-a-mole patching of individual jailbreaks. With MI, we might be able to verify that safety features persist robustly through all layers, or identify where they're vulnerable to corruption.

This is why major AI labs are investing heavily in MI research. Anthropic has published extensively on circuits in language models and recently released work on sparse autoencoders for disentangling features. OpenAI has dedicated interpretability teams working on everything from vision models to GPT-4. DeepMind is exploring mechanistic approaches to understanding their models' capabilities and limitations.

**The promise of MI is verification.** Imagine being able to prove—not just test, but mathematically prove—that a model won't exhibit certain dangerous behaviors because those behaviors would require circuits that don't exist in the architecture. Or being able to identify exactly which features contribute to alignment-relevant behaviors, and verify they remain stable across different prompts and contexts. This is the future MI is working toward: transforming AI safety from empirical evaluation to formal verification.

### The Transformer as Our Model Organism

Why focus on transformers specifically? Beyond being the architecture behind GPT-4, Claude, Gemini, and essentially every frontier language model, transformers have unique properties that make them ideal for mechanistic analysis.

First, they're **modular**. A transformer consists of repeated blocks, each containing attention and MLP layers. These blocks can be analyzed independently, then we can study how they compose. This is unlike, say, recurrent networks where state mixes across time steps in complex ways, or convolutional networks where locality and weight sharing create different analytical challenges.

Second, they have **clean mathematical structure**. Attention is ultimately just matrix multiplications and softmax. MLPs are linear transformations with point-wise non-linearities. The residual connections are simple addition. There are no complicated architectural tricks, no hand-crafted inductive biases. This mathematical cleanliness means we can use linear algebra tools directly—SVD, eigendecomposition, linear approximations—and they actually tell us something meaningful.

Third, there's a **growing body of MI research** on transformers that we can build on. Anthropic's "Mathematical Framework for Transformer Circuits" paper laid crucial groundwork. Researchers have identified specific circuits: induction heads that enable in-context learning, previous token heads, duplicate token heads, and increasingly complex compositions. This isn't unexplored territory—we have a foundation to stand on and extend.

Finally, transformers are **what matters practically**. If our goal is AI safety for real deployed systems, we need to understand the architecture those systems use. Insights from analyzing transformers directly apply to the models that matter most.

In this series, we'll focus on **decoder-only transformers**—the GPT architecture. These are simpler than encoder-decoder models (no cross-attention complications) and are the dominant architecture for language modeling. Everything we learn transfers to more complex variants with minor modifications.

Think of transformers as our "model organism" for AI, analogous to how biologists use E. coli or fruit flies. They're simple enough to study rigorously but complex enough to exhibit interesting behaviors. The circuits we discover in GPT-2 teach us principles that apply to GPT-4, even though we can't fully analyze the larger model. And the techniques we develop for understanding transformers will likely transfer to whatever architectures come next.


---

## 2. The Traditional View—And Why It's Wrong for MI

### How Transformers Are Usually Taught

If you've learned about transformers from most tutorials or courses, you've probably seen something like this:

```
Input Embedding
    ↓
Positional Encoding
    ↓
[Layer 1: Multi-Head Attention → Add & Norm → FFN → Add & Norm]
    ↓
[Layer 2: Multi-Head Attention → Add & Norm → FFN → Add & Norm]
    ↓
...
    ↓
[Layer 12: Multi-Head Attention → Add & Norm → FFN → Add & Norm]
    ↓
Output / Unembedding
```

The standard explanation of attention goes something like this:

1. Compute Queries (Q), Keys (K), and Values (V) by multiplying the input by three weight matrices
2. Compute attention scores: `scores = Q @ K^T / sqrt(d_k)`
3. Apply softmax to get attention weights
4. Multiply attention weights by Values: `output = softmax(scores) @ V`
5. Project back out with an output matrix

This view emphasizes the **computational graph**—how to implement attention efficiently, how gradients flow backward, how to parallelize across GPUs. Here's how you'd typically implement it:


```python
# Traditional view - implementation-focused
import torch
import torch.nn.functional as F

def traditional_attention(x, W_Q, W_K, W_V, W_O):
    """
    Traditional attention implementation
    x: [batch, seq_len, d_model]
    """
    # Project to Q, K, V
    Q = x @ W_Q  # [batch, seq_len, d_head]
    K = x @ W_K  # [batch, seq_len, d_head]
    V = x @ W_V  # [batch, seq_len, d_head]
    
    # Compute attention scores
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    attended_values = attention_weights @ V
    
    # Project back out
    output = attended_values @ W_O
    
    return output

print("Traditional view: clear implementation, sequential steps")
```

### Why This View Exists

To be fair to the traditional framing, it exists for good reasons:

- **Pedagogically clear**: One operation at a time, easy to follow
- **Maps to code**: Directly translates to implementation
- **Optimized for training**: Emphasizes gradient flow and backpropagation
- **Practical**: Helps you actually build transformers from scratch

If you're trying to implement a transformer or understand how to train one efficiently, this view is perfect.

### Why This View Obscures Understanding

But if you're trying to understand what a trained transformer **computes**—which algorithms it implements, how information flows, why it makes particular predictions—the traditional view actively gets in your way.

**Problem 1: The residual stream is invisible**

In the diagram above, you see "Add & Norm" between components, but it's treated as a minor implementation detail. The traditional view emphasizes the vertical stack of layers, making it seem like Layer 2 processes the output of Layer 1, which processed the output of Layer 0, etc. This is technically true but misses the crucial insight: **the residual stream is the primary pathway of information flow**.

**Problem 2: QK and OV circuits are hidden**

The traditional view fragments attention into Q, K, V steps, obscuring the fact that attention performs two independent operations:
- **WHERE to look** (determined by Q and K)
- **WHAT to copy** (determined by V and O)

These can be analyzed separately! But the sequential Q→K→V→O framing makes this non-obvious.

**Problem 3: Cross-layer communication is mysterious**

How does Layer 2 know what Layer 5 computed? How do layers coordinate to implement multi-step algorithms? The traditional view has no answer except "through the residuals," treated as a gradient flow mechanism rather than a communication channel.

**Problem 4: The compositional structure is lost**

Transformers implement complex behaviors through composition of simple operations. Induction heads, for instance, require two attention heads in different layers working together. The traditional layer-by-layer view makes these compositions nearly impossible to see.

### An Analogy

Learning transformers from the traditional view is like learning about the internet by studying TCP packet structure. Yes, you need to understand packets to implement networking protocols. But studying packet headers tells you **nothing** about:
- How Google Search works
- Why memes spread
- How social networks create echo chambers
- Why certain content goes viral

For understanding **computation and algorithms**, we need a different level of abstraction. We need to see the forest, not just the trees—or in this case, the information flow, not just the matrix multiplications.

That's where the residual stream view comes in.


---

## 3. The Residual Stream View

### Introducing the Residual Stream

Here's the key insight that changes everything:

> **Forget the vertical stack of layers. Instead, imagine a single vector—call it the residual stream—that flows through the model. Each layer reads from this stream, computes something, and writes back by *adding* to the stream. The stream accumulates information from every layer.**

Visually, instead of this (traditional):

```
Layer 0
   ↓
Layer 1
   ↓
Layer 2
   ↓
...
```

Think of this (residual stream):

```
x₀ → [+Layer0] → x₁ → [+Layer1] → x₂ → [+Layer2] → ... → x₁₂ → Output
      ↑                ↑                ↑
   reads x₀         reads x₁         reads x₂
   writes Δx₀       writes Δx₁       writes Δx₂
```

The residual stream flows **horizontally** through the model. At each step:
1. A layer reads the current stream state
2. Computes some transformation
3. **Adds** its output back to the stream

### The Mathematics

Let's make this concrete with equations:

```
x₀ = Embedding(token) + PositionalEncoding
x₁ = x₀ + Layer₁(x₀)
x₂ = x₁ + Layer₂(x₁)
x₃ = x₂ + Layer₃(x₂)
...
x₁₂ = x₁₁ + Layer₁₂(x₁₁)
logits = Unembed(x₁₂)
```

Notice that we can expand any xᵢ:

```
x₃ = x₀ + Layer₁(x₀) + Layer₂(x₁) + Layer₃(x₂)
```

**The residual stream at layer 3 is the sum of the original embedding plus the contributions from all three layers!**

This is fundamentally different from traditional neural networks where each layer overwrites the previous representation. Here, information **accumulates**.

### Why This View Matters

The residual stream view reveals something crucial:

> **The residual stream is a communication channel. Layers don't connect to each other directly—they communicate by reading from and writing to this shared stream.**

Think of it like a bulletin board:
- Layer 1 posts: "Found a noun at position 3"
- Layer 2 posts: "Found a verb at position 5"
- Layer 5 reads both messages and posts: "Subject-verb agreement detected"
- Layer 9 reads everything and makes a prediction

This makes several things immediately clear:

1. **Information persists**: What Layer 1 writes stays in the stream unless explicitly cancelled
2. **Layers can compose**: Layer 5 can build on what Layers 1-4 wrote
3. **Skip connections are natural**: Layer 8 can directly read what Layer 2 wrote
4. **Parallelism within layers**: Multiple attention heads in Layer 3 all read the same x₂ and all write to x₃


### Key Properties of the Residual Stream

#### Property 1: Linear and Additive

Every interaction with the residual stream is **linear**:
- Reading: `layer_input = x @ W_in` (matrix multiply—linear!)
- Writing: `x_new = x_old + output @ W_out` (addition—linear!)

The non-linearities (GELU, softmax) happen **inside** the layer computations, hidden from the residual stream itself.

**Why this matters:**
- Makes analysis tractable (we can use linear algebra tools)
- Enables superposition of features (multiple features can coexist in the same vector)
- Allows us to decompose: `x₁₂ = x₀ + contribution₁ + contribution₂ + ... + contribution₁₂`

#### Property 2: No Privileged Basis

Here's something wild: we could **rotate all the dimensions** of the residual stream, and if we rotate all the weight matrices accordingly, the model would compute exactly the same thing.

Mathematically: if R is any rotation matrix (orthogonal matrix), then:
```
x' = R @ x                    (rotate residual stream)
W' = R @ W @ R^T              (rotate weight matrices)

The model's outputs are identical!
```

**What this means:**
- Individual dimensions of the residual stream don't have inherent meaning
- You can't say "dimension 137 = dog detector" and have that be a fundamental truth
- Features are **superposed** across dimensions
- Interpretability requires finding the right basis (this is what Sparse Autoencoders try to do)


```python
# Demonstration: No privileged basis
import torch
import numpy as np

# Create a small residual stream vector
x = torch.randn(4)
print(f"Original residual stream:\n{x}\n")

# Create a random rotation matrix (orthogonal)
def random_rotation_matrix(n):
    """Generate a random orthogonal matrix using QR decomposition"""
    A = torch.randn(n, n)
    Q, R = torch.linalg.qr(A)
    return Q

R = random_rotation_matrix(4)
print(f"Rotation matrix R (orthogonal):")
print(f"R @ R.T = I? {torch.allclose(R @ R.T, torch.eye(4))}\n")

# Rotate the residual stream
x_rotated = R @ x
print(f"Rotated residual stream:\n{x_rotated}\n")

# Create a weight matrix
W = torch.randn(4, 4)

# Compute output in original basis
output_original = x @ W
print(f"Output (original basis): {output_original}\n")

# Rotate the weight matrix
W_rotated = R @ W @ R.T

# Compute output in rotated basis
output_rotated = x_rotated @ W_rotated
print(f"Output (rotated basis): {output_rotated}\n")

# Rotate back
output_rotated_back = R.T @ output_rotated
print(f"Output (rotated back): {output_rotated_back}\n")

print(f"Are they the same? {torch.allclose(output_original, output_rotated_back)}")
print("\nConclusion: The choice of basis doesn't affect computations!")
```

#### Property 3: Accumulates Information

The residual stream at layer N contains **everything** written by layers 0 through N-1:

```
x₆ = x₀ + Δx₁ + Δx₂ + Δx₃ + Δx₄ + Δx₅ + Δx₆
```

This is fundamentally different from traditional feedforward networks where Layer 6's representation is computed solely from Layer 5's output.

**Implications:**
- Early layers can influence late layers (even if separated by many layers)
- Information doesn't get "lost" unless explicitly cancelled by later layers
- Features can build on features from arbitrarily far back
- The final prediction depends on contributions from **all** layers

### Seeing Attention Through the Residual Stream Lens

Let's reframe attention in this view:

**Traditional view**: Attention is Q @ K^T → softmax → multiply by V → project with W_O

**Residual stream view**: Attention reads from the stream at multiple positions, computes weighted combinations, and writes back.

Here's a concrete walkthrough:


**Example: Processing "The cat sat on the mat"**

```
Position 2 ("sat"):

1. Read from residual stream at position 2: x[2]
2. Compute attention pattern: "I should attend 70% to 'cat', 20% to 'The', 10% to myself"
3. Read those positions from residual stream: x[1] (cat), x[0] (The), x[2] (sat)
4. Take weighted combination: 0.7 × x[1] + 0.2 × x[0] + 0.1 × x[2]
5. Transform this combination (extract features, e.g., "found subject 'cat'")
6. Write back to residual stream at position 2: x[2] += output

Later layers can now read this information!
```

The attention pattern (step 2) comes from the **QK circuit** (which we'll explore in the next post).  
The transformation (step 5) comes from the **OV circuit** (also next post).

For now, the key insight is: **attention is a mechanism for reading from multiple positions in the residual stream and writing a summary back**.


---

## 4. Virtual Weights: The Hidden Connections

### The Puzzle

We've established that layers only communicate through the residual stream—they never connect directly. But this raises a question:

> **How can we measure the effective connection between Layer 3 and Layer 7? How strong is their coupling? What information flows between them?**

Looking at attention patterns only tells you about information flow **within** a single layer. It doesn't tell you how Layer 3's output specifically influences Layer 7's computation.

We need a way to quantify: **"How much does neuron i in Layer 3 influence neuron j in Layer 7?"**

### The Virtual Weight Solution

Even though layers don't have direct connections in the architecture, they have **implicit** connections through the residual stream. We can compute these "virtual weights" by multiplying the output weights of one layer with the input weights of another.

Here's the insight:

**Layer i writes to residual stream:** When Layer i's neuron k activates, it contributes to the residual stream via `W_out[k, :]` (a d_model-dimensional vector).

**Layer j reads from residual stream:** Layer j's neuron m reads from the residual stream via `W_in[:, m]` (takes d_model input).

**The connection strength:** The effective weight from neuron k (Layer i) to neuron m (Layer j) is:
```
virtual_weight[m, k] = W_in_j[:, m]^T @ W_out_i[k, :]
                     = dot product of write vector and read vector
```

In matrix form:
```
W_virtual = W_in_j^T @ W_out_i
```

This captures the entire neuron-to-neuron coupling between the two layers!


### Concrete Example with Dimensions

Let's use GPT-2 small dimensions:
- d_model = 768 (residual stream)
- d_mlp = 3072 (MLP hidden layer)

**Layer 3 MLP:**
- `W_in_3`: [768, 3072] - maps residual stream → hidden layer
- `W_out_3`: [3072, 768] - maps hidden layer → residual stream

**Layer 7 MLP:**
- `W_in_7`: [768, 3072] - maps residual stream → hidden layer

**Virtual weight (neuron-to-neuron coupling):**
```python
W_virtual = W_in_7.T @ W_out_3
          = [3072, 768] @ [3072, 768].T
          = [3072, 768] @ [768, 3072]
          = [3072, 3072]
```

**W_virtual[i, j]** tells you: "How much does neuron j in Layer 3 influence neuron i in Layer 7?"


### Why Virtual Weights Matter

Virtual weights reveal the **implicit computational graph** of the transformer:

**1. Tracing information flow**  
Which early-layer features does this late layer depend on? If W_virtual has large values for specific neuron pairs, we know those neurons are communicating strongly.

**2. Finding circuits**  
Which layers work together to implement an algorithm? Strong virtual weights indicate functional coupling—these layers are part of the same computational circuit.

**3. Ablation studies**  
If I remove Layer 3, which later layers will be most affected? Virtual weights predict this: layers with strong W_virtual connections to Layer 3 will suffer most.

**4. Understanding composition**  
How do simple operations in early layers compose into complex behaviors in late layers? Virtual weights quantify this composition.

### Limitations to Acknowledge

> **Virtual weights are a linear approximation.**

The actual path from Layer 3 to Layer 7 involves non-linearities:
```
Layer 3: x → W_in → GELU(·) → W_out → residual stream
Layer 7: residual stream → W_in → ...
```

The GELU non-linearity means the actual influence is more complex than W_virtual suggests. But virtual weights give you a **first-order approximation**—they tell you the dominant coupling patterns even if they don't capture every nuance.

Think of it like Newtonian physics: it's an approximation that breaks down at extreme scales, but it's incredibly useful for everyday analysis.

### A Concrete Example: Pronoun Resolution

Suppose you're studying how GPT-2 handles: "The cat chased the mouse. It was tired."

You might find:
- **Layer 2**: Identifies "it" as a pronoun (specific neurons activate)
- **Layer 5**: Finds antecedent "cat" using attention (different neurons)
- **Layer 9**: Uses this information to predict "was" (verb agreement)

Virtual weights would show:
- Strong `W_virtual` from Layer 2 → Layer 5: Pronoun features feed into antecedent finding
- Strong `W_virtual` from Layer 5 → Layer 9: Antecedent information feeds into prediction
- Weak `W_virtual` from Layer 2 → Layer 9: No direct path (information goes through Layer 5)

This reveals the **circuit structure**: 2 → 5 → 9 forms a pipeline for pronoun resolution.


---

## 5. Hands-On Tutorial: Analyzing GPT-2

Now let's get our hands dirty! We'll extract residual streams and compute virtual weights from a real model.

### Setup


```python
# Install dependencies (run once)
# !pip install transformer-lens torch plotly numpy

import torch
from transformer_lens import HookedTransformer
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Load GPT-2 small
model = HookedTransformer.from_pretrained("gpt2-small")

print(f"Model: {model.cfg.model_name}")
print(f"Number of layers: {model.cfg.n_layers}")
print(f"Residual stream dimension (d_model): {model.cfg.d_model}")
print(f"MLP hidden dimension (d_mlp): {model.cfg.d_mlp}")
print(f"Number of attention heads per layer: {model.cfg.n_heads}")
print(f"Attention head dimension (d_head): {model.cfg.d_head}")
```

### Part 1: Extracting and Visualizing the Residual Stream


```python
# Choose a sentence to analyze
text = "The cat sat on the mat because it was"

# Tokenize
tokens = model.to_tokens(text)
print(f"Tokens shape: {tokens.shape}")  # [1, seq_len]
print(f"Tokens: {tokens}")
print(f"\nDecoded tokens:")
for i, token in enumerate(model.to_str_tokens(text)):
    print(f"  Position {i}: '{token}'")
```

```python
# Run model and cache all activations
logits, cache = model.run_with_cache(tokens)

print(f"\nCached {len(cache)} activation tensors")
print(f"Logits shape: {logits.shape}")  # [1, seq_len, vocab_size]
```

```python
# Extract residual stream at each layer
residual_streams = []

for layer in range(model.cfg.n_layers):
    # Get residual stream after this layer processes
    resid = cache[f"blocks.{layer}.hook_resid_post"]
    residual_streams.append(resid)
    print(f"Layer {layer:2d} residual shape: {resid.shape}")  # [1, seq_len, d_model]

# Stack for easy analysis
residuals = torch.stack(residual_streams)  # [n_layers, 1, seq_len, d_model]
print(f"\nStacked residuals shape: {residuals.shape}")
```

```python
# Visualize how residual stream evolves for a specific token
# Let's track the word "it" (position -2)
token_idx = -2
token_str = model.to_string(tokens[0, token_idx])

# Extract this token's residual stream across all layers
token_evolution = residuals[:, 0, token_idx, :]  # [n_layers, d_model]

# Compute L2 norm at each layer
norms = torch.norm(token_evolution, dim=1).cpu().numpy()

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(model.cfg.n_layers)),
    y=norms,
    mode='lines+markers',
    name='L2 Norm',
    line=dict(width=3),
    marker=dict(size=8)
))

fig.update_layout(
    title=f"Residual Stream Evolution for Token '{token_str}'",
    xaxis_title="Layer",
    yaxis_title="L2 Norm",
    font=dict(size=14),
    showlegend=False,
    width=800,
    height=500
)

fig.show()

print(f"\nNorm increases from {norms[0]:.2f} (layer 0) to {norms[-1]:.2f} (layer {model.cfg.n_layers-1})")
print(f"Information is accumulating in the residual stream!")
```

```python
# Visualize how much each layer contributes
# Compute difference between consecutive layers
layer_contributions = []
for i in range(1, model.cfg.n_layers):
    diff = residuals[i, 0, token_idx, :] - residuals[i-1, 0, token_idx, :]
    contribution = torch.norm(diff).item()
    layer_contributions.append(contribution)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=[f"L{i}→L{i+1}" for i in range(model.cfg.n_layers-1)],
    y=layer_contributions,
    marker_color='lightblue'
))

fig.update_layout(
    title=f"Layer-by-Layer Contributions to Token '{token_str}'",
    xaxis_title="Layer Transition",
    yaxis_title="Contribution Magnitude (L2 norm of change)",
    font=dict(size=12),
    width=1000,
    height=500
)

fig.show()

print(f"\nLayers with largest contributions:")
top_layers = np.argsort(layer_contributions)[-3:][::-1]
for idx in top_layers:
    print(f"  Layer {idx}→{idx+1}: {layer_contributions[idx]:.3f}")
```

**Observation:** Notice how the residual stream norm generally increases as we go through layers. Each layer is adding information! Some layers contribute more than others—these are likely doing more complex computations.


### Part 2: Computing Virtual Weights

Now let's compute the implicit connections between layers.


```python
def get_mlp_virtual_weight(model, layer_from, layer_to):
    """
    Compute virtual weight from layer_from MLP to layer_to MLP.
    
    Returns:
        W_virtual: [d_mlp, d_mlp] matrix where W_virtual[i, j] is the
                   influence of neuron j in layer_from on neuron i in layer_to
    """
    # Get output weights from source layer
    W_out_from = model.blocks[layer_from].mlp.W_out  # [d_mlp, d_model]
    
    # Get input weights from destination layer
    W_in_to = model.blocks[layer_to].mlp.W_in  # [d_model, d_mlp]
    
    # Compute virtual weight
    # W_virtual[i, j] = how much neuron j (layer_from) influences neuron i (layer_to)
    W_virtual = W_in_to.T @ W_out_from  # [d_mlp, d_mlp]
    
    return W_virtual

# Example: Compute virtual weight from Layer 2 to Layer 6
W_v = get_mlp_virtual_weight(model, layer_from=2, layer_to=6)

print(f"Virtual weight shape: {W_v.shape}")
print(f"\nStatistics:")
print(f"  Max absolute value: {W_v.abs().max().item():.4f}")
print(f"  Mean absolute value: {W_v.abs().mean().item():.4f}")
print(f"  Median absolute value: {W_v.abs().median().item():.4f}")
print(f"\nThis tells us how strongly Layer 2 neurons influence Layer 6 neurons")
```

```python
# Find the most influential neurons
# Sum over destination neurons to get total influence of each source neuron
neuron_influences = W_v.abs().sum(dim=0)  # [d_mlp]
top_source_neurons = torch.topk(neuron_influences, 10)

print("Top 10 most influential Layer 2 neurons (on Layer 6):")
for rank, (idx, strength) in enumerate(zip(top_source_neurons.indices, top_source_neurons.values)):
    print(f"  {rank+1}. Neuron {idx:4d}: total influence = {strength:.3f}")
```

```python
# Visualize the virtual weight matrix (sample, since 3072x3072 is too large)
# Let's look at top 100 neurons from each layer
sample_size = 100
top_source = torch.topk(neuron_influences, sample_size).indices
top_dest = torch.topk(W_v.abs().sum(dim=1), sample_size).indices

W_v_sample = W_v[top_dest, :][:, top_source].cpu().numpy()

fig = go.Figure(data=go.Heatmap(
    z=W_v_sample,
    x=[f"L2-N{i}" for i in top_source[:10]] + ["..."] + [f"L2-N{i}" for i in top_source[-10:]],
    y=[f"L6-N{i}" for i in top_dest[:10]] + ["..."] + [f"L6-N{i}" for i in top_dest[-10:]],
    colorscale='RdBu',
    zmid=0,
    colorbar=dict(title="Weight")
))

fig.update_layout(
    title="Virtual Weight Matrix (Layer 2 → Layer 6)<br>Top 100 neurons from each layer",
    xaxis_title="Layer 2 Neurons",
    yaxis_title="Layer 6 Neurons",
    width=800,
    height=800
)

fig.show()

print("\nRed = positive influence, Blue = negative influence")
print("Bright colors = strong connections between specific neuron pairs")
```

### Part 3: Computing Virtual Weights Across All Layer Pairs

Now let's get the big picture: how do ALL layers connect to each other?


```python
# Compute virtual weight strength for all layer pairs
n_layers = model.cfg.n_layers
virtual_weight_strengths = np.zeros((n_layers, n_layers))

print("Computing virtual weights for all layer pairs...")
for i in range(n_layers):
    for j in range(i+1, n_layers):  # Only j > i (can't influence the past)
        W_v = get_mlp_virtual_weight(model, layer_from=i, layer_to=j)
        
        # Use Frobenius norm as a measure of overall connection strength
        strength = torch.norm(W_v).item()
        virtual_weight_strengths[j, i] = strength
    
    if i % 3 == 0:
        print(f"  Processed layer {i}...")

print("Done!\n")
```

```python
# Create an interactive heatmap
fig = go.Figure(data=go.Heatmap(
    z=virtual_weight_strengths,
    x=[f"L{i}" for i in range(n_layers)],
    y=[f"L{i}" for i in range(n_layers)],
    colorscale='Viridis',
    colorbar=dict(title="Coupling<br>Strength")
))

fig.update_layout(
    title="Virtual Weight Strengths Across GPT-2 Layers<br>(MLP-to-MLP connections)",
    xaxis_title="Source Layer (writes to residual stream)",
    yaxis_title="Destination Layer (reads from residual stream)",
    font=dict(size=12),
    width=800,
    height=800,
    xaxis=dict(side='top')
)

fig.show()

print("\nKey observations:")
print("- Diagonal band: adjacent layers are strongly coupled")
print("- Off-diagonal structure: some layers have strong long-range connections")
print("- Upper triangle only: layers can only influence future layers, not past ones")
```

```python
# Find the strongest cross-layer connections
print("\nTop 10 strongest layer-to-layer connections:")

# Flatten and sort
connections = []
for i in range(n_layers):
    for j in range(i+1, n_layers):
        connections.append((i, j, virtual_weight_strengths[j, i]))

connections.sort(key=lambda x: x[2], reverse=True)

for rank, (source, dest, strength) in enumerate(connections[:10], 1):
    print(f"  {rank:2d}. Layer {source:2d} → Layer {dest:2d}: {strength:.2f}")
```

**Interpretation:**

This heatmap shows the implicit computational graph of GPT-2:
- **Bright spots** = strong coupling between those layers
- **Dark spots** = weak coupling (less information flow)
- **Diagonal band** = adjacent layers always talk to each other
- **Off-diagonal bright spots** = interesting! These are skip connections where early layers directly influence late layers

For instance, if Layer 2 → Layer 8 is bright, it means Layer 2 computes features that Layer 8 specifically relies on. This could be part of a computational circuit!


### Part 4: Exploring Further (Optional)

Here are some experiments you can try:


```python
# Experiment 1: Compare different sentences
# Do virtual weight patterns change based on input?
# (Spoiler: No! Virtual weights are fixed by the model parameters)

texts = [
    "The cat sat on the mat",
    "To be or not to be",
    "Machine learning is awesome"
]

for text in texts:
    tokens = model.to_tokens(text)
    logits, cache = model.run_with_cache(tokens)
    
    # Virtual weights don't depend on input!
    # They're properties of the model, not the data
    print(f"Text: '{text}'")
    print(f"  Virtual weights: exactly the same for any input!\n")
```

```python
# Experiment 2: Virtual weights for attention heads
# We focused on MLPs, but we can also compute virtual weights between attention heads

def get_attention_virtual_weight(model, layer_from, head_from, layer_to, head_to):
    """
    Compute virtual weight from one attention head to another.
    This uses the OV circuit (W_V @ W_O) from the source head
    and QK circuit inputs from the destination head.
    """
    # Source head's OV circuit (what it writes)
    W_V_from = model.blocks[layer_from].attn.W_V[head_from]  # [d_model, d_head]
    W_O_from = model.blocks[layer_from].attn.W_O[head_from]  # [d_head, d_model]
    W_OV_from = W_V_from @ W_O_from  # [d_model, d_model]
    
    # Destination head's Q input (what it reads for queries)
    W_Q_to = model.blocks[layer_to].attn.W_Q[head_to]  # [d_model, d_head]
    
    # Virtual weight: how source head's output affects dest head's queries
    W_virtual = W_Q_to.T @ W_OV_from  # [d_head, d_model]
    
    return W_virtual

# Example: Head 0 in Layer 1 → Head 3 in Layer 5
W_v_attn = get_attention_virtual_weight(model, 
                                        layer_from=1, head_from=0,
                                        layer_to=5, head_to=3)

print(f"Attention virtual weight shape: {W_v_attn.shape}")
print(f"Strength: {torch.norm(W_v_attn).item():.3f}")
print("\nThis tells us how Head 1.0's output influences Head 5.3's attention pattern")
```

---

## 6. Why This Matters for AI Safety

Let's connect everything back to the motivation: building safe AI systems.

### Tracking "Intent" Features

If a model is being deceptive, it might maintain separate representations of:
- **Stated intent**: What it claims to be doing
- **Actual intent**: What it's really optimizing for

The residual stream view gives us tools to track this:

1. **Identify candidate features**: Which dimensions of the residual stream encode "intent"?
2. **Trace their evolution**: How do these features change across layers?
3. **Find divergence points**: Where do "stated" and "actual" intent features diverge?
4. **Use virtual weights**: Which layers compute these features? Which layers read them?

This transforms deception detection from behavioral testing ("does it sometimes lie?") to mechanistic verification ("are there divergent intent features?").

### Understanding Refusal Mechanisms

When Claude refuses a harmful request, where does that decision happen?

With residual stream analysis:
1. Compare residual streams for refused vs. complied requests
2. Find which layers show the biggest differences ("refusal features activate here")
3. Use virtual weights to trace: which earlier layers feed into the refusal circuit?
4. Verify robustness: do refusal features persist through all subsequent layers?

This is active research at Anthropic. The goal: verify that safety mechanisms are deeply integrated, not superficial.

### Detecting Capability Emergence

Suppose GPT-5 suddenly exhibits a concerning capability—say, generating highly effective phishing emails.

Is this a novel circuit, or composition of existing ones?

Analysis approach:
1. Compare GPT-5 vs GPT-4 residual streams on the same task
2. Identify which layers show new activation patterns
3. Compute virtual weights to see if these are new circuits or novel compositions
4. Trace information flow: is this capability localized or distributed?

This helps capability oversight: we can identify **where** new behaviors come from and **how** to monitor or control them.

### The Broader Vision

Traditional ML treats models as input-output mappings optimized for accuracy. Mechanistic interpretability treats them as **algorithms we can understand, verify, and control**.

The shift is profound:
- From "it usually works" → "we can prove it works"
- From "test on more examples" → "analyze the mechanism"
- From "hope it's safe" → "verify it's safe"

The residual stream view and virtual weights are foundational tools for this transformation. They let us:
- See inside the black box
- Understand information flow
- Identify computational circuits
- Verify safety properties mechanistically

This is just the beginning. In the next post, we'll go deeper: dissecting attention into QK and OV circuits, understanding how heads compose to implement algorithms, and discovering interpretable patterns like induction heads.


---

## 7. Conclusion and Next Steps

### What We've Learned

In this post, we've fundamentally reframed how to think about transformers:

**Mechanistic interpretability** is about reverse-engineering algorithms, not just probing representations or computing attributions. It's the difference between observing behavior and understanding mechanism.

**The residual stream** is not just a gradient flow trick—it's a communication channel. Layers read from it, compute transformations, and write back by adding. Information accumulates rather than being overwritten.

**Key properties** of the residual stream:
- Linear and additive (makes analysis tractable)
- No privileged basis (features are superposed)
- Accumulates information (early layers influence late layers)

**Virtual weights** capture implicit connections between layers. Even though Layer 3 and Layer 7 don't connect directly, we can quantify their coupling through the residual stream. This reveals the computational graph hidden in the architecture.

**Why it matters for safety**: Understanding mechanisms enables verification. We can track how "intent" features flow, verify that safety circuits are robust, and detect when novel capabilities emerge.

### What's Next

We've seen how layers communicate, but we haven't looked inside attention heads themselves. That's the next frontier.

**Coming in Part 2: Attention Circuits (QK and OV)**

Attention can be decomposed into two independent circuits:

**The QK circuit** (W_QK = W_Q @ W_K^T) determines **where to look**:
- Previous token attention
- Induction patterns
- Syntactic relationships

**The OV circuit** (W_OV = W_V @ W_O) determines **what to copy**:
- Token identity
- Semantic features  
- Transformed representations

These circuits compose in beautiful ways:
- Induction heads use two layers (previous token head + induction head)
- Virtual weights between circuits (W_OV₁ @ W_QK₂) show composition
- We can find and verify specific algorithms in trained models

**You'll learn:**
- How to extract and analyze QK and OV circuits
- What patterns different heads implement
- How to find multi-head algorithms like induction
- Practical techniques for circuit analysis

### Try It Yourself

All the code from this post is available:
- **GitHub repository**: [link to your repo]
- **Colab notebook**: [link to Colab version]
- **TransformerLens docs**: https://neelnanda-io.github.io/TransformerLens/

**Suggested experiments:**
1. Analyze different models (GPT-2 medium, Pythia, etc.)
2. Compare virtual weight patterns across architectures
3. Find which layer pairs have the strongest coupling
4. Track specific tokens through the residual stream

### Let's Connect

I'm actively transitioning from ML engineering to AI safety research. If you're working on mechanistic interpretability, interested in collaborating, or just want to discuss these ideas:

- **Twitter/X**: [@your_handle]
- **GitHub**: [your GitHub]
- **Email**: [your email]

I'm especially interested in:
- Circuit analysis in production models
- Applications to AI safety and alignment
- Research collaborations
- Opportunities in safety-focused organizations

### Further Reading

**Foundational papers:**
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) - Anthropic
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) - Anthropic
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) - Anthropic

**Learning resources:**
- Neel Nanda's [MI tutorial series](https://www.neelnanda.io/mechanistic-interpretability/quickstart)
- TransformerLens [documentation](https://neelnanda-io.github.io/TransformerLens/)
- [MI reading list](https://www.neelnanda.io/mechanistic-interpretability/prereqs) - comprehensive

**Recent work:**
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) - Anthropic (Oct 2024)
- [Sparse Autoencoders](https://transformer-circuits.pub/2023/monosemantic-features/index.html) - Anthropic

---

Thanks for reading! See you in Part 2 where we'll dissect attention heads and discover the circuits they implement.

**Questions? Feedback? Found this helpful?** Let me know in the comments or reach out directly. I'd love to hear what resonated and what you'd like to see in future posts.


---

## Appendix: Technical Notes

### On the Quality of Linear Approximations

Virtual weights are a linear approximation. How good is this approximation?

**When it works well:**
- Adjacent layers (only one non-linearity between them)
- Small activations (where GELU ≈ linear)
- Understanding directional influence (which neurons affect which)

**When it breaks down:**
- Many layers apart (compounds non-linear errors)
- Large activations (GELU is very non-linear for large |x|)
- Predicting exact activation magnitudes

**Rule of thumb**: Use virtual weights for qualitative understanding ("these layers are coupled") rather than quantitative prediction ("this exact activation value").

### On Superposition and the Privileged Basis Problem

Why does "no privileged basis" matter?

In principle, a network could learn to use dimensions cleanly:
- Dimension 0: "is dog"
- Dimension 1: "is cat"  
- Dimension 2: "is animal"
- etc.

But networks learn to **superpose** features: many features are represented as combinations of dimensions. It's like storing 1000 files on a hard drive with only 100 blocks by using compression.

This is why sparse autoencoders (SAEs) are valuable: they try to find a basis where features are sparse (mostly zeros), making them interpretable.

### Connection to Other MI Concepts

**Logit lens**: Projects intermediate residual streams to vocabulary space to see what the model is "thinking" at each layer. Uses the residual stream directly!

**Activation patching**: Replaces activations (often in residual stream) to measure causal effects. The residual stream is the natural intervention point.

**Path patching**: Traces specific paths through the network by patching along virtual weight connections.

All of these techniques build on the residual stream framework we've developed.

### Code Repository

Full code with additional examples and utilities:
[Link to GitHub repo]

Includes:
- Extended visualization functions
- Virtual weight computation for attention heads
- Utilities for analyzing multiple models
- Example analyses of specific heads

