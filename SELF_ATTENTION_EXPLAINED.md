# How Self-Attention Learns Relationships Between Words: A Detailed Explanation

## Table of Contents
1. [Intuition: What is Self-Attention?](#intuition)
2. [The Core Mechanism: Query, Key, Value](#qkv)
3. [Step-by-Step Mathematical Process](#mathematics)
4. [How It Learns During Training](#learning)
5. [Concrete Example Walkthrough](#example)
6. [Multi-Head Attention: Capturing Different Relationships](#multihead)
7. [Comparison with Other Architectures](#comparison)
8. [Visual Understanding](#visual)

---

## 1. Intuition: What is Self-Attention? {#intuition}

**Self-attention** allows each word in a sequence to "look at" and "attend to" all other words in the sequence to understand their relationships.

### The Key Idea

Imagine reading a sentence: **"The cat sat on the mat"**

When processing the word **"it"** in **"The cat sat on the mat. It was happy"**, you need to know:
- What does "it" refer to? → "cat"
- This is a **relationship** between words

Self-attention learns to identify these relationships automatically by:
1. Computing how **relevant** each word is to every other word
2. Creating **weighted combinations** of word representations
3. Learning these relevance patterns from data

### Why "Self"?

It's called "self-attention" because:
- Each word attends to **all words in the same sequence** (including itself)
- Unlike cross-attention (used in encoder-decoder), which attends across different sequences

---

## 2. The Core Mechanism: Query, Key, Value (Q, K, V) {#qkv}

The self-attention mechanism uses three learned representations for each word:

### Query (Q): "What am I looking for?"
- Represents: **What information does this word need?**
- Example: The word "it" needs to find its referent

### Key (K): "What information do I have?"
- Represents: **What information does this word provide?**
- Example: The word "cat" provides information about a noun entity

### Value (V): "What information do I contain?"
- Represents: **The actual content/meaning of the word**
- Example: The semantic representation of "cat"

### The Analogy

Think of it like a **library system**:
- **Query (Q)**: Your search query ("I need information about animals")
- **Key (K)**: Book titles/index (what each book is about)
- **Value (V)**: The actual book content

You compare your query to all keys, find the most relevant ones, then read the corresponding values.

---

## 3. Step-by-Step Mathematical Process {#mathematics}

Let's trace through the exact computation using your code:

### Step 1: Input Embeddings

**Input:** Sequence of token IDs
```
"The cat sat on the mat" → [token_1, token_2, token_3, token_4, token_5, token_6]
```

**Process:**
```python
# From model.py line 260
x = self.token_embedding(x) * math.sqrt(self.d_model)
# Result: [seq_len, d_model] - each word is now a d_model-dimensional vector
```

Each word becomes a dense vector (e.g., 512 dimensions) that encodes its meaning.

### Step 2: Add Positional Encoding

**Why?** Word embeddings don't contain position information. "cat" before "sat" vs. after "sat" matters!

```python
# From model.py line 263
x = self.pos_encoding(x)
```

**Result:** Each word now has:
- **Semantic information** (from embedding)
- **Position information** (from positional encoding)

### Step 3: Create Query, Key, Value Matrices

**From model.py lines 23-25:**
```python
self.w_q = nn.Linear(d_model, d_model)  # Query projection
self.w_k = nn.Linear(d_model, d_model)  # Key projection
self.w_v = nn.Linear(d_model, d_model)  # Value projection
```

**Computation (lines 48-50):**
```python
Q = self.w_q(x)  # [batch, seq_len, d_model]
K = self.w_k(x)  # [batch, seq_len, d_model]
V = self.w_v(x)  # [batch, seq_len, d_model]
```

**What happens:**
- Each word's embedding is transformed into three different representations
- These transformations are **learned** during training
- Q, K, V capture different aspects of each word

### Step 4: Compute Attention Scores

**From model.py line 32:**
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
```

**Mathematical breakdown:**

For each word position `i`, we compute how relevant every other word `j` is:

```
score[i, j] = (Q[i] · K[j]) / √d_k
```

**What this means:**
- **Dot product** `Q[i] · K[j]`: Measures similarity between:
  - What word `i` is looking for (Query)
  - What word `j` provides (Key)
- **Scaling** `/√d_k`: Prevents scores from becoming too large (helps with gradient flow)

**Result:** A matrix of scores `[seq_len, seq_len]`
- Row `i`: How relevant each word is to word `i`
- Column `j`: How relevant word `j` is to each word

### Step 5: Apply Masking (for Causal/Decoder Models)

**From model.py lines 34-36:**
```python
if mask is not None:
    scores = scores.masked_fill(mask, float('-inf'))
```

**Why?** In autoregressive models (like GPT), we prevent words from seeing future words:
- Word at position 3 can only attend to positions 0, 1, 2
- This maintains the autoregressive property

### Step 6: Apply Softmax to Get Attention Weights

**From model.py line 38:**
```python
attention_weights = F.softmax(scores, dim=-1)
```

**What happens:**
- Converts scores into **probabilities** (sum to 1.0)
- Higher scores → higher probabilities
- Each row sums to 1.0 (each word distributes attention across all words)

**Example:**
```
Before softmax: [2.0, 1.0, 0.5, -1.0]
After softmax:  [0.65, 0.24, 0.09, 0.02]  (sums to 1.0)
```

### Step 7: Weighted Sum of Values

**From model.py line 41:**
```python
output = torch.matmul(attention_weights, V)
```

**Mathematical operation:**
```
output[i] = Σ(attention_weights[i, j] × V[j]) for all j
```

**What this means:**
- For each word `i`, we create a **weighted combination** of all word values
- Words with higher attention weights contribute more
- The result is a new representation that incorporates information from relevant words

---

## 4. How It Learns During Training {#learning}

### The Learning Process

**During training, the model learns:**
1. **What to look for** (Query weights `W_q`)
2. **What information to provide** (Key weights `W_k`)
3. **What content to share** (Value weights `W_v`)

### Backpropagation Flow

**Forward pass:**
```
Input words → Embeddings → Q, K, V → Attention scores → Attention weights → Output
```

**Backward pass (gradient flow):**
```
Loss ← Output ← Attention weights ← Scores ← Q, K, V ← Embeddings ← Input
```

**What gets updated:**
- `W_q`, `W_k`, `W_v` matrices learn to:
  - Extract relevant information for queries
  - Provide useful keys for matching
  - Encode meaningful values

### Example: Learning Pronoun Resolution

**Training example:**
```
"The cat sat on the mat. It was happy."
Target: Predict "happy" correctly
```

**Learning process:**
1. **Initial state:** Random Q, K, V weights
   - "It" might attend randomly to all words

2. **After training:**
   - `W_q` learns: "it" should look for noun entities
   - `W_k` learns: "cat" should provide entity information
   - `W_v` learns: "cat" contains entity semantics
   
3. **Result:**
   - Attention weight: `attention["it", "cat"]` becomes high (e.g., 0.8)
   - Attention weight: `attention["it", "mat"]` becomes low (e.g., 0.05)
   - The model learns the relationship!

### What Relationships Are Learned?

Self-attention learns various types of relationships:

1. **Syntactic relationships:**
   - Subject-verb: "cat" → "sat"
   - Adjective-noun: "happy" → "cat"
   - Preposition-object: "on" → "mat"

2. **Semantic relationships:**
   - Coreference: "it" → "cat"
   - Antonyms: "happy" vs "sad"
   - Synonyms: "cat" vs "feline"

3. **Long-range dependencies:**
   - "The cat that I saw yesterday was happy"
   - "cat" relates to "happy" despite distance

4. **Contextual relationships:**
   - "bank" (financial) vs "bank" (river)
   - Context determines meaning

---

## 5. Concrete Example Walkthrough {#example}

Let's trace through a concrete example:

### Input Sentence
```
"The cat sat on the mat"
```

### Step-by-Step Computation

#### Step 1: Tokenization and Embedding
```
"The"   → embedding_1: [0.1, 0.3, -0.2, ..., 0.5]  (512 dims)
"cat"   → embedding_2: [0.4, -0.1, 0.6, ..., -0.3]
"sat"   → embedding_3: [-0.2, 0.5, 0.1, ..., 0.4]
"on"    → embedding_4: [0.3, 0.2, -0.4, ..., 0.1]
"the"   → embedding_5: [0.1, 0.3, -0.2, ..., 0.5]
"mat"   → embedding_6: [0.2, -0.3, 0.4, ..., -0.2]
```

#### Step 2: Add Positional Encoding
Each embedding now includes position information:
```
"The"   → [0.1+pos_0, 0.3+pos_0, ..., 0.5+pos_0]
"cat"   → [0.4+pos_1, -0.1+pos_1, ..., -0.3+pos_1]
...
```

#### Step 3: Compute Q, K, V
```python
Q = W_q × embeddings  # What each word is looking for
K = W_k × embeddings  # What each word provides
V = W_v × embeddings  # What each word contains
```

**Example for "cat" (position 1):**
```
Q["cat"] = W_q × embedding_2  # "I need to find my verb and modifiers"
K["cat"] = W_k × embedding_2  # "I am a noun entity, subject"
V["cat"] = W_v × embedding_2  # [semantic content of "cat"]
```

#### Step 4: Compute Attention Scores

**For word "sat" (position 2) attending to all words:**

```
score["sat", "The"]   = Q["sat"] · K["The"]   / √d_k = 0.3
score["sat", "cat"]   = Q["sat"] · K["cat"]   / √d_k = 8.5  ← HIGH!
score["sat", "sat"]   = Q["sat"] · K["sat"]   / √d_k = 2.1
score["sat", "on"]    = Q["sat"] · K["on"]    / √d_k = 1.2
score["sat", "the"]   = Q["sat"] · K["the"]   / √d_k = 0.2
score["sat", "mat"]   = Q["sat"] · K["mat"]   / √d_k = 0.1
```

**Why is "cat" high?**
- Q["sat"] learned: "I need my subject"
- K["cat"] learned: "I am a noun subject"
- High similarity → high score!

#### Step 5: Apply Softmax

```
attention_weights["sat", "The"]   = 0.01
attention_weights["sat", "cat"]   = 0.85  ← Most attention!
attention_weights["sat", "sat"]   = 0.08
attention_weights["sat", "on"]   = 0.04
attention_weights["sat", "the"]   = 0.01
attention_weights["sat", "mat"]   = 0.01
```

**Interpretation:** "sat" pays 85% attention to "cat" (its subject)!

#### Step 6: Weighted Sum

```
output["sat"] = 0.01×V["The"] + 0.85×V["cat"] + 0.08×V["sat"] + 
                0.04×V["on"] + 0.01×V["the"] + 0.01×V["mat"]
```

**Result:** The representation of "sat" now incorporates:
- **85%** information from "cat" (subject relationship)
- **8%** from itself
- **6%** from other words

This creates a **context-aware representation**!

---

## 6. Multi-Head Attention: Capturing Different Relationships {#multihead}

### Why Multiple Heads?

**Single head** might learn one type of relationship. **Multiple heads** learn different types simultaneously!

### How It Works

**From model.py lines 48-50:**
```python
# Split d_model into num_heads parts
Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
V = V.view(batch_size, seq_len, self.num_heads, self.d_k)
```

**Example with 8 heads (d_model=512, d_k=64):**
- Head 1: Might learn **syntactic relationships** (subject-verb)
- Head 2: Might learn **semantic relationships** (synonyms)
- Head 3: Might learn **positional relationships** (nearby words)
- Head 4: Might learn **long-range dependencies**
- Head 5: Might learn **coreference** (pronouns)
- Head 6: Might learn **modifier relationships** (adjective-noun)
- Head 7: Might learn **functional words** (articles, prepositions)
- Head 8: Might learn **domain-specific** patterns

### Example: Different Heads for "it"

**Sentence:** "The cat sat on the mat. It was happy."

**Head 1 (Coreference):**
```
attention["it", "cat"] = 0.90  ← Finds referent
attention["it", "mat"] = 0.05
```

**Head 2 (Syntactic):**
```
attention["it", "was"] = 0.60  ← Finds verb
attention["it", "happy"] = 0.25
```

**Head 3 (Semantic):**
```
attention["it", "cat"] = 0.70  ← Entity information
attention["it", "mat"] = 0.20  ← Location context
```

**Result:** Each head captures different aspects, then they're combined!

---

## 7. Comparison with Other Architectures {#comparison}

### vs. Recurrent Neural Networks (RNNs)

**RNN approach:**
```
word_1 → hidden_state_1 → word_2 → hidden_state_2 → word_3 → ...
```

**Problems:**
- Sequential processing (slow)
- Information bottleneck (early words fade)
- Hard to capture long-range dependencies
- Vanishing gradients

**Self-attention:**
- ✅ Parallel processing (all words simultaneously)
- ✅ Direct connections (any word to any word)
- ✅ No information bottleneck
- ✅ Better gradient flow

### vs. Convolutional Neural Networks (CNNs)

**CNN approach:**
```
Local windows: [word_1, word_2, word_3] → feature
```

**Problems:**
- Limited receptive field (needs many layers)
- Fixed patterns (convolution kernels)
- Doesn't capture arbitrary relationships

**Self-attention:**
- ✅ Global receptive field (one layer)
- ✅ Learned patterns (adaptive)
- ✅ Captures any relationship

---

## 8. Visual Understanding {#visual}

### Attention Visualization

For the sentence **"The cat sat on the mat"**, a trained model might show:

```
        The   cat   sat   on   the   mat
The     0.1   0.0   0.0   0.0   0.0   0.0
cat     0.2   0.3   0.4   0.05  0.0   0.05  ← "cat" attends to "sat" (verb)
sat     0.05  0.8   0.1   0.03  0.0   0.02  ← "sat" attends to "cat" (subject)
on      0.1   0.1   0.2   0.1   0.2   0.3   ← "on" attends to "mat" (object)
the     0.1   0.0   0.0   0.0   0.1   0.0
mat     0.05  0.0   0.0   0.3   0.1   0.55  ← "mat" attends to "on" (preposition)
```

**Key observations:**
- **"cat" → "sat"**: Subject-verb relationship (0.4)
- **"sat" → "cat"**: Verb-subject relationship (0.8)
- **"on" → "mat"**: Preposition-object relationship (0.3)
- **"mat" → "on"**: Object-preposition relationship (0.3)

### Information Flow

```
Input: "The cat sat on the mat"

Layer 1 (First Transformer Block):
  - Learns basic relationships (nearby words)
  - "cat" and "sat" start to connect
  
Layer 2:
  - Refines relationships
  - "cat" → "sat" relationship strengthens
  
Layer 3:
  - Captures more complex patterns
  - Long-range dependencies emerge
  
Layer 4+:
  - High-level semantic relationships
  - Context-aware representations
```

---

## 9. Key Insights

### What Makes Self-Attention Powerful?

1. **Direct Relationships:**
   - Any word can directly attend to any other word
   - No need to pass through intermediate states

2. **Learned Relevance:**
   - The model learns which relationships matter
   - Different from fixed rules (like grammar parsers)

3. **Context-Aware:**
   - Same word in different contexts gets different attention patterns
   - "bank" (financial) vs "bank" (river) have different attention

4. **Compositional:**
   - Multiple layers build increasingly complex relationships
   - Early layers: simple patterns
   - Later layers: complex semantic relationships

### What Gets Learned?

**The weight matrices W_q, W_k, W_v learn:**
- How to extract relevant information (queries)
- How to provide useful information (keys)
- How to encode meaningful content (values)

**Through training:**
- The model sees millions of examples
- Backpropagation updates weights to minimize loss
- Attention patterns emerge that help predict next words
- These patterns encode linguistic relationships!

---

## 10. Summary

**Self-attention learns word relationships by:**

1. **Creating three representations** (Q, K, V) for each word
2. **Computing similarity scores** between queries and keys
3. **Weighting values** based on these similarities
4. **Learning the transformations** (W_q, W_k, W_v) that extract useful relationships
5. **Stacking multiple layers** to build complex relationships
6. **Using multiple heads** to capture different types of relationships

**The magic:** Through training on large text corpora, the model automatically discovers linguistic patterns (syntax, semantics, coreference, etc.) without explicit rules!

**Result:** Each word's representation becomes **context-aware**, incorporating information from all relevant words in the sequence, enabling the model to understand complex language relationships.



