# Extracting Word Embeddings from the LLM

## Overview

There are **three main options** for word embeddings in this LLM, each with different use cases:

1. **Input Token Embeddings** (`self.token_embedding`) - Static, context-independent
2. **Context-Aware Embeddings** (after transformer blocks) - Context-dependent
3. **Vocabulary-Level Embeddings** (averaged contextual) - One embedding per word type

---

## Option 1: Input Token Embeddings (Static)

### What It Is

**From model.py line 183:**
```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

This is the **initial embedding layer** - one fixed vector per vocabulary token.

### Characteristics

- ✅ **Static**: Same embedding for a word regardless of context
- ✅ **Fast**: Direct lookup, no computation needed
- ✅ **Vocabulary-level**: One embedding per word type
- ❌ **No context**: "bank" (financial) and "bank" (river) have same embedding
- ❌ **Less semantic**: Not as rich as contextual embeddings

### How to Extract

```python
import torch

# Load model
checkpoint = torch.load('checkpoints_tinystories/checkpoint_epoch_1.pt')
model = checkpoint['model']  # or load your model
model.eval()

# Extract embeddings
word_embeddings = model.token_embedding.weight.data
# Shape: [vocab_size, d_model]
# Example: [50000, 512]

# Get embedding for specific token
tokenizer = ...  # Your tokenizer
token_id = tokenizer.encode("cat")[0]
cat_embedding = word_embeddings[token_id]
# Shape: [d_model] = [512]
```

### Use Cases

- Fast similarity search
- Initialization for other models
- Vocabulary-level analysis
- When context doesn't matter

---

## Option 2: Context-Aware Embeddings (Recommended)

### What It Is

The **output after transformer blocks** (before the final vocabulary projection). These embeddings incorporate context from the entire sequence.

### Characteristics

- ✅ **Context-aware**: Same word gets different embeddings in different contexts
- ✅ **Rich semantics**: Incorporates information from all words in sequence
- ✅ **Better quality**: More useful for downstream tasks
- ❌ **Context-dependent**: Same word has different embeddings in different sentences
- ❌ **Requires forward pass**: Need to process sequences

### How to Extract

**Method 1: Modify forward() to return embeddings**

Add this method to `TransformerLLM` class:

```python
def get_embeddings(self, x, mask=None, return_logits=False):
    """
    Extract context-aware word embeddings.
    
    Args:
        x: Input token indices [batch_size, seq_len]
        mask: Optional attention mask
        return_logits: If True, also return logits
    
    Returns:
        embeddings: [batch_size, seq_len, d_model]
        logits: [batch_size, seq_len, vocab_size] (if return_logits=True)
    """
    batch_size, seq_len = x.size()
    
    if mask is None:
        mask = self.generate_mask(seq_len).to(x.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Token embeddings
    x = self.token_embedding(x) * math.sqrt(self.d_model)
    
    # Positional encoding
    x = self.pos_encoding(x)
    x = self.dropout(x)
    
    # Pass through transformer blocks
    for transformer_block in self.transformer_blocks:
        x = transformer_block(x, mask)
    
    # Final layer norm (this is the embedding!)
    embeddings = self.layer_norm(x)
    
    if return_logits:
        logits = self.fc_out(embeddings)
        return embeddings, logits
    
    return embeddings
```

**Method 2: Extract from existing forward()**

Modify the forward method to optionally return embeddings:

```python
def forward(self, x, mask=None, return_embeddings=False):
    # ... existing code ...
    
    # Final layer norm and output projection
    x = self.layer_norm(x)
    
    if return_embeddings:
        return x  # Return embeddings before vocab projection
    
    logits = self.fc_out(x)
    return logits
```

**Usage:**

```python
# Load model
model = TransformerLLM(...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Tokenize input
tokenizer = ...  # Your tokenizer
text = "The cat sat on the mat"
tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([tokens]).to(device)

# Extract embeddings
with torch.no_grad():
    embeddings = model.get_embeddings(tokens_tensor)
    # Shape: [batch_size, seq_len, d_model] = [1, 6, 512]

# Get embedding for specific word
word_position = 1  # "cat" is at position 1
cat_embedding = embeddings[0, word_position]
# Shape: [d_model] = [512]
```

### Use Cases

- Semantic similarity (context-aware)
- Downstream NLP tasks
- Sentence/document embeddings
- When context matters

---

## Option 3: Vocabulary-Level Contextual Embeddings

### What It Is

**Average contextual embeddings** across many contexts to get one embedding per word type.

### Characteristics

- ✅ **Best of both worlds**: Context-aware but vocabulary-level
- ✅ **Rich semantics**: Learned from many contexts
- ✅ **One per word**: Consistent representation per word type
- ❌ **Requires corpus**: Need many examples of each word
- ❌ **Computation**: Need to process many sentences

### How to Extract

```python
def extract_vocabulary_embeddings(model, tokenizer, corpus, device='cpu'):
    """
    Extract vocabulary-level embeddings by averaging contextual embeddings
    across many examples.
    
    Args:
        model: Trained TransformerLLM
        tokenizer: Tokenizer instance
        corpus: List of sentences containing vocabulary words
        device: Device to run on
    
    Returns:
        vocab_embeddings: Dict mapping token_id to averaged embedding
    """
    model.eval()
    
    # Track embeddings for each token
    token_embeddings = {}  # token_id -> list of embeddings
    token_counts = {}       # token_id -> count
    
    with torch.no_grad():
        for sentence in corpus:
            tokens = tokenizer.encode(sentence)
            tokens_tensor = torch.tensor([tokens]).to(device)
            
            # Get contextual embeddings
            embeddings = model.get_embeddings(tokens_tensor)
            # Shape: [1, seq_len, d_model]
            
            # Store embeddings for each token
            for pos, token_id in enumerate(tokens):
                if token_id not in token_embeddings:
                    token_embeddings[token_id] = []
                    token_counts[token_id] = 0
                
                token_embeddings[token_id].append(embeddings[0, pos].cpu())
                token_counts[token_id] += 1
    
    # Average embeddings for each token
    vocab_embeddings = {}
    for token_id, emb_list in token_embeddings.items():
        avg_embedding = torch.stack(emb_list).mean(dim=0)
        vocab_embeddings[token_id] = avg_embedding
    
    return vocab_embeddings
```

**Usage:**

```python
# Load model and tokenizer
model = ...
tokenizer = ...

# Prepare corpus (many sentences)
corpus = [
    "The cat sat on the mat",
    "A cat is a pet",
    "The cat was happy",
    "I saw a cat",
    # ... many more examples
]

# Extract vocabulary embeddings
vocab_embeddings = extract_vocabulary_embeddings(model, tokenizer, corpus)

# Get embedding for "cat"
cat_token_id = tokenizer.encode("cat")[0]
cat_embedding = vocab_embeddings[cat_token_id]
# Shape: [d_model] = [512]
```

---

## Comparison Table

| Feature | Input Embeddings | Context-Aware | Vocabulary-Level Contextual |
|---------|------------------|---------------|----------------------------|
| **Context** | ❌ No | ✅ Yes | ✅ Yes (averaged) |
| **Speed** | ✅ Fast (lookup) | ⚠️ Medium (forward pass) | ❌ Slow (many forward passes) |
| **Quality** | ⚠️ Basic | ✅ Excellent | ✅ Excellent |
| **Vocabulary-level** | ✅ Yes | ❌ No | ✅ Yes |
| **Use case** | Fast similarity | Context tasks | Best quality, vocab-level |

---

## Practical Examples

### Example 1: Extract Input Embeddings

```python
import torch
from model import TransformerLLM

# Load checkpoint
checkpoint = torch.load('checkpoints_tinystories/checkpoint_epoch_1.pt', map_location='cpu')

# Recreate model (need to know architecture)
model = TransformerLLM(
    vocab_size=50000,
    d_model=512,
    num_heads=8,
    num_layers=8,
    d_ff=2048
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract input embeddings
word_embeddings = model.token_embedding.weight.data
print(f"Embedding shape: {word_embeddings.shape}")  # [50000, 512]

# Get embedding for token ID 100
token_100_embedding = word_embeddings[100]
print(f"Token 100 embedding: {token_100_embedding.shape}")  # [512]
```

### Example 2: Extract Context-Aware Embeddings

```python
# Add get_embeddings method to model (see Method 1 above)

# Load model
model = ...
tokenizer = ...

# Process a sentence
text = "The cat sat on the mat"
tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([tokens]).to(device)

# Get contextual embeddings
with torch.no_grad():
    embeddings = model.get_embeddings(tokens_tensor)
    # Shape: [1, 6, 512]

# Each position has a context-aware embedding
the_embedding = embeddings[0, 0]   # "The" (context-aware)
cat_embedding = embeddings[0, 1]    # "cat" (context-aware)
sat_embedding = embeddings[0, 2]    # "sat" (context-aware)
```

### Example 3: Compute Word Similarity

```python
import torch.nn.functional as F

def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings."""
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

# Using input embeddings
cat_emb = word_embeddings[tokenizer.encode("cat")[0]]
dog_emb = word_embeddings[tokenizer.encode("dog")[0]]
mat_emb = word_embeddings[tokenizer.encode("mat")[0]]

similarity_cat_dog = cosine_similarity(cat_emb, dog_emb)
similarity_cat_mat = cosine_similarity(cat_emb, mat_emb)

print(f"cat-dog similarity: {similarity_cat_dog:.3f}")  # Should be high
print(f"cat-mat similarity: {similarity_cat_mat:.3f}")    # Should be low
```

### Example 4: Save Embeddings to File

```python
import numpy as np

# Extract embeddings
word_embeddings = model.token_embedding.weight.data.cpu().numpy()

# Save as numpy array
np.save('word_embeddings.npy', word_embeddings)

# Or save as text file (for compatibility)
with open('word_embeddings.txt', 'w') as f:
    for token_id in range(vocab_size):
        embedding = word_embeddings[token_id]
        embedding_str = ' '.join(map(str, embedding))
        f.write(f"{token_id}\t{embedding_str}\n")
```

---

## Which One Should You Use?

### Use **Input Embeddings** (`token_embedding.weight`) when:
- ✅ You need fast vocabulary-level embeddings
- ✅ Context doesn't matter for your task
- ✅ You want one embedding per word type
- ✅ You're doing simple similarity search

### Use **Context-Aware Embeddings** when:
- ✅ Context matters (e.g., "bank" financial vs river)
- ✅ You're doing sentence/document tasks
- ✅ You need the best quality embeddings
- ✅ You can afford a forward pass

### Use **Vocabulary-Level Contextual** when:
- ✅ You need both context-awareness AND vocabulary-level
- ✅ You have a large corpus
- ✅ You want the best possible embeddings
- ✅ You can pre-compute and cache

---

## Implementation: Adding Embedding Extraction to Model

Here's how to modify your model to easily extract embeddings:

```python
# Add to TransformerLLM class in model.py

def get_embeddings(self, x, mask=None):
    """
    Extract context-aware word embeddings.
    
    Args:
        x: Input token indices [batch_size, seq_len]
        mask: Optional attention mask
    
    Returns:
        embeddings: [batch_size, seq_len, d_model]
    """
    batch_size, seq_len = x.size()
    
    if mask is None:
        mask = self.generate_mask(seq_len).to(x.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Token embeddings
    x = self.token_embedding(x) * math.sqrt(self.d_model)
    
    # Positional encoding
    x = self.pos_encoding(x)
    x = self.dropout(x)
    
    # Pass through transformer blocks
    for transformer_block in self.transformer_blocks:
        x = transformer_block(x, mask)
    
    # Final layer norm (this is the embedding!)
    embeddings = self.layer_norm(x)
    
    return embeddings

def get_input_embeddings(self):
    """
    Get static input token embeddings.
    
    Returns:
        embeddings: [vocab_size, d_model]
    """
    return self.token_embedding.weight.data
```

---

## Summary

**Three options for word embeddings:**

1. **`model.token_embedding.weight`** - Static, vocabulary-level, fast
2. **`model.get_embeddings(x)`** - Context-aware, sequence-level, best quality
3. **Averaged contextual embeddings** - Context-aware, vocabulary-level, best overall

**Recommendation:** 
- For most tasks: Use **context-aware embeddings** (Option 2)
- For fast similarity: Use **input embeddings** (Option 1)
- For best quality vocab-level: Use **averaged contextual** (Option 3)






