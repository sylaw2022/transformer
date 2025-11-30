# How Backpropagation Updates Q, K, V Using Next Token Prediction Error

## Overview

This document explains how the error from predicting the next token flows backward through the network to update the Query (Q), Key (K), and Value (V) weight matrices in self-attention.

---

## Table of Contents

1. [The Complete Flow](#flow)
2. [Forward Pass: Step-by-Step](#forward)
3. [Loss Computation](#loss)
4. [Backward Pass: Gradient Flow](#backward)
5. [Concrete Numerical Example](#example)
6. [How Gradients Update W_q, W_k, W_v](#updates)
7. [Why This Learns Relationships](#why)

---

## 1. The Complete Flow {#flow}

```
Input: "The cat sat"
Target: "on"

Forward Pass:
  Input tokens → Embeddings → Q, K, V → Attention → Output → Logits → Loss

Backward Pass:
  Loss → Gradients → Update W_q, W_k, W_v
```

---

## 2. Forward Pass: Step-by-Step {#forward}

### Example Sentence
```
Input sequence:  ["The", "cat", "sat"]
Target sequence: ["cat", "sat", "on"]  (shifted by 1 for next token prediction)
```

### Step 1: Token Embeddings

**From model.py line 260:**
```python
x = self.token_embedding(inputs) * math.sqrt(self.d_model)
```

**Result:** Each word becomes a vector
```
"The" → e₁ = [0.1, 0.3, -0.2, ..., 0.5]  (d_model dimensions, e.g., 512)
"cat" → e₂ = [0.4, -0.1, 0.6, ..., -0.3]
"sat" → e₃ = [-0.2, 0.5, 0.1, ..., 0.4]
```

### Step 2: Positional Encoding

**From model.py line 263:**
```python
x = self.pos_encoding(x)
```

**Result:** Position information added
```
x₁ = e₁ + pos_0  (position 0)
x₂ = e₂ + pos_1  (position 1)
x₃ = e₃ + pos_2  (position 2)
```

### Step 3: Compute Q, K, V

**From model.py lines 48-50:**
```python
Q = self.w_q(x)  # Query: What each word is looking for
K = self.w_k(x)  # Key: What each word provides
V = self.w_v(x)  # Value: What each word contains
```

**Mathematical operation:**
```
Q = X × W_q^T  (matrix multiplication)
K = X × W_k^T
V = X × W_v^T
```

Where:
- `X` = [x₁, x₂, x₃] shape: [3, d_model]
- `W_q`, `W_k`, `W_v` shape: [d_model, d_model]

**Result:**
```
Q = [q₁, q₂, q₃]  # Query vectors
K = [k₁, k₂, k₃]  # Key vectors
V = [v₁, v₂, v₃]  # Value vectors
```

### Step 4: Compute Attention Scores

**From model.py line 32:**
```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
```

**Mathematical operation:**
```
scores[i, j] = (Q[i] · K[j]) / √d_k
```

**Result: Attention score matrix**
```
        k₁    k₂    k₃
q₁    [s₁₁,  s₁₂,  s₁₃]  ← How relevant each word is to "The"
q₂    [s₂₁,  s₂₂,  s₂₃]  ← How relevant each word is to "cat"
q₃    [s₃₁,  s₃₂,  s₃₃]  ← How relevant each word is to "sat"
```

**Example values (before training, random):**
```
        k₁    k₂    k₃
q₁    [0.5,  0.2,  0.1]
q₂    [0.3,  0.4,  0.2]
q₃    [0.1,  0.3,  0.5]
```

### Step 5: Apply Softmax

**From model.py line 38:**
```python
attention_weights = F.softmax(scores, dim=-1)
```

**Mathematical operation:**
```
attention_weights[i, j] = exp(scores[i, j]) / Σₖ exp(scores[i, k])
```

**Result:**
```
        k₁    k₂    k₃
q₁    [0.55, 0.30, 0.15]  ← "The" attends to [itself, "cat", "sat"]
q₂    [0.30, 0.45, 0.25]  ← "cat" attends to ["The", itself, "sat"]
q₃    [0.20, 0.35, 0.45]  ← "sat" attends to ["The", "cat", itself]
```

### Step 6: Weighted Sum of Values

**From model.py line 41:**
```python
output = torch.matmul(attention_weights, V)
```

**Mathematical operation:**
```
output[i] = Σⱼ (attention_weights[i, j] × V[j])
```

**Result:**
```
output₁ = 0.55×v₁ + 0.30×v₂ + 0.15×v₃  ← Context-aware representation of "The"
output₂ = 0.30×v₁ + 0.45×v₂ + 0.25×v₃  ← Context-aware representation of "cat"
output₃ = 0.20×v₁ + 0.35×v₂ + 0.45×v₃  ← Context-aware representation of "sat"
```

### Step 7: Final Output Projection

**From model.py line 272:**
```python
logits = self.fc_out(x)
```

**Result:** Logits for vocabulary prediction
```
logits[position_2] = [logit_0, logit_1, ..., logit_vocab_size]
                     ↑ probability distribution over vocabulary
```

**For position 2 ("sat"), we predict the next token:**
```
logits["sat"] = [0.1, 0.05, 0.3, 0.2, 0.15, ...]
                 ↑    ↑     ↑    ↑    ↑
                "the" "cat" "on" "mat" "was" ...
```

---

## 3. Loss Computation {#loss}

### Cross-Entropy Loss

**From train.py line 90:**
```python
loss = criterion(logits, targets)
```

**Target:** For position 2 ("sat"), the next token should be "on" (token_id = 42)

**Mathematical operation:**
```
loss = -log(softmax(logits["sat"])[42])
     = -log(exp(logits["sat"][42]) / Σᵢ exp(logits["sat"][i]))
```

**Example:**
```
logits["sat"] = [0.1, 0.05, 0.3, 0.2, 0.15, ..., 0.4, ...]
                              ↑                    ↑
                           "on" (target)      other tokens

softmax(logits["sat"])[42] = exp(0.4) / Σ exp(logits) = 0.15
loss = -log(0.15) = 1.90
```

**Interpretation:**
- Low probability (0.15) for correct token → High loss (1.90)
- Model needs to increase probability for "on" when "sat" is the context

---

## 4. Backward Pass: Gradient Flow {#backward}

### Step 1: Loss Gradient

**Initial gradient:**
```
∂loss/∂loss = 1.0
```

### Step 2: Gradient Through Output Layer

**Gradient flows to logits:**
```
∂loss/∂logits["sat"][i] = softmax(logits["sat"])[i] - (1 if i == 42 else 0)
```

**For our example:**
```
∂loss/∂logits["sat"][42] = 0.15 - 1.0 = -0.85  ← Large negative gradient!
∂loss/∂logits["sat"][j]  = softmax[j] - 0.0    ← Small positive gradients
```

**Interpretation:**
- Large negative gradient for target token → "increase this logit!"
- Small positive gradients for other tokens → "decrease these logits!"

### Step 3: Gradient Through Transformer Layers

**Gradient flows backward through:**
```
Loss → fc_out → LayerNorm → TransformerBlock_N → ... → TransformerBlock_1 → Attention
```

**At each transformer block:**
```
∂loss/∂attention_output = (gradient from above) × (residual connection)
```

### Step 4: Gradient Through Attention Output

**From attention output to attention weights:**
```
∂loss/∂attention_weights[i, j] = (gradient from above) · V[j]
```

**This tells us:**
- Which attention weights need to change
- How much each attention relationship matters

**Example for position 2 ("sat"):**
```
∂loss/∂attention_weights[2, 0] = gradient · v₁  (attention to "The")
∂loss/∂attention_weights[2, 1] = gradient · v₂  (attention to "cat")
∂loss/∂attention_weights[2, 2] = gradient · v₃  (attention to "sat")
```

### Step 5: Gradient Through Attention Scores

**From attention weights to scores:**
```
∂loss/∂scores[i, j] = attention_weights[i, j] × (∂loss/∂attention_weights[i, j] - Σₖ attention_weights[i, k] × ∂loss/∂attention_weights[i, k])
```

**This is the softmax backward pass:**
- Gradients flow through the softmax function
- Adjusts which word pairs should have higher/lower attention

**Example:**
```
If model should attend more to "cat" when processing "sat":
  ∂loss/∂scores[2, 1] = large positive value  ← Increase attention to "cat"
  ∂loss/∂scores[2, 0] = negative value       ← Decrease attention to "The"
  ∂loss/∂scores[2, 2] = negative value       ← Decrease attention to itself
```

### Step 6: Gradient Through Q and K

**From scores to Q and K:**
```
∂loss/∂scores[i, j] = (Q[i] · K[j]) / √d_k
```

**Gradients:**
```
∂loss/∂Q[i] = Σⱼ (∂loss/∂scores[i, j] × K[j]) / √d_k
∂loss/∂K[j] = Σᵢ (∂loss/∂scores[i, j] × Q[i]) / √d_k
```

**Interpretation:**
- Gradient for Q[i]: How should word i's query change?
- Gradient for K[j]: How should word j's key change?

**Example:**
```
For "sat" (position 2) to better attend to "cat" (position 1):
  ∂loss/∂Q[2] = large component in direction of K[1]  ← Make Q["sat"] more like K["cat"]
  ∂loss/∂K[1] = large component in direction of Q[2]  ← Make K["cat"] more like Q["sat"]
```

### Step 7: Gradient Through W_q, W_k, W_v

**From Q, K, V to weight matrices:**
```
Q = X × W_q^T
K = X × W_k^T
V = X × W_v^T
```

**Gradients:**
```
∂loss/∂W_q = X^T × (∂loss/∂Q)
∂loss/∂W_k = X^T × (∂loss/∂K)
∂loss/∂W_v = X^T × (∂loss/∂V)
```

**This is the key step!** The gradients tell us how to update the weight matrices.

---

## 5. Concrete Numerical Example {#example}

Let's trace through a complete example with actual numbers.

### Setup

**Input:** "The cat sat"
**Target:** "on" (next token after "sat")
**Configuration:** d_model = 4 (simplified for example)

### Forward Pass

#### Step 1: Embeddings
```
x₁ = [0.1, 0.3, -0.2, 0.5]  ("The")
x₂ = [0.4, -0.1, 0.6, -0.3]  ("cat")
x₃ = [-0.2, 0.5, 0.1, 0.4]   ("sat")
```

#### Step 2: Compute Q, K, V

**Initial W_q (random):**
```
W_q = [[0.1, 0.2, -0.1, 0.3],
       [0.2, -0.1, 0.3, 0.1],
       [-0.1, 0.3, 0.2, -0.2],
       [0.3, 0.1, -0.2, 0.2]]
```

**Compute Q:**
```
Q = X × W_q^T

q₁ = x₁ × W_q^T = [0.1, 0.3, -0.2, 0.5] × W_q^T = [0.25, 0.15, 0.05, 0.20]
q₂ = x₂ × W_q^T = [0.4, -0.1, 0.6, -0.3] × W_q^T = [0.10, 0.35, 0.20, -0.05]
q₃ = x₃ × W_q^T = [-0.2, 0.5, 0.1, 0.4] × W_q^T = [0.15, 0.10, 0.25, 0.30]
```

**Similarly for K and V (with W_k and W_v):**
```
k₁ = [0.20, 0.10, 0.15, 0.25]
k₂ = [0.15, 0.30, 0.10, 0.20]
k₃ = [0.10, 0.20, 0.25, 0.15]

v₁ = [0.30, 0.20, 0.10, 0.40]
v₂ = [0.20, 0.30, 0.40, 0.10]
v₃ = [0.10, 0.40, 0.30, 0.20]
```

#### Step 3: Compute Attention Scores

**For position 2 ("sat"):**
```
scores[2, 0] = (q₃ · k₁) / √4 = (0.15×0.20 + 0.10×0.10 + 0.25×0.15 + 0.30×0.25) / 2
            = 0.0875 / 2 = 0.04375

scores[2, 1] = (q₃ · k₂) / √4 = (0.15×0.15 + 0.10×0.30 + 0.25×0.10 + 0.30×0.20) / 2
            = 0.1025 / 2 = 0.05125

scores[2, 2] = (q₃ · k₃) / √4 = (0.15×0.10 + 0.10×0.20 + 0.25×0.25 + 0.30×0.15) / 2
            = 0.1125 / 2 = 0.05625
```

#### Step 4: Apply Softmax

```
exp(0.04375) = 1.0447
exp(0.05125) = 1.0526
exp(0.05625) = 1.0579
sum = 3.1552

attention_weights[2, 0] = 1.0447 / 3.1552 = 0.331
attention_weights[2, 1] = 1.0526 / 3.1552 = 0.334
attention_weights[2, 2] = 1.0579 / 3.1552 = 0.335
```

#### Step 5: Weighted Sum

```
output₃ = 0.331×v₁ + 0.334×v₂ + 0.335×v₃
        = 0.331×[0.30, 0.20, 0.10, 0.40] + 
          0.334×[0.20, 0.30, 0.40, 0.10] + 
          0.335×[0.10, 0.40, 0.30, 0.20]
        = [0.200, 0.300, 0.267, 0.233]
```

#### Step 6: Final Prediction

**After passing through remaining layers:**
```
logits = [0.1, 0.05, 0.3, 0.2, 0.15, ..., 0.4, ...]
                              ↑                    ↑
                           "on" (target)      other tokens

softmax(logits)[42] = 0.15  (15% probability for "on")
loss = -log(0.15) = 1.90
```

---

## 6. Backward Pass: Detailed Gradient Flow {#backward}

### Step 1: Loss Gradient

```
∂loss/∂loss = 1.0
```

### Step 2: Gradient Through Output Layer

**Gradient w.r.t. logits:**
```
∂loss/∂logits[42] = softmax(logits)[42] - 1.0 = 0.15 - 1.0 = -0.85
∂loss/∂logits[j]  = softmax(logits)[j] - 0.0  = small positive values
```

**This flows back to output₃:**
```
∂loss/∂output₃ = [g₁, g₂, g₃, g₄]  (gradients from output layer)
```

### Step 3: Gradient Through Attention Weights

**From output to attention weights:**
```
∂loss/∂attention_weights[2, j] = (∂loss/∂output₃) · vⱼ
```

**Example:**
```
∂loss/∂attention_weights[2, 0] = [g₁, g₂, g₃, g₄] · [0.30, 0.20, 0.10, 0.40]
                                = g₁×0.30 + g₂×0.20 + g₃×0.10 + g₄×0.40

∂loss/∂attention_weights[2, 1] = [g₁, g₂, g₃, g₄] · [0.20, 0.30, 0.40, 0.10]
                                = g₁×0.20 + g₂×0.30 + g₃×0.40 + g₄×0.10

∂loss/∂attention_weights[2, 2] = [g₁, g₂, g₃, g₄] · [0.10, 0.40, 0.30, 0.20]
                                = g₁×0.10 + g₂×0.40 + g₃×0.30 + g₄×0.20
```

**If the model needs to attend more to "cat" (position 1):**
```
∂loss/∂attention_weights[2, 1] = large positive value  ← Increase attention to "cat"
∂loss/∂attention_weights[2, 0] = negative value        ← Decrease attention to "The"
∂loss/∂attention_weights[2, 2] = negative value        ← Decrease attention to itself
```

### Step 4: Gradient Through Attention Scores

**Softmax backward:**
```
∂loss/∂scores[2, j] = attention_weights[2, j] × (
    ∂loss/∂attention_weights[2, j] - 
    Σₖ attention_weights[2, k] × ∂loss/∂attention_weights[2, k]
)
```

**Example calculation:**
```
sum_term = 0.331×(-0.1) + 0.334×(0.5) + 0.335×(-0.2)
         = -0.0331 + 0.167 - 0.067 = 0.0669

∂loss/∂scores[2, 0] = 0.331 × (-0.1 - 0.0669) = 0.331 × (-0.1669) = -0.0552
∂loss/∂scores[2, 1] = 0.334 × (0.5 - 0.0669)  = 0.334 × 0.4331   = 0.1447  ← Large positive!
∂loss/∂scores[2, 2] = 0.335 × (-0.2 - 0.0669) = 0.335 × (-0.2669) = -0.0894
```

**Interpretation:**
- Large positive gradient for scores[2, 1] → Increase similarity between q₃ and k₂
- Negative gradients for scores[2, 0] and scores[2, 2] → Decrease these similarities

### Step 5: Gradient Through Q and K

**From scores to Q:**
```
scores[2, j] = (q₃ · kⱼ) / √d_k

∂loss/∂q₃ = Σⱼ (∂loss/∂scores[2, j] × kⱼ) / √d_k
           = (-0.0552×k₁ + 0.1447×k₂ + -0.0894×k₃) / 2
           = (-0.0552×[0.20, 0.10, 0.15, 0.25] + 
               0.1447×[0.15, 0.30, 0.10, 0.20] + 
              -0.0894×[0.10, 0.20, 0.25, 0.15]) / 2
           = [0.0046, 0.0147, -0.0059, 0.0012] / 2
           = [0.0023, 0.0074, -0.0030, 0.0006]
```

**From scores to K:**
```
∂loss/∂k₁ = Σᵢ (∂loss/∂scores[i, 1] × qᵢ) / √d_k
           = (small values from other positions) / 2
           ≈ [0.001, 0.002, -0.001, 0.001]

∂loss/∂k₂ = Σᵢ (∂loss/∂scores[i, 2] × qᵢ) / √d_k
           = (0.1447×q₃ + ...) / 2
           = [0.0109, 0.0072, 0.0181, 0.0217]  ← Larger gradient!
```

**Interpretation:**
- q₃ gradient points toward k₂ → Make Q["sat"] more similar to K["cat"]
- k₂ gradient points toward q₃ → Make K["cat"] more similar to Q["sat"]
- This strengthens the relationship!

### Step 6: Gradient Through W_q, W_k, W_v

**From Q to W_q:**
```
Q = X × W_q^T

∂loss/∂W_q = X^T × (∂loss/∂Q)
```

**Matrix multiplication:**
```
X^T = [[0.1,  0.4,  -0.2],    ← "The"
       [0.3, -0.1,   0.5],    ← "cat"
       [-0.2, 0.6,   0.1],    ← "sat"
       [0.5, -0.3,   0.4]]

∂loss/∂Q = [[small, small, small],      ← gradients for q₁
             [small, small, small],      ← gradients for q₂
             [0.0023, 0.0074, -0.0030, 0.0006]]  ← gradients for q₃

∂loss/∂W_q = X^T × (∂loss/∂Q)
           = [[w₁₁, w₁₂, w₁₃, w₁₄],
              [w₂₁, w₂₂, w₂₃, w₂₄],
              [w₃₁, w₃₂, w₃₃, w₃₄],
              [w₄₁, w₄₂, w₄₃, w₄₄]]
```

**Example calculation for one element:**
```
∂loss/∂W_q[0, 0] = 0.1×small + 0.4×small + (-0.2)×0.0023
                 ≈ -0.00046
```

**Similarly for W_k and W_v:**
```
∂loss/∂W_k = X^T × (∂loss/∂K)
∂loss/∂W_v = X^T × (∂loss/∂V)
```

### Step 7: Weight Update

**From train.py lines 93-96:**
```python
optimizer.zero_grad()  # Clear previous gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights
```

**Weight update (Adam optimizer example):**
```
W_q_new = W_q_old - learning_rate × (adjusted_gradient)
W_k_new = W_k_old - learning_rate × (adjusted_gradient)
W_v_new = W_v_old - learning_rate × (adjusted_gradient)
```

**Example:**
```
learning_rate = 0.0001

W_q[0, 0] = W_q[0, 0] - 0.0001 × (-0.00046)
           = W_q[0, 0] + 0.000000046  ← Small update

W_k[1, 0] = W_k[1, 0] - 0.0001 × (0.0109)
           = W_k[1, 0] - 0.00000109  ← Larger update (more important)
```

---

## 7. How Gradients Update W_q, W_k, W_v {#updates}

### The Update Mechanism

**After backpropagation, we have:**
```
∂loss/∂W_q  (gradients for Query weights)
∂loss/∂W_k  (gradients for Key weights)
∂loss/∂W_v  (gradients for Value weights)
```

**Weight update:**
```
W_q = W_q - α × (∂loss/∂W_q)  (α = learning rate)
W_k = W_k - α × (∂loss/∂W_k)
W_v = W_v - α × (∂loss/∂W_v)
```

### What Gets Updated?

**W_q updates:**
- Learn what information each word should look for
- Example: "sat" learns to look for subject nouns

**W_k updates:**
- Learn what information each word should provide
- Example: "cat" learns to provide entity/subject information

**W_v updates:**
- Learn what content each word should share
- Example: "cat" learns to share semantic entity information

### Example: Learning Subject-Verb Relationship

**Initial state (random weights):**
```
Q["sat"] · K["cat"] = 0.05125  (low similarity)
attention_weights["sat", "cat"] = 0.334  (moderate)
```

**After many training examples:**
```
W_q learns: Q["sat"] should encode "I need my subject"
W_k learns: K["cat"] should encode "I am a noun subject"

Result:
Q["sat"] · K["cat"] = 0.85  (high similarity!)
attention_weights["sat", "cat"] = 0.75  (strong attention)
```

**The model learned the subject-verb relationship!**

---

## 8. Why This Learns Relationships {#why}

### The Learning Signal

**The loss function provides the signal:**
- "When you see 'sat', predict 'on'"
- But to predict correctly, the model needs context
- "sat" alone isn't enough → need to know what sat

**Backpropagation discovers:**
- "cat" is relevant to "sat" (subject-verb)
- Attention to "cat" helps predict "on"
- Gradients strengthen this relationship

### Iterative Refinement

**Over many training examples:**

1. **Example 1:** "The cat sat on..."
   - Model learns: "sat" should attend to "cat"
   - W_q, W_k update to strengthen this

2. **Example 2:** "The dog sat on..."
   - Model learns: "sat" should attend to subject nouns
   - Generalizes the pattern

3. **Example 3:** "The cat ran on..."
   - Model learns: verbs attend to subjects
   - Pattern becomes robust

**Result:** The model learns general linguistic relationships!

### What Relationships Emerge?

**Through training, the model learns:**

1. **Syntactic:**
   - Subject-verb: "cat" ↔ "sat"
   - Adjective-noun: "happy" ↔ "cat"
   - Preposition-object: "on" ↔ "mat"

2. **Semantic:**
   - Synonyms: "cat" ↔ "feline"
   - Related concepts: "cat" ↔ "pet"

3. **Coreference:**
   - Pronouns: "it" ↔ "cat"

4. **Long-range:**
   - Dependencies across many words

---

## 9. Complete Example: Full Forward and Backward Pass

### Input Sequence
```
["The", "cat", "sat"]
```

### Target (Next Token Prediction)
```
Position 0: "The" → predict "cat" ✓
Position 1: "cat" → predict "sat" ✓
Position 2: "sat" → predict "on" ✗ (wrong prediction)
```

### Forward Pass Summary

```
Input: [x₁, x₂, x₃]
  ↓
Q, K, V: [q₁, q₂, q₃], [k₁, k₂, k₃], [v₁, v₂, v₃]
  ↓
Attention scores: scores[2, 1] = 0.05125 (low)
  ↓
Attention weights: attention[2, 1] = 0.334 (moderate)
  ↓
Output: output₃ = weighted combination
  ↓
Logits: logits["on"] = 0.4
  ↓
Loss: -log(0.15) = 1.90
```

### Backward Pass Summary

```
Loss gradient: 1.0
  ↓
Logits gradient: ∂loss/∂logits["on"] = -0.85
  ↓
Output gradient: flows back through layers
  ↓
Attention weights gradient: ∂loss/∂attention[2, 1] = +0.5 (increase!)
  ↓
Scores gradient: ∂loss/∂scores[2, 1] = +0.1447 (increase similarity!)
  ↓
Q, K gradients: 
  ∂loss/∂q₃ points toward k₂
  ∂loss/∂k₂ points toward q₃
  ↓
W_q, W_k gradients:
  ∂loss/∂W_q updates to make Q["sat"] more like K["cat"]
  ∂loss/∂W_k updates to make K["cat"] more like Q["sat"]
  ↓
Weight update:
  W_q = W_q - α × (∂loss/∂W_q)
  W_k = W_k - α × (∂loss/∂W_k)
```

### After Update

**Next forward pass:**
```
Q["sat"] · K["cat"] = 0.85  (increased from 0.05125!)
attention_weights["sat", "cat"] = 0.75  (increased from 0.334!)
```

**Result:**
- Model now pays more attention to "cat" when processing "sat"
- Better context → better prediction of "on"
- Loss decreases!

---

## 10. Key Insights

### Why Backpropagation Works

1. **Error Signal:**
   - Loss tells us: "prediction was wrong"
   - Gradients tell us: "how to fix it"

2. **Gradient Flow:**
   - Gradients flow from loss back to weights
   - Each layer receives "how much to change" signal

3. **Weight Updates:**
   - Weights adjust to minimize loss
   - Over many examples, patterns emerge

4. **Relationship Learning:**
   - When attention helps prediction → gradients strengthen it
   - When attention hurts prediction → gradients weaken it
   - Model learns which relationships matter!

### The Magic

**Through millions of training examples:**
- The model sees: "sat" + "cat" → predict "on" ✓
- The model sees: "sat" + "dog" → predict "on" ✓
- The model sees: "sat" + "chair" → predict "on" ✓

**Pattern emerges:**
- Verbs should attend to their subjects
- W_q, W_k learn this pattern
- Attention weights reflect this relationship

**Result:** The model automatically discovers linguistic relationships without explicit rules!

---

## Summary

**Backpropagation updates Q, K, V by:**

1. **Forward pass:** Compute predictions using current weights
2. **Loss calculation:** Measure error (next token prediction)
3. **Backward pass:** Compute gradients flowing from loss to weights
4. **Gradient computation:** Calculate ∂loss/∂W_q, ∂loss/∂W_k, ∂loss/∂W_v
5. **Weight update:** Adjust weights to reduce loss
6. **Iteration:** Repeat for millions of examples

**The key insight:** Gradients tell the model which attention patterns help prediction, and the weights adjust to strengthen those patterns. Over time, this learns meaningful linguistic relationships!






