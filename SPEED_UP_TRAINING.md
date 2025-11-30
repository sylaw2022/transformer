# Speeding Up Training with Larger GPU Memory (No Code Changes)

## Quick Answer

**Yes!** You can significantly speed up training with a larger GPU by **increasing the batch size** via command-line arguments. No code changes needed!

---

## Current Training Status

From your training log:
- **Current batch size**: Likely 2 (based on slow training speed ~7 it/s)
- **Model**: 76M parameters
- **Training speed**: ~7 iterations/second
- **Batches per epoch**: 5,706,516
- **Estimated time per epoch**: ~226 hours (9+ days) at current speed

---

## How to Speed Up (No Code Changes)

### Option 1: Increase Batch Size (Easiest & Most Effective)

**Current (likely):**
```bash
python3 train.py \
    --data_file data/tinystories_processed.txt \
    --output_dir ./checkpoints_tinystories \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 512 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 8 \
    --d_ff 2048 \
    --batch_size 2 \  # ← Currently limited by GPU memory
    --device cuda
```

**With larger GPU (e.g., 16GB+):**
```bash
python3 train.py \
    --data_file data/tinystories_processed.txt \
    --output_dir ./checkpoints_tinystories \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 512 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 8 \
    --d_ff 2048 \
    --batch_size 8 \  # ← Increased from 2 to 8 (4x speedup!)
    --device cuda
```

**Speedup calculation:**
- Batch size 2 → 8: **4x faster** (processes 4x more samples per iteration)
- If current: 7 it/s → New: ~28 it/s (theoretical)
- Time per epoch: ~226 hours → **~56 hours** (4x reduction)

### Option 2: Increase Batch Size + Sequence Length

**With very large GPU (24GB+):**
```bash
python3 train.py \
    --data_file data/tinystories_processed.txt \
    --output_dir ./checkpoints_tinystories \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 1024 \  # ← Increased from 512 (better context)
    --d_model 512 \
    --num_heads 8 \
    --num_layers 8 \
    --d_ff 2048 \
    --batch_size 4 \  # ← Can use larger batch with longer sequences
    --device cuda
```

**Trade-off:**
- Longer sequences = better context understanding
- But: Memory scales quadratically with sequence length
- May need to reduce batch size to compensate

---

## Memory Requirements

### Current Configuration (batch_size=2, max_seq_len=512)
```
Model parameters: ~76M
Memory per batch: ~2-3 GB
Total GPU memory needed: ~4-6 GB
```

### With batch_size=8, max_seq_len=512
```
Model parameters: ~76M
Memory per batch: ~8-12 GB
Total GPU memory needed: ~12-16 GB
```

### With batch_size=4, max_seq_len=1024
```
Model parameters: ~76M
Memory per batch: ~8-12 GB (quadratic scaling!)
Total GPU memory needed: ~12-16 GB
```

---

## Recommended Settings by GPU Memory

### 8GB GPU (e.g., RTX 3060, RTX 3070)
```bash
--batch_size 4
--max_seq_len 512
```
**Expected speedup**: 2x

### 16GB GPU (e.g., RTX 3080, RTX 4080, A4000)
```bash
--batch_size 8
--max_seq_len 512
```
**Expected speedup**: 4x

**Or:**
```bash
--batch_size 4
--max_seq_len 1024  # Better context
```
**Expected speedup**: 2x, but better model quality

### 24GB GPU (e.g., RTX 3090, RTX 4090, A5000)
```bash
--batch_size 16
--max_seq_len 512
```
**Expected speedup**: 8x

**Or:**
```bash
--batch_size 8
--max_seq_len 1024  # Best of both worlds
```
**Expected speedup**: 4x, better quality

### 40GB+ GPU (e.g., A100, H100)
```bash
--batch_size 32
--max_seq_len 1024
```
**Expected speedup**: 16x+

---

## How to Find Optimal Batch Size

### Method 1: Start High and Reduce

```bash
# Try batch_size=16 first
python3 train.py ... --batch_size 16

# If you get OOM (Out of Memory) error, reduce:
python3 train.py ... --batch_size 8

# Continue reducing until it works
```

### Method 2: Monitor GPU Memory

```bash
# In another terminal, watch GPU memory:
watch -n 1 nvidia-smi

# Start training and observe memory usage
# Increase batch_size until you use ~90% of GPU memory
```

### Method 3: Use Gradient Accumulation (Advanced)

If you hit memory limits but want larger effective batch size:

**Note:** This would require code changes, but here's the concept:
- Use smaller batch_size (e.g., 4)
- Accumulate gradients over multiple batches
- Update weights less frequently
- Effective batch size = batch_size × accumulation_steps

---

## Expected Speedup Examples

### Example 1: 2x Batch Size Increase

**Before:**
- Batch size: 2
- Iterations/second: 7
- Time per epoch: 226 hours

**After:**
- Batch size: 4
- Iterations/second: ~14
- Time per epoch: **113 hours** (2x faster)

### Example 2: 4x Batch Size Increase

**Before:**
- Batch size: 2
- Iterations/second: 7
- Time per epoch: 226 hours

**After:**
- Batch size: 8
- Iterations/second: ~28
- Time per epoch: **56 hours** (4x faster)

### Example 3: 8x Batch Size Increase

**Before:**
- Batch size: 2
- Iterations/second: 7
- Time per epoch: 226 hours

**After:**
- Batch size: 16
- Iterations/second: ~56
- Time per epoch: **28 hours** (8x faster)

---

## Important Considerations

### 1. Learning Rate Scaling

**Rule of thumb:** When increasing batch size, you may need to scale learning rate:

```
new_lr = base_lr × (new_batch_size / old_batch_size)
```

**Example:**
- Old: batch_size=2, lr=6e-5
- New: batch_size=8
- New lr: 6e-5 × (8/2) = **2.4e-4**

**However:** Your current learning rate (6e-5) is already quite small, so you might be fine without scaling, or use a smaller multiplier like 1.5-2x.

### 2. Gradient Accumulation vs. Larger Batch

**Larger batch (no code change):**
- ✅ Faster training
- ✅ Better GPU utilization
- ❌ Limited by GPU memory

**Gradient accumulation (requires code change):**
- ✅ Can simulate larger batches
- ✅ Works with limited memory
- ❌ Slightly slower (more forward passes)

### 3. Sequence Length Trade-off

**Longer sequences (max_seq_len=1024):**
- ✅ Better context understanding
- ✅ Better model quality
- ❌ More memory (quadratic scaling)
- ❌ May need smaller batch size

**Shorter sequences (max_seq_len=512):**
- ✅ Less memory
- ✅ Can use larger batch size
- ✅ Faster training
- ❌ Less context

---

## Practical Example: Resuming Training with Larger Batch

If you want to resume training with a larger batch size:

```bash
python3 train.py \
    --data_file data/tinystories_processed.txt \
    --output_dir ./checkpoints_tinystories \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 512 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 8 \
    --d_ff 2048 \
    --batch_size 8 \  # ← Increased from 2
    --learning_rate 1.2e-4 \  # ← Slightly increased (2x)
    --scheduler cosine_warmup \
    --warmup_epochs 5 \
    --epochs 50 \
    --device cuda \
    --resume checkpoints_tinystories/checkpoint_epoch_1.pt  # Resume from checkpoint
```

---

## Monitoring Training Speed

### Check Current Speed

```bash
# Watch training log
tail -f training_tinystories.log | grep "it/s"
```

### Calculate Time Remaining

```
Time per epoch = (total_batches / iterations_per_second) / 3600 hours
```

**Example:**
- Total batches: 5,706,516
- Speed: 28 it/s (with batch_size=8)
- Time: 5,706,516 / 28 / 3600 = **56.5 hours per epoch**

---

## Summary

### Yes, you can speed up training without code changes!

**How:**
1. Increase `--batch_size` parameter (2 → 4, 8, 16, etc.)
2. Optionally increase `--max_seq_len` (512 → 1024) for better quality

**Expected speedup:**
- 2x batch size → 2x faster
- 4x batch size → 4x faster
- 8x batch size → 8x faster

**Requirements:**
- Larger GPU memory (8GB+ recommended)
- May need to adjust learning rate slightly

**No code changes needed** - just change command-line arguments!

---

## Quick Start

**For 16GB GPU:**
```bash
python3 train.py \
    --data_file data/tinystories_processed.txt \
    --output_dir ./checkpoints_tinystories \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 512 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 8 \
    --d_ff 2048 \
    --batch_size 8 \  # ← Changed from 2
    --learning_rate 1.2e-4 \  # ← Slightly increased
    --scheduler cosine_warmup \
    --warmup_epochs 5 \
    --epochs 50 \
    --device cuda \
    --resume checkpoints_tinystories/checkpoint_epoch_1.pt
```

**Expected result:**
- Training speed: ~28 it/s (4x faster)
- Time per epoch: ~56 hours (down from 226 hours)
- Total training time: Much shorter!






