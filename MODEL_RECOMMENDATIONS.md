# Model Configuration Recommendations

## Current Model (Small)
- **Parameters**: ~49M
- **d_model**: 512
- **num_layers**: 6
- **num_heads**: 8
- **d_ff**: 2048
- **vocab_size**: 30,000 (word-level)
- **Result**: Loss ~6.3, poor text generation

## Recommended Model Configurations

### 1. Medium Model (Recommended Starting Point)
**Target**: ~150-200M parameters

```bash
python3 train.py \
    --data_file data/wikipedia_processed.txt \
    --output_dir ./checkpoints_bpe_medium \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 1024 \
    --d_model 768 \
    --num_heads 12 \
    --num_layers 12 \
    --d_ff 3072 \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --device cuda
```

**Expected**: ~180M parameters, better text quality

### 2. Large Model (Better Quality)
**Target**: ~350-400M parameters

```bash
python3 train.py \
    --data_file data/wikipedia_processed.txt \
    --output_dir ./checkpoints_bpe_large \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 1024 \
    --d_model 1024 \
    --num_heads 16 \
    --num_layers 16 \
    --d_ff 4096 \
    --epochs 50 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --device cuda
```

**Expected**: ~380M parameters, significantly better quality

### 3. Very Large Model (Research/Production)
**Target**: ~1B+ parameters

```bash
python3 train.py \
    --data_file data/wikipedia_processed.txt \
    --output_dir ./checkpoints_bpe_xlarge \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 2048 \
    --d_model 1536 \
    --num_heads 24 \
    --num_layers 24 \
    --d_ff 6144 \
    --epochs 50 \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --device cuda
```

**Expected**: ~1.2B parameters, requires significant GPU memory

## Key Improvements

### 1. Subword Tokenization (BPE)
- **Why**: Better handling of rare words, no `<UNK>` tokens
- **Implementation**: Use `--tokenizer_mode bpe`
- **Vocab size**: 30k-50k subword tokens (more efficient than word-level)

### 2. Larger Model Dimensions
- **d_model**: 768-1024 (vs 512)
- **num_layers**: 12-16 (vs 6)
- **num_heads**: 12-16 (vs 8)
- **d_ff**: 3072-4096 (vs 2048)

### 3. Larger Context Window
- **max_seq_len**: 1024-2048 (vs 512)
- Allows model to see more context

### 4. Better Dataset
- **Wikipedia**: High-quality, diverse text
- **Size**: 20-50 GB recommended
- **Quality**: Well-structured, clean text

## Memory Requirements

- **Medium Model**: ~8-12 GB GPU memory
- **Large Model**: ~16-24 GB GPU memory
- **Very Large Model**: ~40+ GB GPU memory

## Training Time Estimates

- **Medium Model**: ~2-3 days on single GPU
- **Large Model**: ~5-7 days on single GPU
- **Very Large Model**: ~2-3 weeks on single GPU

## Expected Results

With BPE + larger model + better dataset:
- **Loss**: Should drop to 3.5-4.5 range
- **Text Quality**: Coherent sentences, better grammar
- **Vocabulary Coverage**: No `<UNK>` tokens with BPE

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install tokenizers datasets
   ```

2. **Download Wikipedia dataset**:
   ```bash
   python3 download_wikipedia.py --output data/wikipedia_processed.txt
   ```

3. **Train medium model**:
   ```bash
   ./train_large_model.sh
   ```

4. **Generate text**:
   ```bash
   python3 generate.py \
       --checkpoint checkpoints_bpe_medium/checkpoint_epoch_50.pt \
       --tokenizer checkpoints_bpe_medium/tokenizer.json \
       --prompt "The sun was setting" \
       --max_length 200
   ```


