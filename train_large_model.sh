#!/bin/bash
# Training script for larger model with BPE tokenization
# This uses a larger model architecture and subword tokenization

# Model configuration:
# - d_model: 768 (increased from 512)
# - num_layers: 12 (increased from 6)
# - num_heads: 12 (increased from 8)
# - d_ff: 3072 (increased from 2048)
# - vocab_size: 50000 (BPE subword tokens)
# - batch_size: 4 (reduced due to larger model)

python3 train.py \
    --data_file data/wikipedia_processed.txt \
    --output_dir ./checkpoints_bpe_large \
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
    --device cuda \
    2>&1 | tee training_bpe_large.log


