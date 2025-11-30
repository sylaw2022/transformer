#!/bin/bash

# Configuration
DATA_FILE="data/tinystories_processed.txt"
OUTPUT_DIR="./checkpoints_improved"
VOCAB_SIZE=50000

# Model Architecture (Llama-style)
D_MODEL=512
NUM_HEADS=8
NUM_LAYERS=8
D_FF=2048  # SwiGLU uses a different internal size, but this input param is fine
MAX_SEQ_LEN=512

# Training Hyperparameters
EPOCHS=50
BATCH_SIZE=4  # Flash Attention is more memory efficient, might allow larger batches
LEARNING_RATE=3e-4 # Slightly higher LR often works with RMSNorm/SwiGLU
WARMUP_EPOCHS=3

echo "Starting training with IMPROVED architecture..."
echo "Model: Llama-style (RoPE, RMSNorm, SwiGLU, FlashAttention)"
echo "Output Directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
PYTHONUNBUFFERED=1 python3 train_improved.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --tokenizer_mode bpe \
    --vocab_size "$VOCAB_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --d_model "$D_MODEL" \
    --num_heads "$NUM_HEADS" \
    --num_layers "$NUM_LAYERS" \
    --d_ff "$D_FF" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --scheduler cosine_warmup \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --device cuda \
    2>&1 | tee training_improved.log


