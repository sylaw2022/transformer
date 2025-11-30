#!/bin/bash
# Script to start training with proper configuration
# This prevents suspend and ensures unbuffered output

# Disable suspend during training (optional - comment out if you want suspend enabled)
# sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

# Start training with unbuffered output
PYTHONUNBUFFERED=1 python3 train.py \
    --data_file data/tinystories_processed.txt \
    --output_dir ./checkpoints_tinystories \
    --tokenizer_mode bpe \
    --vocab_size 50000 \
    --max_seq_len 512 \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 8 \
    --d_ff 2048 \
    --epochs 50 \
    --batch_size 2 \
    --learning_rate 6e-5 \
    --scheduler cosine_warmup \
    --warmup_epochs 5 \
    --device cuda \
    2>&1 | tee training_tinystories.log

# Re-enable suspend after training (if you disabled it above)
# sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target





