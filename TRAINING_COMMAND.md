# Training Command Reference

## Quick Start

### Option 1: Use the script (Recommended)
```bash
./start_training.sh
```

### Option 2: Direct command
```bash
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
```

## Command Breakdown

### Required Arguments
- `--data_file`: Path to your training data file
- `--output_dir`: Directory where checkpoints will be saved

### Model Architecture
- `--d_model 512`: Model dimension (embedding size)
- `--num_heads 8`: Number of attention heads
- `--num_layers 8`: Number of transformer layers
- `--d_ff 2048`: Feed-forward network dimension
- `--max_seq_len 512`: Maximum sequence length (context window)

### Training Configuration
- `--epochs 50`: Number of training epochs
- `--batch_size 2`: Batch size (adjust based on GPU memory)
- `--learning_rate 6e-5`: Learning rate (0.00006)
- `--scheduler cosine_warmup`: Learning rate scheduler
- `--warmup_epochs 5`: Warmup epochs for scheduler
- `--device cuda`: Use GPU (or 'cpu' for CPU training)

### Tokenizer Settings
- `--tokenizer_mode bpe`: Use BPE (Byte Pair Encoding) tokenizer
- `--vocab_size 50000`: Vocabulary size

## Important Notes

### 1. Unbuffered Output
Always use `PYTHONUNBUFFERED=1` to see real-time log updates:
```bash
PYTHONUNBUFFERED=1 python3 train.py [arguments]
```

### 2. Logging
Redirect output to a log file:
```bash
python3 train.py [arguments] 2>&1 | tee training.log
```

### 3. Background Training
Run in background with nohup:
```bash
nohup ./start_training.sh > training.log 2>&1 &
```

### 4. Resume from Checkpoint
If you want to resume from a checkpoint:
```bash
python3 train.py [arguments] --resume ./checkpoints_tinystories/checkpoint_epoch_1_40pct.pt
```

### 5. Prevent Suspend
To prevent system suspend during training:
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

Re-enable after training:
```bash
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

## Current Configuration Summary

Based on your previous training run:
- **Model**: 8 layers, 8 heads, 512 dimensions
- **Training**: 50 epochs, batch size 2, learning rate 6e-5
- **Data**: TinyStories dataset with BPE tokenizer (50k vocab)
- **Checkpoints**: Saved at 20%, 40%, 60%, 80%, and 100% of each epoch

## Adjusting Batch Size

If you have more GPU memory, you can increase batch size:
- `--batch_size 4` - Double the batch size
- `--batch_size 8` - Quadruple the batch size

**Note**: You may need to adjust learning rate proportionally:
- Batch size 4 → learning rate ~1.2e-4
- Batch size 8 → learning rate ~2.4e-4

## Monitoring Training

### View real-time log:
```bash
tail -f training_tinystories.log
```

### Check GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Check training process:
```bash
ps aux | grep train.py
```


