"""
Training script for transformer-based LLM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
import gc
import math
import signal
from tqdm import tqdm
import json

from model_improved import ImprovedTransformerLLM
from tokenizer import SimpleTokenizer
try:
    from tokenizer_bpe import BPETokenizer
    BPE_AVAILABLE = True
except ImportError:
    BPE_AVAILABLE = False
    print("Warning: BPE tokenizer not available. Install with: pip install tokenizers")


class TextDataset(Dataset):
    """Dataset for text data."""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Encode text
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input and target (shifted by one)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        # Pad sequences
        pad_length = self.max_length - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
        target_ids = target_ids + [self.tokenizer.pad_token_id] * pad_length
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def save_checkpoint_at_interval(model, optimizer, scheduler, args, vocab_size, epoch, batch_idx, total_batches, loss, interval_pct):
    """Save checkpoint at a specific completion interval."""
    try:
        checkpoint_name = f'checkpoint_epoch_{epoch}_{interval_pct}pct.pt'
        checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
        
        if args.device == 'cuda':
            torch.cuda.synchronize()
        
        temp_path = checkpoint_path + '.tmp'
        checkpoint_dict = {
            'epoch': epoch,
            'batch': batch_idx,
            'total_batches': total_batches,
            'completion_pct': interval_pct,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_config': {
                'vocab_size': vocab_size,
                'd_model': args.d_model,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'd_ff': args.d_ff,
                'max_seq_len': args.max_seq_len,
            }
        }
        if scheduler is not None:
            checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint_dict, temp_path)
        os.rename(temp_path, checkpoint_path)
        print(f"\n[CHECKPOINT] Saved at {interval_pct}% completion: {checkpoint_path} (Loss: {loss:.4f})", flush=True)
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to save checkpoint at {interval_pct}%: {e}", flush=True)
        return False


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, epoch=None, total_epochs=None, 
                args=None, vocab_size=None):
    """Train for one epoch."""
    global global_current_batch
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Create description with epoch number
    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch}/{total_epochs}"
    else:
        desc = "Training"
    
    # Track running average for periodic logging
    running_loss = 0.0
    log_interval = 100  # Log every 100 batches
    current_avg_loss = 0.0  # Track current average loss for checkpoint saving
    
    # Calculate checkpoint intervals (10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%)
    total_batches = len(dataloader)
    checkpoint_intervals = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    checkpoint_batches = [int(total_batches * interval) for interval in checkpoint_intervals]
    checkpoint_saved = {interval: False for interval in checkpoint_intervals}
    
    pbar = tqdm(dataloader, desc=desc)
    for batch_idx, (inputs, targets) in enumerate(pbar, 1):
        global_current_batch = batch_idx
        try:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            # Forward pass
            logits = model(inputs)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            # Calculate loss (ignore padding tokens)
            loss = criterion(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_item = loss.item()
            total_loss += loss_item
            running_loss += loss_item
            num_batches += 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nCUDA out of memory at batch {batch_idx}. Skipping batch...", flush=True)
                if device == 'cuda':
                    torch.cuda.empty_cache()
                continue
            else:
                print(f"\nError at batch {batch_idx}: {e}", flush=True)
                raise
        
        # Log loss periodically
        if batch_idx % log_interval == 0:
            current_avg_loss = running_loss / log_interval
            # Use print with flush to ensure output appears even when piped
            print(f"Batch {batch_idx}/{len(dataloader)}: Loss = {current_avg_loss:.4f}", flush=True)
            running_loss = 0.0
        else:
            # Update current average loss based on running average
            current_avg_loss = running_loss / (batch_idx % log_interval) if (batch_idx % log_interval) > 0 else current_avg_loss
        
        # Save checkpoint at completion intervals (10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%)
        if args is not None and vocab_size is not None:
            for interval_pct, interval_batch in zip(checkpoint_intervals, checkpoint_batches):
                if batch_idx >= interval_batch and not checkpoint_saved[interval_pct]:
                    # Use current average loss or calculate from total so far
                    checkpoint_loss = total_loss / num_batches if num_batches > 0 else current_avg_loss
                    save_checkpoint_at_interval(
                        model, optimizer, scheduler, args, vocab_size,
                        epoch, batch_idx, total_batches, checkpoint_loss, int(interval_pct * 100)
                    )
                    checkpoint_saved[interval_pct] = True
                    break  # Only save one checkpoint per batch
        
        # Save checkpoint on demand (check for trigger file)
        if args is not None:
            checkpoint_trigger = os.path.join(args.output_dir, 'SAVE_CHECKPOINT_NOW')
            if os.path.exists(checkpoint_trigger):
                try:
                    checkpoint_name = 'checkpoint_1'  # Default name
                    if os.path.exists(os.path.join(args.output_dir, 'checkpoint_name.txt')):
                        with open(os.path.join(args.output_dir, 'checkpoint_name.txt'), 'r') as cf:
                            checkpoint_name = cf.read().strip()
                    
                    checkpoint_path = os.path.join(args.output_dir, f'{checkpoint_name}.pt')
                    if args.device == 'cuda':
                        torch.cuda.synchronize()
                    
                    temp_path = checkpoint_path + '.tmp'
                    checkpoint_dict = {
                        'epoch': epoch + 1,
                        'batch': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_avg_loss,
                        'model_config': {
                            'vocab_size': vocab_size,
                            'd_model': args.d_model,
                            'num_heads': args.num_heads,
                            'num_layers': args.num_layers,
                            'd_ff': args.d_ff,
                            'max_seq_len': args.max_seq_len,
                        }
                    }
                    if scheduler is not None:
                        checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(checkpoint_dict, temp_path)
                    os.rename(temp_path, checkpoint_path)
                    os.remove(checkpoint_trigger)  # Remove trigger file
                    print(f"\n[ON-DEMAND] Checkpoint saved to {checkpoint_path} (Loss: {current_avg_loss:.4f})", flush=True)
                except Exception as e:
                    print(f"\nError saving on-demand checkpoint: {e}", flush=True)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Step scheduler at end of epoch
    if scheduler is not None:
        scheduler.step()
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train Transformer LLM')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to text data file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='Feed-forward dimension')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='Maximum sequence length (context window)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'cosine_warmup', 'step', 'none'],
                        help='Learning rate scheduler: cosine, cosine_warmup, step, or none')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of warmup epochs for cosine_warmup scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='Learning rate decay rate for step scheduler')
    parser.add_argument('--lr_decay_epochs', type=int, default=10,
                        help='Decay learning rate every N epochs for step scheduler')
    parser.add_argument('--tokenizer_mode', type=str, default='word',
                        choices=['char', 'word', 'bpe'],
                        help='Tokenizer mode: char, word, or bpe (subword)')
    parser.add_argument('--vocab_size', type=int, default=10000,
                        help='Vocabulary size (None for all tokens, default: 10000 for word-level)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoints_word/checkpoint_epoch_5.pt)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Loaded {len(texts)} text samples")
    
    # Initialize tokenizer
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.json')
    
    # Check if tokenizer already exists
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer from {tokenizer_path}...")
        if args.tokenizer_mode == 'bpe':
            if not BPE_AVAILABLE:
                raise ImportError("BPE tokenizer requires 'tokenizers' library. Install with: pip install tokenizers")
            tokenizer = BPETokenizer.load(tokenizer_path)
            vocab_size = len(tokenizer)
            print(f"BPE Vocabulary size: {vocab_size}")
        else:
            tokenizer = SimpleTokenizer.load(tokenizer_path)
            vocab_size = len(tokenizer)
            print(f"Vocabulary size: {vocab_size}")
    else:
        print("Building tokenizer...")
        if args.tokenizer_mode == 'bpe':
            if not BPE_AVAILABLE:
                raise ImportError("BPE tokenizer requires 'tokenizers' library. Install with: pip install tokenizers")
            tokenizer = BPETokenizer()
            tokenizer.train(texts, vocab_size=args.vocab_size)
            vocab_size = len(tokenizer)
            print(f"BPE Vocabulary size: {vocab_size}")
        else:
            tokenizer = SimpleTokenizer(mode=args.tokenizer_mode)
            tokenizer.build_vocab(texts, vocab_size=args.vocab_size)
            vocab_size = len(tokenizer)
            print(f"Vocabulary size: {vocab_size}")
        
        # Save tokenizer
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_length=args.max_seq_len)
    # Use num_workers=0 to avoid multiprocessing issues, pin_memory=True for faster GPU transfer
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=0, pin_memory=True if args.device == 'cuda' else False)
    
    # Initialize model
    print("Initializing Improved Llama-style Model...")
    model = ImprovedTransformerLLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=0.1
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01
        )
        print(f"Using CosineAnnealingLR scheduler (T_max={args.epochs}, eta_min={args.learning_rate * 0.01})")
    elif args.scheduler == 'cosine_warmup':
        # Custom warmup + cosine annealing
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / args.warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
                return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(f"Using CosineAnnealingLR with warmup (warmup_epochs={args.warmup_epochs})")
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_rate
        )
        print(f"Using StepLR scheduler (step_size={args.lr_decay_epochs}, gamma={args.lr_decay_rate})")
    else:
        print("No learning rate scheduler (using constant learning rate)")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Resumed scheduler state")
            start_epoch = checkpoint['epoch']
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")
    
    # Clear GPU cache before starting training
    if args.device == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("GPU cache cleared. Ready to start training.")
    
    # Global variables for signal handler
    global_save_checkpoint = False
    global_model = model
    global_optimizer = optimizer
    global_scheduler = scheduler
    global_args = args
    global_vocab_size = vocab_size
    global_current_epoch = start_epoch
    global_current_batch = 0
    global_current_loss = 0.0
    
    def save_checkpoint_on_exit(signum, frame):
        """Save checkpoint when interrupted"""
        global global_save_checkpoint, global_model, global_optimizer, global_scheduler
        global global_args, global_vocab_size, global_current_epoch, global_current_batch, global_current_loss
        
        print(f"\n\n[INTERRUPT] Received signal {signum}. Saving checkpoint before exit...", flush=True)
        global_save_checkpoint = True
        
        try:
            checkpoint_name = 'checkpoint_1'
            checkpoint_path = os.path.join(global_args.output_dir, f'{checkpoint_name}.pt')
            
            if global_args.device == 'cuda':
                torch.cuda.synchronize()
            
            temp_path = checkpoint_path + '.tmp'
            checkpoint_dict = {
                'epoch': global_current_epoch + 1,
                'batch': global_current_batch,
                'model_state_dict': global_model.state_dict(),
                'optimizer_state_dict': global_optimizer.state_dict(),
                'loss': global_current_loss,
                'model_config': {
                    'vocab_size': global_vocab_size,
                    'd_model': global_args.d_model,
                    'num_heads': global_args.num_heads,
                    'num_layers': global_args.num_layers,
                    'd_ff': global_args.d_ff,
                    'max_seq_len': global_args.max_seq_len,
                },
                'interrupted': True
            }
            if global_scheduler is not None:
                checkpoint_dict['scheduler_state_dict'] = global_scheduler.state_dict()
            
            torch.save(checkpoint_dict, temp_path)
            os.rename(temp_path, checkpoint_path)
            print(f"[INTERRUPT] Checkpoint saved to {checkpoint_path}", flush=True)
        except Exception as e:
            print(f"[INTERRUPT] Error saving checkpoint: {e}", flush=True)
        
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, save_checkpoint_on_exit)
    signal.signal(signal.SIGINT, save_checkpoint_on_exit)
    print("Signal handlers registered. Press Ctrl+C or send SIGTERM to save checkpoint and exit.")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        global_current_epoch = epoch
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")
        loss = train_epoch(model, dataloader, optimizer, criterion, args.device, scheduler=scheduler, 
                           epoch=epoch + 1, total_epochs=args.epochs, args=args, vocab_size=vocab_size)
        global_current_loss = loss
        print(f"Average loss: {loss:.4f}")
        
        # Save checkpoint (with error handling and sync)
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        try:
            # Ensure GPU operations are complete before saving
            if args.device == 'cuda':
                torch.cuda.synchronize()
            
            # Save checkpoint atomically (write to temp file then rename)
            temp_path = checkpoint_path + '.tmp'
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'model_config': {
                    'vocab_size': vocab_size,
                    'd_model': args.d_model,
                    'num_heads': args.num_heads,
                    'num_layers': args.num_layers,
                    'd_ff': args.d_ff,
                    'max_seq_len': args.max_seq_len,
                }
            }
            if scheduler is not None:
                checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint_dict, temp_path)
            os.rename(temp_path, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            print("Training will continue, but checkpoint was not saved.")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()


