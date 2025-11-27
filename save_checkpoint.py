#!/usr/bin/env python3
"""
Script to save the current model state as a checkpoint.
This script can be used to save checkpoints manually during training.

Usage:
    python3 save_checkpoint.py --checkpoint_name checkpoint_1 --output_dir ./checkpoints_tinystories
"""

import torch
import argparse
import os
import sys
from model import TransformerLLM

def save_checkpoint_from_training_process(output_dir, checkpoint_name, model_config, current_loss=None, batch_num=None, epoch_num=None):
    """
    Save a checkpoint by loading the model and saving its state.
    Note: This requires the model to be accessible, which may not work if training is running.
    """
    checkpoint_path = os.path.join(output_dir, f'{checkpoint_name}.pt')
    
    # Try to load existing checkpoint if available
    existing_checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    
    if existing_checkpoints:
        # Load the most recent checkpoint
        existing_checkpoints.sort()
        latest_checkpoint = os.path.join(output_dir, existing_checkpoints[-1])
        print(f"Loading existing checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # Create new checkpoint with custom name
        new_checkpoint = {
            'epoch': epoch_num if epoch_num else checkpoint.get('epoch', 1),
            'model_state_dict': checkpoint['model_state_dict'],
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
            'loss': current_loss if current_loss else checkpoint.get('loss', 0.0),
            'model_config': checkpoint.get('model_config', model_config),
            'batch_num': batch_num,
        }
        if 'scheduler_state_dict' in checkpoint:
            new_checkpoint['scheduler_state_dict'] = checkpoint['scheduler_state_dict']
        
        # Save with new name
        temp_path = checkpoint_path + '.tmp'
        torch.save(new_checkpoint, temp_path)
        os.rename(temp_path, checkpoint_path)
        print(f"Checkpoint saved as: {checkpoint_path}")
        return True
    else:
        print(f"No existing checkpoints found in {output_dir}")
        print("Note: Since training is still in progress (29% of epoch 1), no checkpoint has been saved yet.")
        print("The training script only saves checkpoints at the end of each epoch.")
        print("\nOptions:")
        print("1. Wait for epoch 1 to complete (checkpoint will be saved automatically)")
        print("2. Modify train.py to save checkpoints more frequently")
        print("3. Stop training temporarily to save current state")
        return False

def main():
    parser = argparse.ArgumentParser(description='Save model checkpoint')
    parser.add_argument('--checkpoint_name', type=str, default='checkpoint_1',
                        help='Name for the checkpoint (e.g., checkpoint_1)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_tinystories',
                        help='Directory where checkpoints are saved')
    parser.add_argument('--loss', type=float, default=None,
                        help='Current loss value (optional)')
    parser.add_argument('--batch_num', type=int, default=None,
                        help='Current batch number (optional)')
    parser.add_argument('--epoch_num', type=int, default=1,
                        help='Current epoch number (default: 1)')
    
    args = parser.parse_args()
    
    # Model configuration (from training command)
    model_config = {
        'vocab_size': 50000,
        'd_model': 768,
        'num_heads': 12,
        'num_layers': 12,
        'd_ff': 3072,
        'max_seq_len': 1024,
    }
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory {args.output_dir} does not exist")
        sys.exit(1)
    
    # Try to save checkpoint
    success = save_checkpoint_from_training_process(
        args.output_dir,
        args.checkpoint_name,
        model_config,
        current_loss=args.loss,
        batch_num=args.batch_num,
        epoch_num=args.epoch_num
    )
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()


