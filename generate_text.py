#!/usr/bin/env python3
"""
Convenience script for text generation.
Usage: python3 generate_text.py "Your prompt here"
   or: python3 generate_text.py  (uses default prompt)
"""

import sys
import subprocess

if __name__ == '__main__':
    prompt = sys.argv[1] if len(sys.argv) > 1 else "To be or not to be"
    
    # Use the latest checkpoint (d_model=768 model)
    checkpoint = "checkpoints_768/checkpoint_epoch_50.pt"
    tokenizer = "checkpoints_768/tokenizer.json"
    
    cmd = [
        "python3", "generate.py",
        "--checkpoint", checkpoint,
        "--tokenizer", tokenizer,
        "--prompt", prompt,
        "--max_length", "200",
        "--temperature", "0.8",
        "--top_k", "50",
        "--top_p", "0.9",
        "--device", "cpu"
    ]
    
    subprocess.run(cmd)

