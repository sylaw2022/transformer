#!/usr/bin/env python3
"""
Convenience script for text generation.
Usage: ./generate_text.sh "Your prompt here"
"""

import sys
import subprocess

if __name__ == '__main__':
    prompt = sys.argv[1] if len(sys.argv) > 1 else "To be or not to be"
    
    # Use the latest checkpoint
    checkpoint = "checkpoints/checkpoint_epoch_5.pt"
    tokenizer = "checkpoints/tokenizer.json"
    
    cmd = [
        "python3", "generate.py",
        "--checkpoint", checkpoint,
        "--tokenizer", tokenizer,
        "--prompt", prompt,
        "--max_length", "200",
        "--temperature", "0.8",
        "--top_k", "50",
        "--top_p", "0.9"
    ]
    
    subprocess.run(cmd)



