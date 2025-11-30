#!/usr/bin/env python3
"""
Generate text using prompts extracted from the training data.
This helps test if the model has learned patterns from the training data.
"""

import random
import subprocess
import sys
import os

def extract_prompt_from_training_data(data_file, num_prompts=1, prompt_length=25):
    """Extract random prompts from training data"""
    prompts = []
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and len(line.strip()) > 50]
        
        if not lines:
            print(f"Error: No suitable lines found in {data_file}")
            return []
        
        # Get random samples
        samples = random.sample(lines, min(num_prompts, len(lines)))
        
        for sample in samples:
            # Extract first N words as prompt
            words = sample.split()[:prompt_length]
            prompt = ' '.join(words)
            prompts.append({
                'prompt': prompt,
                'original': sample[:200] + '...' if len(sample) > 200 else sample
            })
        
        return prompts
    except FileNotFoundError:
        print(f"Error: Training data file not found: {data_file}")
        return []
    except Exception as e:
        print(f"Error reading training data: {e}")
        return []

def generate_text(checkpoint, tokenizer, prompt, max_length=200, temperature=0.8, 
                 top_k=50, top_p=0.9, device='cuda'):
    """Generate text using the model"""
    cmd = [
        "python3", "generate.py",
        "--checkpoint", checkpoint,
        "--tokenizer", tokenizer,
        "--prompt", prompt,
        "--max_length", str(max_length),
        "--temperature", str(temperature),
        "--top_k", str(top_k),
        "--top_p", str(top_p),
        "--device", device
    ]
    
    print(f"\n{'='*80}")
    print(f"PROMPT (from training data): {prompt}")
    print(f"{'='*80}\n")
    
    subprocess.run(cmd)

def main():
    # Configuration
    data_file = "data/tinystories_processed.txt"
    checkpoint_dir = "checkpoints_tinystories"
    tokenizer = f"{checkpoint_dir}/tokenizer.json"
    
    # Find latest checkpoint
    import glob
    checkpoints = glob.glob(f"{checkpoint_dir}/*.pt")
    if not checkpoints:
        print(f"Error: No checkpoints found in {checkpoint_dir}")
        sys.exit(1)
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Using checkpoint: {latest_checkpoint}")
    
    # Parse command line arguments
    num_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    prompt_length = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    temperature = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    
    # Extract prompts from training data
    print(f"Extracting {num_prompts} prompt(s) from training data...")
    prompts = extract_prompt_from_training_data(data_file, num_prompts, prompt_length)
    
    if not prompts:
        sys.exit(1)
    
    # Generate text for each prompt
    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n{'#'*80}")
        print(f"# Generation {i}/{len(prompts)}")
        print(f"{'#'*80}")
        print(f"\nOriginal training sample (first 200 chars):")
        print(f"{prompt_data['original']}\n")
        
        generate_text(
            latest_checkpoint,
            tokenizer,
            prompt_data['prompt'],
            max_length=max_length,
            temperature=temperature
        )
        
        if i < len(prompts):
            print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()



