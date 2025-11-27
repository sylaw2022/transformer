"""
Inference script for autoregressive text generation.
"""

import torch
import argparse
import os

from model import TransformerLLM
from tokenizer import SimpleTokenizer
try:
    from tokenizer_bpe import BPETokenizer
    BPE_AVAILABLE = True
except ImportError:
    BPE_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='Generate text with Transformer LLM')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer file')
    parser.add_argument('--prompt', type=str, default='',
                        help='Input prompt text')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling parameter')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    # Try to detect tokenizer type by checking if it's a HuggingFace tokenizer file
    try:
        # Try loading as BPE tokenizer first (HuggingFace format)
        if BPE_AVAILABLE:
            tokenizer = BPETokenizer.load(args.tokenizer)
            print("Loaded BPE tokenizer")
        else:
            raise ImportError("BPE tokenizer not available")
    except:
        # Fall back to SimpleTokenizer
        tokenizer = SimpleTokenizer.load(args.tokenizer)
        print("Loaded SimpleTokenizer")
    
    # Load checkpoint
    print("Loading model checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_config = checkpoint['model_config']
    
    # Initialize model
    model = TransformerLLM(**model_config).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Vocabulary size: {model_config['vocab_size']}")
    
    # Generate text
    print(f"\nPrompt: '{args.prompt}'")
    print("\nGenerating text...")
    
    generated_text = model.generate(
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )
    
    print(f"\nGenerated text:\n{generated_text}")


if __name__ == '__main__':
    main()



