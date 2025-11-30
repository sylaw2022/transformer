"""
Inference script for autoregressive text generation with Improved Llama-style Model.
"""

import torch
import argparse
import os

from model_improved import ImprovedTransformerLLM
from tokenizer import SimpleTokenizer
try:
    from tokenizer_bpe import BPETokenizer
    BPE_AVAILABLE = True
except ImportError:
    BPE_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='Generate text with Improved Transformer LLM')
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
    print(f"Loading model checkpoint: {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_config = checkpoint['model_config']
    
    print("Initializing Improved Llama-style Model...")
    # Initialize model
    model = ImprovedTransformerLLM(**model_config).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded. Vocabulary size: {model_config['vocab_size']}")
    
    # Generate text
    print(f"\nPrompt: '{args.prompt}'")
    print("\nGenerating text...")
    
    # Custom generation loop to force output if needed
    model.eval()
    tokens = tokenizer.encode(args.prompt)
    tokens = torch.tensor([tokens], dtype=torch.long).to(args.device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for i in range(args.max_length):
            # Crop context if too long
            context = tokens[:, -model_config['max_seq_len']:]
            
            logits = model(context)
            next_token_logits = logits[0, -1, :] / args.temperature
            
            # Prevent EOS generation for the first 10 tokens to see something
            if i < 10 and hasattr(tokenizer, 'eos_token_id'):
                 next_token_logits[tokenizer.eos_token_id] = float('-inf')

            # Filter
            if args.top_k > 0:
                v, _ = torch.topk(next_token_logits, args.top_k)
                next_token_logits[next_token_logits < v[[-1]]] = float('-inf')
                
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())
            
            if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                break
                
    generated_text = tokenizer.decode(tokens[0].cpu().tolist())
    print(f"\nGenerated text:\n{generated_text}")


if __name__ == '__main__':
    main()
