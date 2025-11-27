"""
Script to extract word embeddings from a trained Transformer LLM model.

Usage:
    python extract_embeddings.py --checkpoint checkpoints_tinystories/checkpoint_epoch_1.pt \
                                  --tokenizer checkpoints_tinystories/tokenizer.json \
                                  --output embeddings.npy \
                                  --mode input  # or 'contextual' or 'vocab'
"""

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from model import TransformerLLM
from tokenizer_bpe import BPETokenizer


def extract_input_embeddings(model):
    """
    Extract static input token embeddings.
    
    Returns:
        embeddings: numpy array of shape [vocab_size, d_model]
    """
    embeddings = model.token_embedding.weight.data.cpu().numpy()
    return embeddings


def extract_contextual_embeddings(model, tokenizer, texts, device='cpu'):
    """
    Extract context-aware embeddings for given texts.
    
    Args:
        model: TransformerLLM model
        tokenizer: Tokenizer instance
        texts: List of text strings
        device: Device to run on
    
    Returns:
        embeddings: List of numpy arrays, one per text
        Each array has shape [seq_len, d_model]
    """
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenize
            tokens = tokenizer.encode(text)
            tokens_tensor = torch.tensor([tokens]).to(device)
            
            # Get embeddings using modified forward
            batch_size, seq_len = tokens_tensor.size()
            
            # Generate mask
            mask = model.generate_mask(seq_len).to(device)
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Forward pass to get embeddings
            x = model.token_embedding(tokens_tensor) * (model.d_model ** 0.5)
            x = model.pos_encoding(x)
            x = model.dropout(x)
            
            for transformer_block in model.transformer_blocks:
                x = transformer_block(x, mask)
            
            # Final layer norm (this is the embedding!)
            embeddings = model.layer_norm(x)
            
            # Convert to numpy
            embeddings_np = embeddings[0].cpu().numpy()  # [seq_len, d_model]
            all_embeddings.append(embeddings_np)
    
    return all_embeddings


def extract_vocabulary_embeddings(model, tokenizer, corpus, device='cpu'):
    """
    Extract vocabulary-level embeddings by averaging contextual embeddings.
    
    Args:
        model: TransformerLLM model
        tokenizer: Tokenizer instance
        corpus: List of sentences
        device: Device to run on
    
    Returns:
        vocab_embeddings: Dict mapping token_id to averaged embedding [d_model]
    """
    model.eval()
    
    # Track embeddings for each token
    token_embeddings = {}  # token_id -> list of embeddings
    token_counts = {}       # token_id -> count
    
    with torch.no_grad():
        for text in corpus:
            tokens = tokenizer.encode(text)
            tokens_tensor = torch.tensor([tokens]).to(device)
            
            batch_size, seq_len = tokens_tensor.size()
            mask = model.generate_mask(seq_len).to(device)
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Get contextual embeddings
            x = model.token_embedding(tokens_tensor) * (model.d_model ** 0.5)
            x = model.pos_encoding(x)
            x = model.dropout(x)
            
            for transformer_block in model.transformer_blocks:
                x = transformer_block(x, mask)
            
            embeddings = model.layer_norm(x)
            embeddings_np = embeddings[0].cpu()  # [seq_len, d_model]
            
            # Store embeddings for each token
            for pos, token_id in enumerate(tokens):
                if token_id not in token_embeddings:
                    token_embeddings[token_id] = []
                    token_counts[token_id] = 0
                
                token_embeddings[token_id].append(embeddings_np[pos])
                token_counts[token_id] += 1
    
    # Average embeddings for each token
    vocab_embeddings = {}
    for token_id, emb_list in token_embeddings.items():
        avg_embedding = torch.stack(emb_list).mean(dim=0).numpy()
        vocab_embeddings[token_id] = avg_embedding
    
    return vocab_embeddings


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = TransformerLLM(
            vocab_size=config.get('vocab_size', 50000),
            d_model=config.get('d_model', 512),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 8),
            d_ff=config.get('d_ff', 2048),
            max_seq_len=config.get('max_seq_len', 1024),
            dropout=config.get('dropout', 0.1)
        )
    else:
        # Default config (adjust if needed)
        print("Warning: No config in checkpoint, using defaults")
        model = TransformerLLM(
            vocab_size=50000,
            d_model=512,
            num_heads=8,
            num_layers=8,
            d_ff=2048
        )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    print("Model loaded successfully!")
    return model


def main():
    parser = argparse.ArgumentParser(description='Extract word embeddings from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Path to tokenizer file')
    parser.add_argument('--output', type=str, default='embeddings.npy',
                        help='Output file path')
    parser.add_argument('--mode', type=str, choices=['input', 'contextual', 'vocab'],
                        default='input', help='Embedding extraction mode')
    parser.add_argument('--texts', type=str, nargs='+', default=None,
                        help='Texts for contextual/vocab mode (optional)')
    parser.add_argument('--corpus_file', type=str, default=None,
                        help='File with corpus (one sentence per line) for vocab mode')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Load model
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model = load_model(args.checkpoint, device)
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print("Tokenizer loaded!")
    
    # Extract embeddings based on mode
    if args.mode == 'input':
        print("Extracting input embeddings...")
        embeddings = extract_input_embeddings(model)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Save as numpy array
        np.save(args.output, embeddings)
        print(f"Saved to {args.output}")
        
        # Also save metadata
        metadata = {
            'shape': embeddings.shape,
            'vocab_size': embeddings.shape[0],
            'd_model': embeddings.shape[1],
            'mode': 'input'
        }
        metadata_path = args.output.replace('.npy', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_path}")
    
    elif args.mode == 'contextual':
        if args.texts is None:
            print("Error: --texts required for contextual mode")
            return
        
        print(f"Extracting contextual embeddings for {len(args.texts)} texts...")
        embeddings = extract_contextual_embeddings(model, tokenizer, args.texts, device)
        
        # Save as list of numpy arrays
        embeddings_dict = {i: emb for i, emb in enumerate(embeddings)}
        np.savez(args.output, **embeddings_dict)
        print(f"Saved {len(embeddings)} contextual embeddings to {args.output}")
    
    elif args.mode == 'vocab':
        # Load corpus
        if args.corpus_file:
            print(f"Loading corpus from {args.corpus_file}...")
            with open(args.corpus_file, 'r', encoding='utf-8') as f:
                corpus = [line.strip() for line in f if line.strip()]
        elif args.texts:
            corpus = args.texts
        else:
            print("Error: --corpus_file or --texts required for vocab mode")
            return
        
        print(f"Extracting vocabulary embeddings from {len(corpus)} sentences...")
        vocab_embeddings = extract_vocabulary_embeddings(model, tokenizer, corpus, device)
        
        # Convert to numpy array (vocab_size x d_model)
        vocab_size = len(vocab_embeddings)
        d_model = next(iter(vocab_embeddings.values())).shape[0]
        embeddings_array = np.zeros((vocab_size, d_model))
        
        for token_id, embedding in vocab_embeddings.items():
            if token_id < vocab_size:
                embeddings_array[token_id] = embedding
        
        np.save(args.output, embeddings_array)
        print(f"Saved vocabulary embeddings to {args.output}")
        print(f"Shape: {embeddings_array.shape}, Tokens covered: {len(vocab_embeddings)}")
    
    print("Done!")


if __name__ == '__main__':
    main()



