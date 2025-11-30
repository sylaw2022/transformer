"""
Download and process TinyStories dataset for training.
TinyStories is a dataset of short stories designed for training small language models.
It contains simple stories with vocabulary understandable by 3-4 year olds.
"""

from datasets import load_dataset
import os
from tqdm import tqdm

def download_tinystories(output_file='data/tinystories_processed.txt', split='train'):
    """
    Download TinyStories dataset and save as processed text file.
    
    Args:
        output_file: Output file path
        split: Dataset split to download ('train', 'validation', or 'test')
    """
    print("Loading TinyStories dataset...")
    print("This may take a while on first run (dataset is ~2.2GB)...")
    
    try:
        # Load TinyStories dataset from Hugging Face
        # The dataset is available under roneneldan/TinyStories
        print(f"Loading TinyStories {split} split...")
        dataset = load_dataset('roneneldan/TinyStories', split=split)
        print(f"Loaded TinyStories dataset with {len(dataset)} examples")
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Process and write to file
        print(f"Processing and writing to {output_file}...")
        processed_lines = []
        total_chars = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Processing"):
                # Extract the story text
                # The dataset format may vary, try common field names
                text = None
                
                if isinstance(example, dict):
                    # Try different possible field names
                    text = example.get('text', None) or \
                           example.get('story', None) or \
                           example.get('content', None) or \
                           example.get('sentence', None)
                    
                    # If still None, try to get the first value that's a string
                    if text is None:
                        for key, value in example.items():
                            if isinstance(value, str) and len(value) > 50:
                                text = value
                                break
                elif isinstance(example, str):
                    text = example
                
                if text and len(text.strip()) > 20:
                    text = text.strip()
                    processed_lines.append(text)
                    total_chars += len(text)
                    f.write(text + '\n')
        
        # Get file size
        file_size_mb = os.path.getsize(output_file) / (1024**2)
        
        print(f"\n✓ Processed dataset saved to {output_file}")
        print(f"  Total samples: {len(processed_lines):,}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Average story length: {total_chars // len(processed_lines) if processed_lines else 0} characters")
        
        return output_file
        
    except Exception as e:
        print(f"✗ Error downloading TinyStories: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have the 'datasets' library installed:")
        print("   pip install datasets")
        print("\n2. If the dataset name is incorrect, try:")
        print("   - Check Hugging Face: https://huggingface.co/datasets/roneneldan/TinyStories")
        print("   - The dataset might be under a different name")
        print("\n3. Alternative: Download manually from:")
        print("   https://huggingface.co/datasets/roneneldan/TinyStories")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download TinyStories dataset')
    parser.add_argument('--output', type=str, default='data/tinystories_processed.txt',
                        help='Output file path')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to download (default: train)')
    
    args = parser.parse_args()
    
    result = download_tinystories(args.output, args.split)
    
    if result:
        print(f"\n✓ Dataset ready at: {result}")
        print(f"\nYou can now train the model with:")
        print(f"  python3 train.py --data_file {result} --output_dir ./checkpoints_tinystories")
        print(f"\nRecommended settings for TinyStories:")
        print(f"  python3 train.py \\")
        print(f"    --data_file {result} \\")
        print(f"    --output_dir ./checkpoints_tinystories \\")
        print(f"    --tokenizer_mode bpe \\")
        print(f"    --vocab_size 50000 \\")
        print(f"    --max_seq_len 1024 \\")
        print(f"    --d_model 768 \\")
        print(f"    --num_heads 12 \\")
        print(f"    --num_layers 12 \\")
        print(f"    --d_ff 3072 \\")
        print(f"    --epochs 50 \\")
        print(f"    --batch_size 4 \\")
        print(f"    --learning_rate 6e-5 \\")
        print(f"    --scheduler cosine_warmup \\")
        print(f"    --warmup_epochs 5 \\")
        print(f"    --device cuda")
    else:
        print("\n✗ Failed to download dataset")







