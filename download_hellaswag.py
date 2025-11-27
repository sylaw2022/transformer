"""
Download and process HellaSwag dataset for training.
HellaSwag is a commonsense reasoning dataset that can be used for training.
"""

from datasets import load_dataset
import os
from tqdm import tqdm

def download_hellaswag(output_file='data/hellaswag_processed.txt'):
    """
    Download HellaSwag dataset and save as processed text file.
    
    Args:
        output_file: Output file path
    """
    print("Loading HellaSwag dataset...")
    print("This may take a while on first run...")
    
    try:
        # Load HellaSwag dataset
        dataset = load_dataset('Rowan/hellaswag', split='train')
        print(f"Loaded HellaSwag dataset with {len(dataset)} examples")
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        # Process and write to file
        print(f"Processing and writing to {output_file}...")
        processed_lines = []
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc="Processing"):
                # Extract the context and activity label
                context = example.get('ctx', '')
                activity_label = example.get('activity_label', '')
                
                # Combine context and activity label for better training data
                if context and len(context.strip()) > 10:
                    text = context.strip()
                    if activity_label:
                        text = f"{activity_label}. {text}"
                    processed_lines.append(text)
                    f.write(text + '\n')
                
                # Also include the endings as separate training examples
                endings = example.get('endings', [])
                for ending in endings:
                    if ending and len(ending.strip()) > 10:
                        processed_lines.append(ending.strip())
                        f.write(ending.strip() + '\n')
        
        print(f"\n✓ Processed dataset saved to {output_file}")
        print(f"  Total samples: {len(processed_lines):,}")
        print(f"  Total characters: {sum(len(line) for line in processed_lines):,}")
        
        return output_file
        
    except Exception as e:
        print(f"✗ Error downloading HellaSwag: {e}")
        print("\nMake sure you have the 'datasets' library installed:")
        print("  pip install datasets")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download HellaSwag dataset')
    parser.add_argument('--output', type=str, default='data/hellaswag_processed.txt',
                        help='Output file path')
    
    args = parser.parse_args()
    
    result = download_hellaswag(args.output)
    
    if result:
        print(f"\n✓ Dataset ready at: {result}")
        print(f"\nYou can now train the model with:")
        print(f"  python3 train.py --data_file {result} --output_dir ./checkpoints")
    else:
        print("\n✗ Failed to download dataset")













