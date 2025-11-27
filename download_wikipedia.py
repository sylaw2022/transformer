"""
Download and process Wikipedia or alternative datasets for training.
Uses WikiText (processed Wikipedia subset) as primary option since 
the full Wikipedia dataset has compatibility issues.
"""

from datasets import load_dataset
import os
from tqdm import tqdm

def download_wikipedia(output_file='data/wikipedia_processed.txt', max_articles=None, use_wikitext=True):
    """
    Download Wikipedia/WikiText dataset and save as processed text file.
    
    Args:
        output_file: Output file path
        max_articles: Maximum number of articles to download (None for all)
        use_wikitext: If True, use WikiText dataset (recommended). If False, try full Wikipedia.
    """
    print("Loading dataset...")
    print("This may take a while on first run...")
    
    dataset = None
    streaming = False
    dataset_name = ""
    
    # Method 1: Try WikiText (recommended - it's a processed Wikipedia subset)
    if use_wikitext:
        try:
            print("Trying WikiText-103 (large Wikipedia subset)...")
            # WikiText-103 is a large, processed Wikipedia dataset
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=False)
            streaming = False
            dataset_name = "WikiText-103"
            print(f"Loaded {dataset_name} dataset (non-streaming)")
        except Exception as e1:
            print(f"WikiText-103 failed: {e1}")
            try:
                print("Trying WikiText-2 (smaller subset)...")
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=False)
                streaming = False
                dataset_name = "WikiText-2"
                print(f"Loaded {dataset_name} dataset (non-streaming)")
            except Exception as e2:
                print(f"WikiText-2 also failed: {e2}")
                use_wikitext = False  # Fall back to other methods
    
    # Method 2: Try full Wikipedia (may not work due to library compatibility)
    if not dataset and not use_wikitext:
        try:
            print("Trying full Wikipedia dataset...")
            # Try without date specification first
            dataset = load_dataset('wikipedia', split='train', streaming=True)
            streaming = True
            dataset_name = "Wikipedia"
            print(f"Loaded {dataset_name} dataset (streaming)")
        except Exception as e3:
            print(f"Wikipedia failed: {e3}")
            # Method 3: Try other alternatives
            try:
                print("Trying OpenWebText...")
                dataset = load_dataset('openwebtext', split='train', streaming=True)
                streaming = True
                dataset_name = "OpenWebText"
                print(f"Loaded {dataset_name} dataset (streaming)")
            except Exception as e4:
                print(f"OpenWebText failed: {e4}")
                print("\nAll dataset loading methods failed.")
                print("Please use existing data or manually download a dataset.")
                return
    
    if not dataset:
        print("Error: Could not load any dataset.")
        return
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    if streaming:
        print(f"Processing {dataset_name} articles (streaming mode)...")
        print(f"Writing to {output_file}...")
        article_count = 0
        total_chars = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Processing"):
                if max_articles and article_count >= max_articles:
                    break
                
                # Handle different dataset formats
                text = item.get('text', item.get('content', '')).strip()
                
                # Filter very short articles and keep only substantial content
                if len(text) > 200:  # At least 200 characters
                    # Split into paragraphs and write each as a line
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        para = para.strip()
                        # Filter out headers and very short paragraphs
                        if len(para) > 100 and not para.startswith('='):  # Skip section headers
                            f.write(para + '\n')
                            total_chars += len(para)
                    article_count += 1
    else:
        # For non-streaming, limit dataset size if requested
        if max_articles:
            dataset = dataset.select(range(min(max_articles, len(dataset))))
        
        print(f"Processing {len(dataset)} items from {dataset_name}...")
        print(f"Writing to {output_file}...")
        total_chars = 0
        article_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc="Processing"):
                # Handle different dataset formats
                text = item.get('text', item.get('content', '')).strip()
                
                # Filter very short articles and keep only substantial content
                if len(text) > 200:  # At least 200 characters
                    # Split into paragraphs and write each as a line
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        para = para.strip()
                        # Filter out headers and very short paragraphs
                        if len(para) > 100 and not para.startswith('='):  # Skip section headers
                            f.write(para + '\n')
                            total_chars += len(para)
                    article_count += 1
    
    # Get file size and stats
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        file_size_gb = file_size / 1024  # GB
        print(f"\n✅ Completed! Saved {file_size:.2f} MB ({file_size_gb:.3f} GB) to {output_file}")
        print(f"   Processed {article_count if streaming else len(dataset)} items from {dataset_name}")
    else:
        print("\n❌ Error: Output file was not created.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download Wikipedia/WikiText dataset')
    parser.add_argument('--output', type=str, default='data/wikipedia_processed.txt',
                        help='Output file path')
    parser.add_argument('--max_articles', type=int, default=None,
                        help='Maximum number of articles to download')
    parser.add_argument('--use_wikitext', action='store_true', default=True,
                        help='Use WikiText dataset (recommended, default: True)')
    parser.add_argument('--use_wikipedia', action='store_true', default=False,
                        help='Try full Wikipedia dataset instead of WikiText')
    args = parser.parse_args()
    
    # If user explicitly wants Wikipedia, set use_wikitext to False
    use_wikitext = not args.use_wikipedia
    
    download_wikipedia(args.output, args.max_articles, use_wikitext=use_wikitext)

