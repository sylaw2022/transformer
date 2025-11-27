"""
Script to download test datasets for transformer model training.
Supports both small test datasets and larger training corpora.
"""

import os
import urllib.request
import argparse
from pathlib import Path
import time


def download_shakespeare(output_file='data/shakespeare.txt'):
    """Download Shakespeare's complete works from Project Gutenberg."""
    url = 'https://www.gutenberg.org/files/100/100-0.txt'
    
    print(f"Downloading Shakespeare's complete works from {url}...")
    print("This may take a few moments...")
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✓ Successfully downloaded to {output_file}")
        
        # Process the file to extract text samples
        print("Processing file...")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip Project Gutenberg header and footer
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if '*** START OF THE PROJECT GUTENBERG EBOOK' in line:
                start_idx = i + 1
            if '*** END OF THE PROJECT GUTENBERG EBOOK' in line:
                end_idx = i
                break
        
        # Extract text content
        content_lines = lines[start_idx:end_idx]
        
        # Process into sentences/chunks (one per line)
        processed_lines = []
        current_chunk = []
        
        for line in content_lines:
            line = line.strip()
            if not line:
                if current_chunk:
                    processed_lines.append(' '.join(current_chunk))
                    current_chunk = []
                continue
            
            # Skip very short lines that are likely headers
            if len(line) < 10:
                continue
            
            # Add to current chunk
            current_chunk.append(line)
            
            # If chunk is getting long, save it
            if len(' '.join(current_chunk)) > 200:
                processed_lines.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining chunk
        if current_chunk:
            processed_lines.append(' '.join(current_chunk))
        
        # Write processed data
        processed_file = output_file.replace('.txt', '_processed.txt')
        with open(processed_file, 'w', encoding='utf-8') as f:
            for line in processed_lines:
                if len(line.strip()) > 20:  # Filter very short lines
                    f.write(line.strip() + '\n')
        
        print(f"✓ Processed file saved to {processed_file}")
        print(f"  Total samples: {len(processed_lines)}")
        print(f"  Total characters: {sum(len(line) for line in processed_lines):,}")
        
        return processed_file
        
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        print("\nTrying alternative source...")
        return download_shakespeare_alternative(output_file)


def download_shakespeare_alternative(output_file='data/shakespeare.txt'):
    """Fallback: Create a smaller sample dataset."""
    print("Creating sample Shakespeare dataset...")
    
    sample_text = """To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them.
All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts.
Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name.
What's in a name? That which we call a rose
By any other name would smell as sweet.
The lady doth protest too much, methinks.
Double, double toil and trouble;
Fire burn and caldron bubble.
If music be the food of love, play on.
Now is the winter of our discontent
Made glorious summer by this son of York.
How sharper than a serpent's tooth it is
To have a thankless child.
Cowards die many times before their deaths;
The valiant never taste of death but once.
We are such stuff as dreams are made on,
And our little life is rounded with a sleep.
What light through yonder window breaks?
It is the east, and Juliet is the sun.
A rose by any other name would smell as sweet.
Out, out, brief candle!
Life's but a walking shadow, a poor player
That struts and frets his hour upon the stage
And then is heard no more.
Fair is foul, and foul is fair.
Hover through the fog and filthy air.
Something is rotten in the state of Denmark.
The course of true love never did run smooth.
I am one who loved not wisely but too well.
Laughing matters most.
The better part of valor is discretion.
Men at some time are masters of their fates.
The fault, dear Brutus, is not in our stars,
But in ourselves, that we are underlings."""
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    print(f"✓ Sample dataset created at {output_file}")
    return output_file


def download_tiny_shakespeare(output_file='data/tiny_shakespeare.txt'):
    """Download tiny_shakespeare dataset (popular for testing)."""
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    print(f"Downloading Tiny Shakespeare dataset from {url}...")
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✓ Successfully downloaded to {output_file}")
        
        # Process into lines
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into sentences/lines
        processed_file = output_file.replace('.txt', '_processed.txt')
        with open(processed_file, 'w', encoding='utf-8') as f:
            # Split by newlines and periods
            sentences = content.replace('\n', ' ').split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    f.write(sentence + '.\n')
        
        print(f"✓ Processed file saved to {processed_file}")
        return processed_file
        
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return None


def download_gutenberg_collection(output_file='data/gutenberg_collection.txt', num_books=10):
    """
    Download multiple classic books from Project Gutenberg.
    Creates a larger corpus by combining multiple books.
    """
    # Popular Project Gutenberg books (public domain)
    gutenberg_books = [
        ('100', 'Shakespeare Complete Works'),
        ('11', 'Alice in Wonderland'),
        ('84', 'Frankenstein'),
        ('1342', 'Pride and Prejudice'),
        ('74', 'Treasure Island'),
        ('98', 'A Tale of Two Cities'),
        ('2701', 'Moby Dick'),
        ('5200', 'Metamorphosis'),
        ('1661', 'Adventures of Sherlock Holmes'),
        ('76', 'Adventures of Huckleberry Finn'),
        ('158', 'Emma'),
        ('345', 'Dracula'),
        ('514', 'Little Women'),
        ('1232', 'The Prince'),
        ('43', 'Dr. Jekyll and Mr. Hyde'),
    ]
    
    print(f"Downloading {num_books} books from Project Gutenberg...")
    print("This will create a larger training corpus.\n")
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    all_texts = []
    base_url = 'https://www.gutenberg.org/files/{}/{}-0.txt'
    
    for i, (book_id, title) in enumerate(gutenberg_books[:num_books]):
        url = base_url.format(book_id, book_id)
        print(f"[{i+1}/{num_books}] Downloading {title}...")
        
        try:
            # Create temporary file
            temp_file = f'/tmp/gutenberg_{book_id}.txt'
            urllib.request.urlretrieve(url, temp_file)
            
            # Extract text content
            with open(temp_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Find content boundaries
            start_idx = 0
            end_idx = len(lines)
            
            for j, line in enumerate(lines):
                if '*** START OF THE PROJECT GUTENBERG EBOOK' in line:
                    start_idx = j + 1
                if '*** END OF THE PROJECT GUTENBERG EBOOK' in line:
                    end_idx = j
                    break
            
            # Extract and clean text
            content = '\n'.join(lines[start_idx:end_idx])
            # Remove excessive whitespace
            content = '\n'.join([line.strip() for line in content.split('\n') if line.strip()])
            all_texts.append(content)
            
            # Clean up temp file
            os.remove(temp_file)
            
            print(f"  ✓ Downloaded {len(content):,} characters")
            time.sleep(1)  # Be respectful to Project Gutenberg servers
            
        except Exception as e:
            print(f"  ✗ Error downloading {title}: {e}")
            continue
    
    if not all_texts:
        print("\n✗ Failed to download any books")
        return None
    
    # Combine all texts
    print(f"\nCombining {len(all_texts)} books...")
    combined_text = '\n\n'.join(all_texts)
    
    # Process into lines (one sentence/chunk per line)
    processed_lines = []
    current_chunk = []
    
    for line in combined_text.split('\n'):
        line = line.strip()
        if not line:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 20:
                    processed_lines.append(chunk_text)
                current_chunk = []
            continue
        
        # Skip headers and very short lines
        if len(line) < 10 or line.isupper():
            continue
        
        current_chunk.append(line)
        
        # Save chunk if it's long enough
        if len(' '.join(current_chunk)) > 200:
            processed_lines.append(' '.join(current_chunk))
            current_chunk = []
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) > 20:
            processed_lines.append(chunk_text)
    
    # Write processed file
    processed_file = output_file.replace('.txt', '_processed.txt')
    with open(processed_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line.strip() + '\n')
    
    total_chars = sum(len(line) for line in processed_lines)
    print(f"✓ Processed corpus saved to {processed_file}")
    print(f"  Total samples: {len(processed_lines):,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Estimated size: ~{total_chars / 1_000_000:.1f} MB")
    
    return processed_file


def download_wikipedia_sample(output_file='data/wikipedia_sample.txt', num_articles=1000):
    """
    Download a sample of Wikipedia articles.
    Note: This uses a simplified approach. For production, use official Wikipedia dumps.
    """
    print("Wikipedia download requires official dumps.")
    print("For now, recommending Project Gutenberg collection instead.")
    print("Use --dataset gutenberg_collection for a larger corpus.")
    return None


def main():
    parser = argparse.ArgumentParser(description='Download datasets for transformer training')
    parser.add_argument('--dataset', type=str, default='shakespeare',
                        choices=['shakespeare', 'tiny_shakespeare', 'gutenberg_collection'],
                        help='Dataset to download')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: data/<dataset>.txt)')
    parser.add_argument('--num_books', type=int, default=10,
                        help='Number of books to download (for gutenberg_collection)')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'data/{args.dataset}.txt'
    
    print(f"Downloading {args.dataset} dataset...")
    print(f"Output file: {args.output}\n")
    
    if args.dataset == 'shakespeare':
        result = download_shakespeare(args.output)
    elif args.dataset == 'tiny_shakespeare':
        result = download_tiny_shakespeare(args.output)
    elif args.dataset == 'gutenberg_collection':
        result = download_gutenberg_collection(args.output, args.num_books)
    
    if result:
        print(f"\n✓ Dataset ready at: {result}")
        print(f"\nYou can now train the model with:")
        print(f"  python3 train.py --data_file {result} --output_dir ./checkpoints")
        print(f"\nRecommended settings for larger corpus:")
        print(f"  python3 train.py \\")
        print(f"    --data_file {result} \\")
        print(f"    --output_dir ./checkpoints_word \\")
        print(f"    --tokenizer_mode word \\")
        print(f"    --vocab_size 10000 \\")
        print(f"    --max_seq_len 1024 \\")
        print(f"    --d_model 768 \\")
        print(f"    --num_layers 6 \\")
        print(f"    --epochs 50")
    else:
        print("\n✗ Failed to download dataset")


if __name__ == '__main__':
    main()


