# Recommended Datasets for Training Large Language Models

## High-Quality Datasets

### 1. **The Pile** (Recommended)
- **Size**: ~825 GB uncompressed
- **Source**: EleutherAI
- **Content**: Diverse text from books, web, code, academic papers
- **Download**: 
  ```bash
  # Use HuggingFace datasets
  pip install datasets
  python -c "from datasets import load_dataset; ds = load_dataset('EleutherAI/pile', split='train'); ds.save_to_disk('data/pile')"
  ```
- **Pros**: Very diverse, high quality, well-curated
- **Cons**: Large download size

### 2. **C4 (Colossal Clean Crawled Corpus)**
- **Size**: ~750 GB
- **Source**: Google
- **Content**: Cleaned web text
- **Download**:
  ```bash
  pip install datasets
  python -c "from datasets import load_dataset; ds = load_dataset('c4', 'en', split='train', streaming=True); ..."
  ```
- **Pros**: Clean, large, web-scale
- **Cons**: Web text quality varies

### 3. **BookCorpus + OpenWebText**
- **Size**: ~38 GB combined
- **Source**: Various
- **Content**: Books and Reddit links
- **Download**:
  ```bash
  # OpenWebText
  git clone https://github.com/eukaryote31/openwebtext.git
  cd openwebtext && python download.py
  ```
- **Pros**: Good quality, manageable size
- **Cons**: May have licensing issues

### 4. **Wikipedia + BookCorpus** (Good Starting Point)
- **Size**: ~20-30 GB
- **Source**: Wikipedia + BookCorpus
- **Content**: Wikipedia articles and books
- **Download**:
  ```bash
  pip install datasets
  python -c "from datasets import load_dataset; wiki = load_dataset('wikipedia', '20220301.en', split='train'); wiki.save_to_disk('data/wikipedia')"
  ```
- **Pros**: High quality, free, well-structured
- **Cons**: Smaller than web-scale datasets

### 5. **Project Gutenberg** (Current - Can Expand)
- **Size**: ~3-10 GB (can download more)
- **Source**: Project Gutenberg
- **Content**: Public domain books
- **Download More**:
  ```bash
  # Use the existing download_dataset.py or expand it
  # Can download all ~70,000 books
  ```
- **Pros**: High quality literature, free
- **Cons**: Limited to public domain, older texts

## Recommended Approach

### For Your Current Setup:
1. **Start with Wikipedia + BookCorpus** (~20-30 GB)
   - High quality
   - Manageable size
   - Good for initial experiments

2. **Then add C4 or The Pile** (if you have storage)
   - More diverse
   - Better generalization

### Quick Start Script

Create a script to download Wikipedia:

```python
# download_wikipedia.py
from datasets import load_dataset
import os

print("Downloading Wikipedia dataset...")
dataset = load_dataset('wikipedia', '20220301.en', split='train')

output_file = 'data/wikipedia_processed.txt'
os.makedirs('data', exist_ok=True)

print(f"Writing {len(dataset)} articles to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    for article in dataset:
        text = article['text'].strip()
        if len(text) > 100:  # Filter very short articles
            f.write(text + '\n')

print(f"Saved to {output_file}")
```

## Dataset Size Recommendations

- **Small (for testing)**: 1-5 GB
- **Medium (good quality)**: 20-50 GB  
- **Large (production)**: 100+ GB
- **Very Large (research)**: 500+ GB

For a 100M+ parameter model, aim for at least 20-50 GB of high-quality text.


