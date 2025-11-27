# Transformer-based LLM with Autoregressive Generation

A PyTorch implementation of a GPT-style decoder-only transformer language model with autoregressive text generation capabilities.

## Features

- **Full Transformer Architecture**: Multi-head self-attention, feed-forward networks, residual connections, and layer normalization
- **Autoregressive Generation**: Supports temperature sampling, top-k sampling, and nucleus (top-p) sampling
- **Flexible Tokenization**: Character-level or word-level tokenization
- **Causal Masking**: Proper causal attention masks for autoregressive generation
- **Training Script**: Complete training pipeline with data loading, optimization, and checkpointing
- **Inference Script**: Easy-to-use text generation interface

## Architecture

The model consists of:
- **Token Embeddings**: Converts token IDs to dense vectors
- **Positional Encoding**: Adds positional information using sinusoidal encoding
- **Transformer Blocks**: Stack of decoder blocks with masked self-attention
- **Output Layer**: Projects hidden states to vocabulary logits

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Downloading Test Dataset

Download a test dataset (Tiny Shakespeare or Complete Shakespeare):
```bash
python3 download_dataset.py --dataset tiny_shakespeare
# or
python3 download_dataset.py --dataset shakespeare
```

This will download and process the dataset to `data/` directory.

### Training

1. Prepare your text data file (one text sample per line):
```bash
echo "Your text data here..." > data.txt
```

Or use the downloaded test dataset:
```bash
python3 train.py --data_file data/tiny_shakespeare_processed.txt --output_dir ./checkpoints
```

2. Train the model:
```bash
python train.py \
    --data_file data.txt \
    --output_dir ./checkpoints \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --tokenizer_mode char
```

### Text Generation

Generate text using a trained model:
```bash
python generate.py \
    --checkpoint ./checkpoints/checkpoint_epoch_10.pt \
    --tokenizer ./checkpoints/tokenizer.json \
    --prompt "The quick brown fox" \
    --max_length 100 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9
```

## Model Parameters

- `d_model`: Dimension of model embeddings (default: 512)
- `num_heads`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer blocks (default: 6)
- `d_ff`: Dimension of feed-forward network (default: 2048)
- `max_seq_len`: Maximum sequence length (default: 512)
- `dropout`: Dropout probability (default: 0.1)

## Generation Parameters

- `temperature`: Controls randomness (lower = more deterministic, higher = more random)
- `top_k`: Limits sampling to top-k most likely tokens
- `top_p`: Nucleus sampling - samples from tokens with cumulative probability ≤ top_p

## Example Usage

```python
from model import TransformerLLM
from tokenizer import SimpleTokenizer
import torch

# Load tokenizer
tokenizer = SimpleTokenizer.load('checkpoints/tokenizer.json')

# Initialize model
model = TransformerLLM(
    vocab_size=len(tokenizer),
    d_model=512,
    num_heads=8,
    num_layers=6
)

# Load weights
checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
generated = model.generate(
    tokenizer=tokenizer,
    prompt="Hello world",
    max_length=100,
    temperature=0.8
)
print(generated)
```

## File Structure

```
transformer/
├── model.py               # Transformer model implementation
├── tokenizer.py           # Tokenizer implementation
├── train.py               # Training script
├── generate.py            # Inference script
├── download_dataset.py    # Dataset download script
├── requirements.txt       # Python dependencies
├── data/                  # Downloaded datasets (created after running download_dataset.py)
└── README.md              # This file
```

## Notes

- The model uses causal (masked) self-attention to ensure autoregressive generation
- Positional encoding uses sinusoidal functions as in the original Transformer paper
- Training includes gradient clipping to stabilize training
- The model supports both CPU and GPU training (automatically detects CUDA)

## License

This implementation is provided as-is for educational and research purposes.

