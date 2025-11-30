# Text Generation Guide

## Quick Start

Generate text using the latest checkpoint:

```bash
./generate_from_latest.sh "Your prompt here"
```

Or use the default prompt:
```bash
./generate_from_latest.sh
```

## Latest Checkpoint

The script automatically uses the most recently saved checkpoint from `checkpoints_tinystories/`.

**Current latest checkpoint**: `checkpoint_epoch_1_20pct.pt`
- **Epoch**: 1
- **Progress**: 20% of epoch 1
- **Loss**: 2.43
- **Model Config**: 
  - vocab_size: 50000
  - d_model: 512
  - num_heads: 8
  - num_layers: 8
  - d_ff: 2048
  - max_seq_len: 512

## Usage Examples

### Basic Generation
```bash
./generate_from_latest.sh "Once upon a time"
```

### Generate Using Training Data Prompts
**Best way to test if the model learned from training data!**

```bash
# Generate with 1 random prompt from training data (default)
./generate_training_prompt.sh

# Generate with 3 different prompts
./generate_training_prompt.sh 3

# Custom: 2 prompts, 25 words each, 200 tokens max, temperature 0.7
./generate_training_prompt.sh 2 25 200 0.7

# Or use Python directly
python3 generate_from_training_data.py 1 20 150 0.7
```

This extracts random prompts from `data/tinystories_processed.txt` and uses them to generate text. This is useful for:
- Testing if the model learned patterns from training data
- Comparing generated text with original training samples
- Evaluating model performance on familiar contexts

### Custom Parameters
```bash
# Syntax: ./generate_from_latest.sh <prompt> <max_length> <temperature> <top_k> <top_p> <device>
./generate_from_latest.sh "The cat sat on" 300 0.7 40 0.95 cuda
```

### Using Python Directly
```bash
python3 generate.py \
    --checkpoint checkpoints_tinystories/checkpoint_epoch_1_20pct.pt \
    --tokenizer checkpoints_tinystories/tokenizer.json \
    --prompt "Your prompt here" \
    --max_length 200 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9 \
    --device cuda
```

## Parameters

- **`--prompt`**: Starting text for generation
- **`--max_length`**: Maximum number of tokens to generate (default: 200)
- **`--temperature`**: Controls randomness (lower = more deterministic, default: 0.8)
  - 0.1-0.5: Very focused, repetitive
  - 0.6-0.9: Balanced creativity
  - 1.0-1.5: More creative, less coherent
- **`--top_k`**: Consider only top K tokens (default: 50)
- **`--top_p`**: Nucleus sampling threshold (default: 0.9)
- **`--device`**: `cuda` or `cpu` (default: cuda if available)

## Available Checkpoints

Check available checkpoints:
```bash
ls -lth checkpoints_tinystories/*.pt
```

Current checkpoints:
- `checkpoint_epoch_1_20pct.pt` (Latest - Nov 29 04:47)
- `checkpoint_epoch_1_10pct.pt` (Nov 28 07:13)
- `checkpoint_1.pt` (Nov 27 09:40)

## Tips for Better Generation

1. **Early Training Checkpoints**: The model is still learning, so expect some repetition and less coherent text.

2. **Temperature Tuning**:
   - For more coherent text: `--temperature 0.6`
   - For more creative text: `--temperature 1.0`
   - For very focused text: `--temperature 0.4`

3. **Prompt Quality**: 
   - Use clear, simple prompts
   - Match the training data style (TinyStories uses simple children's stories)

4. **Length Control**:
   - Start with shorter generations (100-200 tokens)
   - Increase if results are good

## Troubleshooting

### CUDA Out of Memory
```bash
# Use CPU instead
./generate_from_latest.sh "Your prompt" 200 0.8 50 0.9 cpu
```

### Checkpoint Not Found
```bash
# Check available checkpoints
ls checkpoints_tinystories/*.pt
```

### Model Config Mismatch
Make sure the checkpoint matches the model architecture. Check the checkpoint metadata:
```bash
python3 -c "import torch; ckpt = torch.load('checkpoints_tinystories/checkpoint_epoch_1_20pct.pt', map_location='cpu'); print(ckpt['model_config'])"
```

## Example Output

```
Prompt: 'Once upon a time'

Generated text:
Once upon a time as a happy little girl was filled with a big smile. She had found her mum and dad smiled, but the little girl was very happy. She wanted to show her mum, so she hugged her mum, who smiled, said "Well" You can't have a cookie". Her mum hugged her, "You can have a cookie". So the little girl went to bed and went to sleep...
```

Note: Early training checkpoints may show repetition, especially at the end of generations. This improves as training progresses.

