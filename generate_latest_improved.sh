#!/bin/bash
# Generate text using the LATEST checkpoint from the IMPROVED training directory

CHECKPOINT_DIR="checkpoints_improved"
PROMPT="${1:-Once upon a time}"
MAX_LENGTH="${2:-200}"
TEMPERATURE="${3:-0.7}"
TOP_K="${4:-50}"
TOP_P="${5:-0.9}"
DEVICE="cuda"

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

# Find tokenizer (usually in the same directory, or fallback to default)
TOKENIZER_PATH="$CHECKPOINT_DIR/tokenizer.json"
if [ ! -f "$TOKENIZER_PATH" ]; then
    # Fallback to older directory if not found (assuming shared tokenizer)
    TOKENIZER_PATH="checkpoints_tinystories/tokenizer.json"
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"
echo "Using tokenizer: $TOKENIZER_PATH"
echo "Prompt: $PROMPT"

python3 generate_improved.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --tokenizer "$TOKENIZER_PATH" \
    --prompt "$PROMPT" \
    --max_length "$MAX_LENGTH" \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --device "$DEVICE"

