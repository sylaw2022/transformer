#!/bin/bash
# Generate text using the latest saved checkpoint

CHECKPOINT_DIR="checkpoints_tinystories"
TOKENIZER="$CHECKPOINT_DIR/tokenizer.json"

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"
echo ""

# Default prompt if not provided
PROMPT="${1:-Once upon a time}"

# Default parameters
MAX_LENGTH="${2:-200}"
TEMPERATURE="${3:-0.8}"
TOP_K="${4:-50}"
TOP_P="${5:-0.9}"
DEVICE="${6:-cuda}"

echo "Prompt: '$PROMPT'"
echo "Max length: $MAX_LENGTH"
echo "Temperature: $TEMPERATURE"
echo "Top-k: $TOP_K"
echo "Top-p: $TOP_P"
echo "Device: $DEVICE"
echo ""
echo "Generating text..."
echo ""

python3 generate.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --max_length "$MAX_LENGTH" \
    --temperature "$TEMPERATURE" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --device "$DEVICE"



