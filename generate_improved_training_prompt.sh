#!/bin/bash
# Generate text using random prompts from the training data (Improved Model)

NUM_PROMPTS=${1:-3}
PROMPT_LENGTH=${2:-20}
MAX_LENGTH=${3:-200}
TEMPERATURE=${4:-0.8}
DEVICE="cuda"

echo "Generating $NUM_PROMPTS samples with prompt length $PROMPT_LENGTH..."

python3 generate_improved_from_training_data.py \
    "$NUM_PROMPTS" \
    "$PROMPT_LENGTH" \
    "$MAX_LENGTH" \
    "$TEMPERATURE" \
    "$DEVICE"

