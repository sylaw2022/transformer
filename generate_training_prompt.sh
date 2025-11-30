#!/bin/bash
# Quick script to generate text using a random prompt from training data

# Default values
NUM_PROMPTS=${1:-1}
PROMPT_LENGTH=${2:-20}
MAX_LENGTH=${3:-150}
TEMPERATURE=${4:-0.7}

echo "Generating text using prompt(s) from training data..."
echo "Number of prompts: $NUM_PROMPTS"
echo "Prompt length: $PROMPT_LENGTH words"
echo "Max generation length: $MAX_LENGTH"
echo "Temperature: $TEMPERATURE"
echo ""

python3 generate_from_training_data.py "$NUM_PROMPTS" "$PROMPT_LENGTH" "$MAX_LENGTH" "$TEMPERATURE"



