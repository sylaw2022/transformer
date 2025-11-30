#!/bin/bash
# Show clean generated text output

NUM=${1:-3}
python3 generate_from_training_data.py "$NUM" 20 150 0.7 2>&1 | \
    grep -A 200 "PROMPT (from training data):" | \
    grep -E "(PROMPT|Generated text:|^[A-Z]|^[a-z]|^\"|^'|^\.|^,|^!|^\?)" | \
    sed 's/Generated text://' | \
    sed 's/PROMPT (from training data):/---\nPROMPT:/' | \
    sed '/Loading tokenizer/d' | \
    sed '/Loaded BPE tokenizer/d' | \
    sed '/Model loaded/d' | \
    sed '/Vocabulary size/d' | \
    sed '/Generating text/d' | \
    sed '/^$/N;/^\n$/d'



