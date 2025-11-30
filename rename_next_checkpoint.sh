#!/bin/bash
# Script to rename the next checkpoint to checkpoint_1.pt
# This monitors the checkpoints directory and renames the next checkpoint file

OUTPUT_DIR="./checkpoints_tinystories"
CHECKPOINT_NAME="checkpoint_1"

echo "Monitoring $OUTPUT_DIR for new checkpoints..."
echo "Will rename the next checkpoint to ${CHECKPOINT_NAME}.pt"

# Wait for a new checkpoint file
while true; do
    # Check for new checkpoint files
    for file in "$OUTPUT_DIR"/checkpoint_epoch_*.pt; do
        if [ -f "$file" ]; then
            # Get the most recent checkpoint
            latest=$(ls -t "$OUTPUT_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                target="$OUTPUT_DIR/${CHECKPOINT_NAME}.pt"
                if [ ! -f "$target" ] || [ "$latest" -nt "$target" ]; then
                    echo "Found checkpoint: $latest"
                    echo "Copying to: $target"
                    cp "$latest" "$target"
                    echo "Checkpoint saved as: $target"
                    exit 0
                fi
            fi
        fi
    done
    sleep 5  # Check every 5 seconds
done





