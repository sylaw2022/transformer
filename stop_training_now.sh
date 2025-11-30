#!/bin/bash
# Script to stop training and save checkpoint
# Run with: sudo ./stop_training_now.sh

PID=55459
OUTPUT_DIR="./checkpoints_tinystories"
CHECKPOINT_NAME="checkpoint_1"

echo "Stopping training process (PID $PID)..."
echo "Sending SIGTERM signal..."

kill -TERM $PID 2>/dev/null || {
    echo "Error: Cannot send signal to process $PID"
    echo "Make sure you have permission to kill this process"
    exit 1
}

echo "Waiting for process to stop..."
sleep 5

# Check if process is still running
if kill -0 $PID 2>/dev/null; then
    echo "Process still running. Sending SIGKILL..."
    kill -KILL $PID 2>/dev/null
    sleep 2
fi

echo "Training stopped."
echo ""
echo "Checking for saved checkpoints..."

# Wait a bit for any final writes
sleep 2

# Check for checkpoints
if [ -d "$OUTPUT_DIR" ]; then
    CHECKPOINT_FILES=$(ls -t "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)
    
    if [ -n "$CHECKPOINT_FILES" ]; then
        LATEST=$(ls -t "$OUTPUT_DIR"/checkpoint_*.pt 2>/dev/null | head -1)
        TARGET="$OUTPUT_DIR/${CHECKPOINT_NAME}.pt"
        
        echo "Found checkpoint: $(basename $LATEST)"
        cp "$LATEST" "$TARGET"
        echo "✓ Saved as: $TARGET"
        echo "  Size: $(du -h "$TARGET" | cut -f1)"
    else
        echo "⚠ No checkpoint file found."
        echo ""
        echo "Note: The training process was running the old version of train.py"
        echo "which doesn't have signal handling to save on exit."
        echo ""
        echo "Current progress: ~29% of epoch 1, Loss: ~2.12"
        echo ""
        echo "To save checkpoints on interruption in the future:"
        echo "1. Restart training with the updated train.py (has signal handling)"
        echo "2. Then you can stop training anytime with Ctrl+C or SIGTERM"
        echo "3. It will automatically save as checkpoint_1.pt"
    fi
else
    echo "Output directory not found: $OUTPUT_DIR"
fi





