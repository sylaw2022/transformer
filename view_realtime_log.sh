#!/bin/bash
# Script to view real-time training log

LOG_FILE="/home/sylaw/transformer/training_tinystories.log"

echo "Viewing real-time log: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""
echo "Showing last 50 lines, then following new lines..."
echo "=========================================="
echo ""

tail -n 50 -f "$LOG_FILE"


