#!/bin/bash
# Quick fix: Kill stuck training process and clear GPU

echo "=== Fixing CUDA Busy Error ==="
echo ""

# Find and kill stuck training processes
echo "1. Finding stuck training processes..."
PIDS=$(pgrep -f "train.py.*tinystories" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "   No training processes found."
else
    echo "   Found processes: $PIDS"
    echo "2. Killing processes..."
    for PID in $PIDS; do
        echo "   Killing PID $PID..."
        sudo kill -9 $PID 2>/dev/null || kill -9 $PID 2>/dev/null
    done
    sleep 2
    echo "   Processes killed."
fi

echo ""
echo "3. Clearing GPU cache..."
python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize(); print('   GPU cache cleared')" 2>/dev/null || echo "   Could not clear GPU cache (may need to restart)"

echo ""
echo "4. Checking GPU status..."
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | \
    awk -F', ' '{printf "   GPU Utilization: %s%%\n   Memory Used: %s / %s\n", $1, $2, $3}'

echo ""
echo "=== Done ==="
echo "GPU should now be available. Try starting training again:"
echo "  ./start_training.sh"


