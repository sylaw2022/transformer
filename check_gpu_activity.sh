#!/bin/bash
# Script to check GPU activity level

echo "=== GPU Activity Check ==="
echo ""

# Method 1: Quick nvidia-smi check
if command -v nvidia-smi &> /dev/null; then
    echo "1. Current GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s (%s):\n", $1, $2; \
                 printf "  GPU Utilization: %s%%\n", $3; \
                 printf "  Memory Utilization: %s%%\n", $4; \
                 printf "  Memory Used: %s MB / %s MB\n", $5, $6; \
                 printf "  Temperature: %s°C\n", $7; \
                 printf "  Power Draw: %s W\n\n", $8}'
else
    echo "nvidia-smi not found. GPU may not be available or drivers not installed."
    exit 1
fi

# Method 2: Check processes using GPU
echo "2. Processes Using GPU:"
nvidia-smi pmon -c 1 2>/dev/null || nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

echo ""
echo "3. Continuous Monitoring (press Ctrl+C to stop):"
echo "   Run: watch -n 1 nvidia-smi"
echo "   Or:  nvidia-smi -l 1"
echo ""





