#!/bin/bash
# Script to diagnose and fix CUDA "device busy" errors

echo "=== CUDA Error Diagnosis ==="
echo ""

# Check 1: Find processes using GPU
echo "1. Checking processes using GPU..."
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "No processes found or nvidia-smi unavailable"

echo ""
echo "2. Checking for Python processes..."
ps aux | grep -E "python.*train|python.*cuda" | grep -v grep

echo ""
echo "3. GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "=== Solutions ==="
echo ""
echo "Option 1: Kill processes using GPU"
echo "  Run: sudo kill -9 <PID>"
echo ""
echo "Option 2: Reset GPU (if processes are stuck)"
echo "  Run: sudo nvidia-smi --gpu-reset -i 0"
echo "  (Warning: This will reset the GPU and may affect other processes)"
echo ""
echo "Option 3: Clear GPU cache in Python"
echo "  Run: python3 -c 'import torch; torch.cuda.empty_cache(); print(\"GPU cache cleared\")'"
echo ""
echo "Option 4: Restart the system (if nothing else works)"
echo "  Run: sudo reboot"





