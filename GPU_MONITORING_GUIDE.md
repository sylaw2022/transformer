# GPU Activity Monitoring Guide

## Quick Commands

### 1. One-time GPU Status Check
```bash
nvidia-smi
```

### 2. Detailed GPU Metrics
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv
```

### 3. Continuous Monitoring (updates every 1 second)
```bash
watch -n 1 nvidia-smi
```

Or:
```bash
nvidia-smi -l 1
```

### 4. Lightweight Continuous Monitoring (just key metrics)
```bash
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
```

### 5. Check Processes Using GPU
```bash
nvidia-smi pmon -c 1
```

Or:
```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

## Using the Script

```bash
./check_gpu_activity.sh
```

## Understanding GPU Metrics

### Key Metrics:

1. **GPU Utilization (%)**: 
   - 0% = GPU is idle
   - 50-100% = GPU is actively computing
   - For training, expect 80-100%

2. **Memory Utilization (%)**:
   - Percentage of GPU memory being used
   - Should match your model size + batch data

3. **Memory Used/Total**:
   - Actual memory in MB/GB
   - Your model uses ~3388 MB currently

4. **Temperature**:
   - Normal: 30-80°C
   - High: >80°C (may throttle)

5. **Power Draw**:
   - Current power consumption
   - Higher = more active

## What to Look For

### Healthy Training:
- GPU Utilization: **80-100%**
- Memory: **Consistent usage** (not fluctuating wildly)
- Temperature: **Stable** (may increase over time)
- Power: **High and stable**

### Problem Signs:
- GPU Utilization: **0%** (training stopped)
- Memory: **Allocated but unused** (process stuck)
- Temperature: **Dropping** (GPU not working)
- Power: **Low** (idle)

## Current Status Check

Based on your situation:
- **GPU Utilization: 0%** ❌ (Should be 80-100%)
- **Memory: 3388 MB allocated** ⚠️ (Memory held but not used)
- **Conclusion: Training is NOT active**

## Advanced Monitoring

### Monitor with timestamps:
```bash
while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
    sleep 5
done
```

### Save to file:
```bash
nvidia-smi -l 1 > gpu_monitor.log
```

### Monitor specific process:
```bash
# Find PID of training process
ps aux | grep train.py

# Monitor that PID's GPU usage
nvidia-smi pmon -c -1 -s um
```

## Quick Health Check Script

```bash
#!/bin/bash
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if [ "$GPU_UTIL" -lt 10 ]; then
    echo "⚠️  WARNING: GPU utilization is only ${GPU_UTIL}%"
    echo "Training may be stuck or stopped!"
else
    echo "✓ GPU is active: ${GPU_UTIL}% utilization"
fi
```


