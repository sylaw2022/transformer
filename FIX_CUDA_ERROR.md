# How to Fix CUDA "Device Busy or Unavailable" Error

## Common Causes

1. **Another process is using the GPU** (most common)
2. **Previous training process didn't release GPU resources**
3. **GPU is in a bad state** (needs reset)
4. **GPU memory is fragmented** (needs cache clear)

## Quick Diagnosis

Run the diagnostic script:
```bash
./fix_cuda_error.sh
```

Or manually check:
```bash
# Check what's using the GPU
nvidia-smi

# Check for Python processes
ps aux | grep python | grep -v grep

# Check GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
```

## Solutions (in order of preference)

### Solution 1: Kill Processes Using GPU

**Find the process:**
```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
```

**Kill the process:**
```bash
# Replace <PID> with the actual process ID
sudo kill -9 <PID>
```

**Or kill all Python training processes:**
```bash
pkill -9 -f "train.py"
```

### Solution 2: Clear GPU Cache

**In Python:**
```python
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

**Or from command line:**
```bash
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
```

### Solution 3: Reset GPU (Use with caution)

**Reset the GPU:**
```bash
sudo nvidia-smi --gpu-reset -i 0
```

**Warning:** This will reset the GPU and may affect other processes. Only use if other solutions don't work.

### Solution 4: Restart System (Last resort)

If nothing else works:
```bash
sudo reboot
```

## Prevention

### 1. Always Clean Up After Training

Add this to your training script or run after training:
```python
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
```

### 2. Use Context Managers

Wrap GPU operations in try-finally:
```python
try:
    # Training code
    pass
finally:
    torch.cuda.empty_cache()
```

### 3. Check Before Starting

Before starting training, check if GPU is available:
```python
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
    
# Check if GPU is busy
if torch.cuda.is_initialized():
    torch.cuda.empty_cache()
```

### 4. Set CUDA Device

Explicitly set which GPU to use:
```python
import torch
torch.cuda.set_device(0)  # Use GPU 0
```

## For Your Current Situation

Based on your previous training run, you likely have a stuck process:

1. **Check for stuck training process:**
   ```bash
   ps aux | grep "train.py.*tinystories" | grep -v grep
   ```

2. **If found, kill it:**
   ```bash
   sudo kill -9 55459  # Replace with actual PID
   ```

3. **Clear GPU cache:**
   ```bash
   python3 -c "import torch; torch.cuda.empty_cache(); print('Cleared')"
   ```

4. **Verify GPU is free:**
   ```bash
   nvidia-smi
   ```

5. **Then start training again:**
   ```bash
   ./start_training.sh
   ```

## Quick Fix Script

Create a script to automatically fix common issues:

```bash
#!/bin/bash
# Quick fix for CUDA busy error

echo "Killing stuck Python processes..."
pkill -9 -f "train.py" 2>/dev/null
sleep 2

echo "Clearing GPU cache..."
python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null

echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo "Done! Try starting training again."
```

## Additional Debugging

If the error persists:

1. **Check CUDA version compatibility:**
   ```bash
   python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

2. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

3. **Check CUDA runtime:**
   ```bash
   nvcc --version
   ```

4. **Check for multiple CUDA installations:**
   ```bash
   which nvcc
   ldconfig -p | grep cuda
   ```

## Common Error Messages

- `CUDA error: CUDA-capable device(s) is/are busy or unavailable`
  - **Solution:** Kill processes using GPU, clear cache

- `CUDA out of memory`
  - **Solution:** Reduce batch size, clear cache

- `CUDA driver version is insufficient`
  - **Solution:** Update NVIDIA drivers

- `CUDA runtime error: initialization error`
  - **Solution:** Restart system, check driver installation


