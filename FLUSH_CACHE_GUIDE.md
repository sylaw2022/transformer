# How to Flush Various Caches

## 1. Python Output Buffer (for immediate log updates)

### Problem
Python buffers stdout/stderr by default, so log output may not appear immediately in the log file.

### Solutions

#### Option A: Run Python with unbuffered output
```bash
python3 -u train.py [arguments]
```

#### Option B: Set environment variable
```bash
export PYTHONUNBUFFERED=1
python3 train.py [arguments]
```

#### Option C: Add flush in code
In your Python script, add:
```python
import sys
sys.stdout.flush()
sys.stderr.flush()
```

Or after print statements:
```python
print("message", flush=True)
```

#### Option D: Use the script
```bash
./flush_cache.sh python
```

---

## 2. System Page Cache

### Flush Linux page cache
```bash
# Requires root/sudo
sudo sync
sudo sysctl vm.drop_caches=3
```

**What each level does:**
- `vm.drop_caches=1` - Free pagecache only
- `vm.drop_caches=2` - Free dentries and inodes
- `vm.drop_caches=3` - Free pagecache, dentries and inodes

**Or use the script:**
```bash
sudo ./flush_cache.sh system
```

**Warning:** This will free system memory used for caching. Only do this if you need to free memory immediately.

---

## 3. GPU/CUDA Cache

### Flush GPU memory cache
```python
import torch

# Empty CUDA cache
torch.cuda.empty_cache()

# Synchronize (wait for all operations to complete)
torch.cuda.synchronize()

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

**Or use the script:**
```bash
./flush_cache.sh gpu
```

**Note:** This only frees unused GPU memory. Memory in use by tensors won't be freed.

---

## 4. For Your Current Training Process

### To flush Python output buffer for running process:

Since the training is already running, you can't change its buffering mode. However:

1. **Check if it's already unbuffered:**
   ```bash
   ps aux | grep train.py | grep -E "PYTHONUNBUFFERED|-u"
   ```

2. **For future runs, start with:**
   ```bash
   PYTHONUNBUFFERED=1 python3 train.py [your arguments]
   ```

3. **To flush GPU cache (if needed):**
   ```bash
   python3 -c "import torch; torch.cuda.empty_cache(); print('GPU cache flushed')"
   ```

---

## Quick Reference

| Cache Type | Command | Requires Root? |
|------------|---------|----------------|
| Python buffer | `PYTHONUNBUFFERED=1 python3 script.py` | No |
| System cache | `sudo sysctl vm.drop_caches=3` | Yes |
| GPU cache | `python3 -c "import torch; torch.cuda.empty_cache()"` | No |

---

## Most Common Use Case: Immediate Log Updates

If you want to see log updates immediately without waiting for buffer flush:

**For new training runs:**
```bash
PYTHONUNBUFFERED=1 python3 train.py --your-args 2>&1 | tee training.log
```

**For current running process:**
- The process is already running, so you can't change its buffering
- Use `tail -f` to monitor the log file (it will show updates as they're written)
- The log file is being written to, just may be buffered


