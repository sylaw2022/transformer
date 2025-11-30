#!/bin/bash
# Script to flush various caches

echo "=== Cache Flushing Options ==="
echo ""
echo "1. Python output buffer (for immediate log updates)"
echo "2. System page cache"
echo "3. GPU/CUDA cache"
echo "4. All of the above"
echo ""

# Function to flush Python output buffer
flush_python_buffer() {
    echo "Flushing Python output buffer..."
    # This would need to be done from within the Python process
    # or by restarting with unbuffered output
    echo "Note: Python buffer flushing requires:"
    echo "  - Running Python with -u flag (unbuffered)"
    echo "  - Or using sys.stdout.flush() in the code"
    echo "  - Or setting PYTHONUNBUFFERED=1 environment variable"
}

# Function to flush system cache
flush_system_cache() {
    echo "Flushing system page cache..."
    if [ "$EUID" -ne 0 ]; then
        echo "Warning: Requires root privileges. Run with sudo."
        echo "Command: sudo sync && sudo sysctl vm.drop_caches=3"
    else
        sync
        sysctl vm.drop_caches=3
        echo "System cache flushed."
    fi
}

# Function to flush GPU cache
flush_gpu_cache() {
    echo "Flushing GPU/CUDA cache..."
    python3 << EOF
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("GPU cache flushed.")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
else:
    print("CUDA not available.")
EOF
}

# Main menu
case "${1:-}" in
    1|python|buffer)
        flush_python_buffer
        ;;
    2|system|page)
        flush_system_cache
        ;;
    3|gpu|cuda)
        flush_gpu_cache
        ;;
    4|all)
        flush_python_buffer
        echo ""
        flush_system_cache
        echo ""
        flush_gpu_cache
        ;;
    *)
        echo "Usage: $0 [1|python|buffer|2|system|page|3|gpu|cuda|4|all]"
        echo ""
        echo "Examples:"
        echo "  $0 python    # Show Python buffer flushing info"
        echo "  $0 system    # Flush system cache (needs sudo)"
        echo "  $0 gpu       # Flush GPU cache"
        echo "  $0 all       # Flush all caches"
        ;;
esac





