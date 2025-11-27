#!/usr/bin/env python3
"""
Script to clear GPU cache and display memory information.
"""

import torch
import gc


def clear_gpu_cache():
    """Clear GPU cache and perform garbage collection."""
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU cache to clear.")
        return
    
    print("Clearing GPU cache...")
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    
    print("GPU cache cleared successfully!")
    
    # Display memory information
    print("\nGPU Memory Status:")
    print("-" * 50)
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    print(f"Total GPU Memory:     {total:.2f} GB")
    print(f"Allocated Memory:      {allocated:.2f} GB")
    print(f"Reserved Memory:       {reserved:.2f} GB")
    print(f"Free Memory:           {total - reserved:.2f} GB")
    print(f"Utilization:           {(reserved / total * 100):.1f}%")


if __name__ == "__main__":
    clear_gpu_cache()









