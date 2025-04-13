#!/usr/bin/env python3
"""
GPU memory bandwidth benchmark for the GPU benchmark suite.
This benchmark measures the real-world memory performance of the GPU.
"""

import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import platform
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import PyTorch and related libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Some benchmarks will be disabled.")
    TORCH_AVAILABLE = False

# Try to import OpenCL libraries
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    logger.warning("OpenCL not available. Some benchmarks will be disabled.")
    OPENCL_AVAILABLE = False

def get_gpu_info():
    """Get information about available GPUs."""
    gpu_info = {}
    
    if TORCH_AVAILABLE:
        # PyTorch GPU info
        gpu_info["torch"] = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        }
    
    if OPENCL_AVAILABLE:
        # OpenCL platform and device info
        gpu_info["opencl"] = {
            "platforms": []
        }
        
        try:
            for platform in cl.get_platforms():
                platform_info = {
                    "name": platform.name,
                    "vendor": platform.vendor,
                    "version": platform.version,
                    "devices": []
                }
                
                for device in platform.get_devices():
                    device_info = {
                        "name": device.name,
                        "type": cl.device_type.to_string(device.type),
                        "vendor": device.vendor,
                        "version": device.version,
                        "driver_version": device.driver_version,
                        "global_mem_size": device.global_mem_size
                    }
                    platform_info["devices"].append(device_info)
                
                gpu_info["opencl"]["platforms"].append(platform_info)
        except:
            logger.warning("Error getting OpenCL platform information")
    
    return gpu_info

def benchmark_memory_operations_pytorch(sizes_mb, operations=["read", "write", "copy"], iterations=5, device="cuda"):
    """
    Benchmark memory operations using PyTorch.
    
    Args:
        sizes_mb: List of sizes to test in megabytes
        operations: List of operations to test
        iterations: Number of iterations per test
        device: Device to use (cuda, mps, cpu)
        
    Returns:
        Dictionary with benchmark results
    """
    if not TORCH_AVAILABLE:
        return {"error": "PyTorch not available"}
    
    # Check if the requested device is available
    if device == "cuda" and not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    if device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        return {"error": "MPS not available"}
    
    device = torch.device(device)
    results = {}
    
    # Convert sizes from MB to bytes
    sizes_bytes = [size * 1024 * 1024 for size in sizes_mb]
    
    for size_mb, size_bytes in zip(sizes_mb, sizes_bytes):
        logger.info(f"Testing {size_mb} MB")
        result = {}
        
        # Allocate memory on device and host
        try:
            # For large arrays, we need to use float32 instead of float64 to save memory
            dtype = torch.float32
            elements = size_bytes // 4  # 4 bytes per float32
            
            # Allocate tensors
            data_host = torch.randn(elements, dtype=dtype)
            data_device = torch.empty(elements, dtype=dtype, device=device)
            data_device2 = torch.empty(elements, dtype=dtype, device=device)
            
            if "read" in operations:
                # Benchmark device to host transfer (read)
                data_device.copy_(data_host)  # First copy to have data on device
                torch.cuda.synchronize() if device.type == "cuda" else None
                
                start_time = time.time()
                for _ in range(iterations):
                    data_host.copy_(data_device)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.time()
                
                elapsed = end_time - start_time
                bandwidth = (size_bytes * iterations) / elapsed
                result["read_bandwidth"] = bandwidth / (1024 * 1024 * 1024)  # GB/s
                
            if "write" in operations:
                # Benchmark host to device transfer (write)
                torch.cuda.synchronize() if device.type == "cuda" else None
                
                start_time = time.time()
                for _ in range(iterations):
                    data_device.copy_(data_host)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.time()
                
                elapsed = end_time - start_time
                bandwidth = (size_bytes * iterations) / elapsed
                result["write_bandwidth"] = bandwidth / (1024 * 1024 * 1024)  # GB/s
                
            if "copy" in operations:
                # Benchmark device to device transfer (copy)
                data_device.copy_(data_host)  # First copy to have data on device
                torch.cuda.synchronize() if device.type == "cuda" else None
                
                start_time = time.time()
                for _ in range(iterations):
                    data_device2.copy_(data_device)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                end_time = time.time()
                
                elapsed = end_time - start_time
                bandwidth = (size_bytes * iterations) / elapsed
                result["copy_bandwidth"] = bandwidth / (1024 * 1024 * 1024)  # GB/s
            
            results[str(size_mb)] = result
            
        except RuntimeError as e:
            logger.warning(f"Error testing {size_mb} MB: {e}")
            # Try to free memory
            torch.cuda.empty_cache() if device.type == "cuda" else None
            results[str(size_mb)] = {"error": str(e)}
    
    # Find peak bandwidth for each operation
    peak = {}
    for operation in operations:
        op_key = f"{operation}_bandwidth"
        peak[op_key] = 0
        
        for size_result in results.values():
            if op_key in size_result and size_result[op_key] > peak[op_key]:
                peak[op_key] = size_result[op_key]
    
    # Add peak bandwidth to results
    results["peak"] = peak
    
    return results

def benchmark_memory_operations_opencl(sizes_mb, operations=["read", "write", "copy"], iterations=5, platform_index=0, device_index=0):
    """
    Benchmark memory operations using OpenCL.
    
    Args:
        sizes_mb: List of sizes to test in megabytes
        operations: List of operations to test
        iterations: Number of iterations per test
        platform_index: OpenCL platform index
        device_index: OpenCL device index
        
    Returns:
        Dictionary with benchmark results
    """
    if not OPENCL_AVAILABLE:
        return {"error": "OpenCL not available"}
    
    results = {}
    
    try:
        # Get platform and device
        platforms = cl.get_platforms()
        if platform_index >= len(platforms):
            return {"error": f"Platform index {platform_index} out of range (only {len(platforms)} platforms available)"}
            
        platform = platforms[platform_index]
        devices = platform.get_devices()
        
        if device_index >= len(devices):
            return {"error": f"Device index {device_index} out of range (only {len(devices)} devices available)"}
            
        device = devices[device_index]
        
        # Create context and queue
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        
        # Convert sizes from MB to bytes
        sizes_bytes = [size * 1024 * 1024 for size in sizes_mb]
        
        for size_mb, size_bytes in zip(sizes_mb, sizes_bytes):
            logger.info(f"Testing {size_mb} MB")
            result = {}
            
            try:
                # Each float32 is 4 bytes
                elements = size_bytes // 4
                
                # Allocate host buffers
                host_buffer = np.zeros(elements, dtype=np.float32)
                host_buffer_result = np.zeros(elements, dtype=np.float32)
                
                # Fill host buffer with random data
                np.random.seed(12345)  # For reproducibility
                host_buffer[:] = np.random.random(elements).astype(np.float32)
                
                # Allocate device buffers
                device_buffer1 = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=size_bytes)
                device_buffer2 = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=size_bytes)
                
                if "write" in operations:
                    # Benchmark host to device transfer (write)
                    start_time = time.time()
                    for _ in range(iterations):
                        event = cl.enqueue_copy(queue, device_buffer1, host_buffer)
                        event.wait()
                    end_time = time.time()
                    
                    elapsed = end_time - start_time
                    bandwidth = (size_bytes * iterations) / elapsed
                    result["write_bandwidth"] = bandwidth / (1024 * 1024 * 1024)  # GB/s
                    
                if "read" in operations:
                    # First write to device to have data
                    event = cl.enqueue_copy(queue, device_buffer1, host_buffer)
                    event.wait()
                    
                    # Benchmark device to host transfer (read)
                    start_time = time.time()
                    for _ in range(iterations):
                        event = cl.enqueue_copy(queue, host_buffer_result, device_buffer1)
                        event.wait()
                    end_time = time.time()
                    
                    elapsed = end_time - start_time
                    bandwidth = (size_bytes * iterations) / elapsed
                    result["read_bandwidth"] = bandwidth / (1024 * 1024 * 1024)  # GB/s
                    
                if "copy" in operations:
                    # First write to device to have data
                    event = cl.enqueue_copy(queue, device_buffer1, host_buffer)
                    event.wait()
                    
                    # Benchmark device to device transfer (copy)
                    start_time = time.time()
                    for _ in range(iterations):
                        event = cl.enqueue_copy(queue, device_buffer2, device_buffer1)
                        event.wait()
                    end_time = time.time()
                    
                    elapsed = end_time - start_time
                    bandwidth = (size_bytes * iterations) / elapsed
                    result["copy_bandwidth"] = bandwidth / (1024 * 1024 * 1024)  # GB/s
                
                results[str(size_mb)] = result
                
            except cl.RuntimeError as e:
                logger.warning(f"Error testing {size_mb} MB: {e}")
                results[str(size_mb)] = {"error": str(e)}
        
        # Find peak bandwidth for each operation
        peak = {}
        for operation in operations:
            op_key = f"{operation}_bandwidth"
            peak[op_key] = 0
            
            for size_result in results.values():
                if op_key in size_result and size_result[op_key] > peak[op_key]:
                    peak[op_key] = size_result[op_key]
        
        # Add peak bandwidth to results
        results["peak"] = peak
            
    except Exception as e:
        logger.error(f"OpenCL benchmark error: {e}")
        return {"error": str(e)}
    
    return results

def select_benchmark_backend(args):
    """Select the appropriate backend based on arguments and available libraries."""
    if args.backend == "auto":
        # Auto-detect: Prefer PyTorch with CUDA if available
        if TORCH_AVAILABLE:
            if args.device == "cuda" and torch.cuda.is_available():
                return "pytorch", "cuda"
            elif args.device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "pytorch", "mps"
            elif args.device == "cpu":
                return "pytorch", "cpu"
                
        # Fall back to OpenCL if available
        if OPENCL_AVAILABLE:
            return "opencl", None
    
    elif args.backend == "pytorch":
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch backend requested but not available")
            
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        if device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            raise ValueError("MPS device requested but not available")
            
        return "pytorch", device
    
    elif args.backend == "opencl":
        if not OPENCL_AVAILABLE:
            raise ValueError("OpenCL backend requested but not available")
        return "opencl", None
    
    raise ValueError(f"No suitable backend found for {args.backend} and {args.device}")

def main():
    """Main function to run the memory bandwidth benchmark."""
    parser = argparse.ArgumentParser(description="GPU Memory Bandwidth Benchmark")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--backend", choices=["auto", "pytorch", "opencl"], default="auto",
                      help="Backend to use for benchmarking (default: auto)")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                      help="Device to use for PyTorch backend (default: auto)")
    parser.add_argument("--platform-index", type=int, default=0,
                      help="OpenCL platform index (default: 0)")
    parser.add_argument("--device-index", type=int, default=0,
                      help="OpenCL device index (default: 0)")
    parser.add_argument("--sizes", type=str, default="16,64,256,512,1024",
                      help="Comma-separated list of sizes to test in MB (default: 16,64,256,512,1024)")
    parser.add_argument("--iterations", type=int, default=5,
                      help="Number of iterations per test (default: 5)")
    parser.add_argument("--operations", type=str, default="read,write,copy",
                      help="Comma-separated list of operations to test (default: read,write,copy)")
    args = parser.parse_args()
    
    # Parse sizes and operations
    sizes_mb = [int(size) for size in args.sizes.split(",")]
    operations = args.operations.split(",")
    
    # Initialize results
    results = {
        "device_info": get_gpu_info(),
        "benchmark_params": {
            "backend": args.backend,
            "device": args.device,
            "platform_index": args.platform_index,
            "device_index": args.device_index,
            "sizes_mb": sizes_mb,
            "iterations": args.iterations,
            "operations": operations
        }
    }
    
    # Select backend and run benchmark
    try:
        backend, device = select_benchmark_backend(args)
        results["backend_selected"] = backend
        
        if backend == "pytorch":
            results["device_selected"] = device
            benchmark_results = benchmark_memory_operations_pytorch(
                sizes_mb=sizes_mb,
                operations=operations,
                iterations=args.iterations,
                device=device
            )
        elif backend == "opencl":
            results["platform_index"] = args.platform_index
            results["device_index"] = args.device_index
            benchmark_results = benchmark_memory_operations_opencl(
                sizes_mb=sizes_mb,
                operations=operations,
                iterations=args.iterations,
                platform_index=args.platform_index,
                device_index=args.device_index
            )
        
        results["results"] = benchmark_results
        
        # Extract peak bandwidth for summary results
        if "peak" in benchmark_results:
            peak = benchmark_results["peak"]
            for operation in operations:
                op_key = f"{operation}_bandwidth"
                results[op_key] = peak.get(op_key, 0)
                results[f"{op_key}_unit"] = "GB/s"
            
            # Also set a combined MemoryBandwidth result (maximum of all)
            bandwidth_values = [peak.get(f"{op}_bandwidth", 0) for op in operations]
            if bandwidth_values:
                results["MemoryBandwidth"] = max(bandwidth_values)
                results["MemoryBandwidthUnit"] = "GB/s"
            
            # Memory classification
            if results.get("MemoryBandwidth", 0) > 1000:
                results["MemoryGeneration"] = "HBM3 or better"
            elif results.get("MemoryBandwidth", 0) > 750:
                results["MemoryGeneration"] = "HBM2e / GDDR6X"
            elif results.get("MemoryBandwidth", 0) > 500:
                results["MemoryGeneration"] = "HBM2 / GDDR6"
            elif results.get("MemoryBandwidth", 0) > 350:
                results["MemoryGeneration"] = "GDDR5X"
            elif results.get("MemoryBandwidth", 0) > 200:
                results["MemoryGeneration"] = "GDDR5"
            else:
                results["MemoryGeneration"] = "DDR or older GDDR"
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        results["error"] = str(e)
    
    # Output results
    if args.json:
        print(json.dumps(results))
    else:
        logger.info("\nGPU Memory Bandwidth Benchmark Results:")
        
        if "error" in results:
            logger.error(f"Error: {results['error']}")
        else:
            logger.info(f"Backend used: {results.get('backend_selected')}")
            
            if "device_selected" in results:
                logger.info(f"Device used: {results['device_selected']}")
            elif "platform_index" in results and "device_index" in results:
                platform_index = results["platform_index"]
                device_index = results["device_index"]
                
                platforms = results.get("device_info", {}).get("opencl", {}).get("platforms", [])
                if platforms and platform_index < len(platforms):
                    platform = platforms[platform_index]
                    devices = platform.get("devices", [])
                    
                    if devices and device_index < len(devices):
                        device = devices[device_index]
                        logger.info(f"OpenCL Platform: {platform.get('name')}")
                        logger.info(f"OpenCL Device: {device.get('name')}")
            
            # Print peak bandwidth for each operation
            peak = results.get("results", {}).get("peak", {})
            for operation in operations:
                op_key = f"{operation}_bandwidth"
                if op_key in peak:
                    logger.info(f"Peak {operation} bandwidth: {peak[op_key]:.2f} GB/s")
            
            if "MemoryBandwidth" in results:
                logger.info(f"Overall memory bandwidth: {results['MemoryBandwidth']:.2f} GB/s")
                logger.info(f"Memory classification: {results.get('MemoryGeneration', 'Unknown')}")
            
            # Print detailed results for each size
            logger.info("\nDetailed results by size:")
            for size_mb in sizes_mb:
                size_results = results.get("results", {}).get(str(size_mb), {})
                logger.info(f"\n{size_mb} MB:")
                
                for operation in operations:
                    op_key = f"{operation}_bandwidth"
                    if op_key in size_results:
                        logger.info(f"  {operation} bandwidth: {size_results[op_key]:.2f} GB/s")
    
    return 0 if "error" not in results else 1

if __name__ == "__main__":
    sys.exit(main()) 