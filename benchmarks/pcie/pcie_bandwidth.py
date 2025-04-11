#!/usr/bin/env python3
"""
GPU PCIe Bandwidth Benchmark

This script measures the PCIe bandwidth between CPU and GPU by timing
data transfers in both directions. Results are reported in GB/s.
"""

import os
import sys
import json
import time
import argparse
import logging
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PCIe-Bandwidth")

# Add parent directory to path to import common utilities
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Try to import required libraries
try:
    import numpy as np
except ImportError:
    logger.error("Error importing NumPy. Please install it with 'pip install numpy'")
    sys.exit(1)

# Try to import backend libraries - we'll try different ones
BACKEND = None
BACKEND_NAME = None

# Try PyTorch first
try:
    import torch
    if torch.cuda.is_available():
        BACKEND = "pytorch"
        BACKEND_NAME = "PyTorch CUDA"
        logger.info("Using PyTorch CUDA backend")
except ImportError:
    pass

# Try CUDA directly if PyTorch isn't available
if BACKEND is None:
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        BACKEND = "pycuda"
        BACKEND_NAME = "PyCUDA"
        logger.info("Using PyCUDA backend")
    except ImportError:
        pass

# Try OpenCL if neither PyTorch nor CUDA is available
if BACKEND is None:
    try:
        import pyopencl as cl
        BACKEND = "opencl"
        BACKEND_NAME = "OpenCL"
        logger.info("Using OpenCL backend")
    except ImportError:
        pass

# If none of the GPU backends are available, we can't run this benchmark
if BACKEND is None:
    logger.error("No GPU backend available. Please install PyTorch, PyCUDA, or PyOpenCL.")
    logger.error("- PyTorch: pip install torch")
    logger.error("- PyCUDA: pip install pycuda")
    logger.error("- PyOpenCL: pip install pyopencl")
    sys.exit(1)

def get_device_info():
    """Get information about the GPU device using the selected backend"""
    device_info = {
        "backend": BACKEND_NAME,
        "device": "Unknown",
        "pcie_version": "Unknown",
        "platform": platform.system()
    }
    
    if BACKEND == "pytorch":
        device_info["device"] = torch.cuda.get_device_name(0)
        device_info["cuda_version"] = torch.version.cuda
        
    elif BACKEND == "pycuda":
        device = cuda.Device(0)
        device_info["device"] = device.name()
        
    elif BACKEND == "opencl":
        platforms = cl.get_platforms()
        if platforms:
            # Find first GPU device
            for cl_platform in platforms:
                devices = cl_platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    device = devices[0]
                    device_info["device"] = device.name
                    device_info["platform_name"] = cl_platform.name
                    break
    
    # Try to determine PCIe version from device name or OS-specific methods
    # This is not always reliable, but better than nothing
    device_name = device_info["device"].lower()
    
    # Try to guess based on GPU generation
    if "rtx 30" in device_name or "rtx 40" in device_name or "rx 6" in device_name:
        device_info["pcie_version"] = "PCIe 4.0"
    elif "rtx 20" in device_name or "gtx 16" in device_name or "vega" in device_name:
        device_info["pcie_version"] = "PCIe 3.0"
    else:
        # Default to PCIe 3.0 if we can't detect
        device_info["pcie_version"] = "PCIe 3.0 (estimated)"
    
    return device_info

def measure_bandwidth_pytorch(size_mb, iterations=10, pinned=True):
    """Measure PCIe bandwidth using PyTorch CUDA backend"""
    # Convert MB to bytes
    size_bytes = size_mb * 1024 * 1024
    
    # Create CPU tensor
    if pinned:
        cpu_data = torch.empty(size_bytes // 4, dtype=torch.float32, pin_memory=True)
    else:
        cpu_data = torch.empty(size_bytes // 4, dtype=torch.float32)
    
    # Create GPU tensor
    gpu_data = torch.empty(size_bytes // 4, dtype=torch.float32, device="cuda")
    
    # Fill CPU tensor with random data
    cpu_data.random_()
    
    # Warmup
    for _ in range(2):
        gpu_data.copy_(cpu_data)
        torch.cuda.synchronize()
        cpu_data.copy_(gpu_data)
        torch.cuda.synchronize()
    
    # Measure host to device bandwidth (CPU -> GPU)
    h2d_times = []
    for _ in range(iterations):
        start = time.time()
        gpu_data.copy_(cpu_data)
        torch.cuda.synchronize()
        end = time.time()
        h2d_times.append(end - start)
    
    h2d_bandwidth = size_bytes / (min(h2d_times) * 1e9)  # GB/s
    
    # Measure device to host bandwidth (GPU -> CPU)
    d2h_times = []
    for _ in range(iterations):
        start = time.time()
        cpu_data.copy_(gpu_data)
        torch.cuda.synchronize()
        end = time.time()
        d2h_times.append(end - start)
    
    d2h_bandwidth = size_bytes / (min(d2h_times) * 1e9)  # GB/s
    
    return h2d_bandwidth, d2h_bandwidth

def measure_bandwidth_pycuda(size_mb, iterations=10, pinned=True):
    """Measure PCIe bandwidth using PyCUDA backend"""
    import pycuda.driver as cuda
    
    # Convert MB to bytes
    size_bytes = size_mb * 1024 * 1024
    
    # Create CPU array
    if pinned:
        cpu_data = cuda.pagelocked_empty(size_bytes // 4, dtype=np.float32)
        cpu_data_out = cuda.pagelocked_empty(size_bytes // 4, dtype=np.float32)
    else:
        cpu_data = np.random.randn(size_bytes // 4).astype(np.float32)
        cpu_data_out = np.empty(size_bytes // 4, dtype=np.float32)
    
    # Fill with random data
    if pinned:
        cpu_data_np = np.asarray(cpu_data)
        cpu_data_np[:] = np.random.randn(size_bytes // 4).astype(np.float32)
    
    # Allocate GPU memory
    gpu_data = cuda.mem_alloc(size_bytes)
    
    # Warmup
    for _ in range(2):
        cuda.memcpy_htod(gpu_data, cpu_data)
        cuda.memcpy_dtoh(cpu_data_out, gpu_data)
    
    # Measure host to device bandwidth (CPU -> GPU)
    h2d_times = []
    for _ in range(iterations):
        start = time.time()
        cuda.memcpy_htod(gpu_data, cpu_data)
        cuda.Context.synchronize()
        end = time.time()
        h2d_times.append(end - start)
    
    h2d_bandwidth = size_bytes / (min(h2d_times) * 1e9)  # GB/s
    
    # Measure device to host bandwidth (GPU -> CPU)
    d2h_times = []
    for _ in range(iterations):
        start = time.time()
        cuda.memcpy_dtoh(cpu_data_out, gpu_data)
        cuda.Context.synchronize()
        end = time.time()
        d2h_times.append(end - start)
    
    d2h_bandwidth = size_bytes / (min(d2h_times) * 1e9)  # GB/s
    
    return h2d_bandwidth, d2h_bandwidth

def measure_bandwidth_opencl(size_mb, iterations=10):
    """Measure PCIe bandwidth using OpenCL backend"""
    import pyopencl as cl
    
    # Convert MB to bytes
    size_bytes = size_mb * 1024 * 1024
    
    # Get platform and device
    platforms = cl.get_platforms()
    gpu_devices = []
    
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        gpu_devices.extend(devices)
    
    if not gpu_devices:
        raise RuntimeError("No OpenCL GPU device found")
    
    device = gpu_devices[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    # Create host array
    cpu_data = np.random.rand(size_bytes // 4).astype(np.float32)
    cpu_data_out = np.empty_like(cpu_data)
    
    # Create device buffer
    gpu_data = cl.Buffer(context, cl.mem_flags.READ_WRITE, size=size_bytes)
    
    # Warmup
    for _ in range(2):
        cl.enqueue_copy(queue, gpu_data, cpu_data)
        cl.enqueue_copy(queue, cpu_data_out, gpu_data)
        queue.finish()
    
    # Measure host to device bandwidth (CPU -> GPU)
    h2d_times = []
    for _ in range(iterations):
        start = time.time()
        cl.enqueue_copy(queue, gpu_data, cpu_data)
        queue.finish()
        end = time.time()
        h2d_times.append(end - start)
    
    h2d_bandwidth = size_bytes / (min(h2d_times) * 1e9)  # GB/s
    
    # Measure device to host bandwidth (GPU -> CPU)
    d2h_times = []
    for _ in range(iterations):
        start = time.time()
        cl.enqueue_copy(queue, cpu_data_out, gpu_data)
        queue.finish()
        end = time.time()
        d2h_times.append(end - start)
    
    d2h_bandwidth = size_bytes / (min(d2h_times) * 1e9)  # GB/s
    
    return h2d_bandwidth, d2h_bandwidth

def measure_pcie_bandwidth(sizes_mb=[16, 64, 256, 1024], iterations=10):
    """
    Measure PCIe bandwidth at different data sizes and return the results.
    Tests both host-to-device and device-to-host transfers.
    """
    results = {}
    
    # Check system RAM to avoid testing too large sizes
    total_ram_gb = 0
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong)
                ]
            memory_status = MEMORYSTATUS()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memory_status))
            total_ram_gb = memory_status.dwTotalPhys / (1024**3)
            
        elif platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        total_ram_kb = int(line.split()[1])
                        total_ram_gb = total_ram_kb / (1024**2)
                        break
                        
        elif platform.system() == "Darwin":  # macOS
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            total_ram_bytes = int(result.stdout.strip())
            total_ram_gb = total_ram_bytes / (1024**3)
            
    except Exception as e:
        logger.warning(f"Could not determine system RAM: {e}")
        total_ram_gb = 8  # Assume 8GB as a safe default
    
    # Filter sizes to avoid OOM
    max_size_mb = int(total_ram_gb * 1024 * 0.25)  # Use at most 25% of RAM
    filtered_sizes = [size for size in sizes_mb if size <= max_size_mb]
    
    if not filtered_sizes:
        filtered_sizes = [min(sizes_mb)]  # Use at least the smallest size
        
    logger.info(f"Testing PCIe bandwidth with data sizes: {filtered_sizes} MB")
    
    # Measure bandwidth for each size
    for size_mb in filtered_sizes:
        logger.info(f"Testing {size_mb} MB transfers...")
        
        try:
            if BACKEND == "pytorch":
                h2d, d2h = measure_bandwidth_pytorch(size_mb, iterations)
            elif BACKEND == "pycuda":
                h2d, d2h = measure_bandwidth_pycuda(size_mb, iterations)
            elif BACKEND == "opencl":
                h2d, d2h = measure_bandwidth_opencl(size_mb, iterations)
            
            results[size_mb] = {
                "h2d_bandwidth": h2d,
                "d2h_bandwidth": d2h
            }
            
            logger.info(f"  Host to Device: {h2d:.2f} GB/s")
            logger.info(f"  Device to Host: {d2h:.2f} GB/s")
            
        except Exception as e:
            logger.error(f"Error measuring {size_mb} MB: {e}")
            continue
    
    return results

def estimate_pcie_gen(bandwidth_gb_s):
    """Estimate PCIe generation based on measured bandwidth"""
    # Note: These are very rough estimates and depend on many factors
    if bandwidth_gb_s >= 20.0:
        return "PCIe 4.0 x16 or better"
    elif bandwidth_gb_s >= 15.0:
        return "PCIe 3.0 x16 or PCIe 4.0 x8"
    elif bandwidth_gb_s >= 7.5:
        return "PCIe 3.0 x8 or PCIe 4.0 x4"
    elif bandwidth_gb_s >= 3.5:
        return "PCIe 3.0 x4 or PCIe 2.0 x8"
    else:
        return "PCIe 2.0 or lower"

def main():
    parser = argparse.ArgumentParser(description="GPU PCIe Bandwidth Benchmark")
    parser.add_argument("--sizes", type=str, default="16,64,256,1024",
                        help="Comma-separated list of data sizes to test (in MB)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of iterations for each test")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    logger.info("GPU PCIe Bandwidth Benchmark")
    logger.info("=" * 40)
    
    # Parse sizes
    sizes_mb = list(map(int, args.sizes.split(',')))
    
    # Get device info
    device_info = get_device_info()
    logger.info(f"Device: {device_info['device']}")
    logger.info(f"Backend: {device_info['backend']}")
    
    try:
        # Run the benchmark
        bandwidth_results = measure_pcie_bandwidth(sizes_mb, args.iterations)
        
        if not bandwidth_results:
            logger.error("No valid results obtained")
            return 1
        
        # Calculate average bandwidth across all tested sizes
        h2d_values = [result["h2d_bandwidth"] for result in bandwidth_results.values()]
        d2h_values = [result["d2h_bandwidth"] for result in bandwidth_results.values()]
        
        avg_h2d = sum(h2d_values) / len(h2d_values)
        avg_d2h = sum(d2h_values) / len(d2h_values)
        
        # Find the best result (usually from larger transfers)
        best_h2d = max(h2d_values)
        best_d2h = max(d2h_values)
        
        # Estimate PCIe generation based on bandwidth
        # We use the better of H2D or D2H as the indicator
        best_bandwidth = max(best_h2d, best_d2h)
        pcie_estimate = estimate_pcie_gen(best_bandwidth)
        
        # Create final results structure
        final_results = {
            "TransferPCIe": best_bandwidth,
            "TransferPCIeUnit": "GB/s",
            "TransferH2D": best_h2d,
            "TransferH2DUnit": "GB/s",
            "TransferD2H": best_d2h,
            "TransferD2HUnit": "GB/s",
            "PCIeGeneration": pcie_estimate,
            "DetailedResults": bandwidth_results,
            "DeviceInfo": device_info
        }
        
        # Output results
        if args.json:
            # Only print the JSON result
            print(json.dumps(final_results, indent=2))
        else:
            # Print detailed results to console
            logger.info("\nSummary:")
            logger.info(f"Host to Device Bandwidth: {best_h2d:.2f} GB/s")
            logger.info(f"Device to Host Bandwidth: {best_d2h:.2f} GB/s")
            logger.info(f"PCIe Bandwidth: {best_bandwidth:.2f} GB/s")
            logger.info(f"Estimated PCIe: {pcie_estimate}")
            
            # Print JSON at the end for the runner to parse
            print("\n" + json.dumps(final_results))
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        
        error_results = {
            "error": f"PCIe benchmark error: {str(e)}"
        }
        
        if args.json:
            print(json.dumps(error_results, indent=2))
        else:
            print("\n" + json.dumps(error_results))
            
        return 1

if __name__ == "__main__":
    sys.exit(main()) 