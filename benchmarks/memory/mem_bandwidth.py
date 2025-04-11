#!/usr/bin/env python3
"""
GPU Memory Bandwidth Benchmark

This script measures the memory bandwidth of the GPU using PyOpenCL.
It runs a simple kernel that copies data between buffers and measures
the throughput in GB/s.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Memory-Bandwidth")

# Add parent directory to path to import common utilities
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Try to import PyOpenCL - show helpful error if not available
try:
    import numpy as np
    import pyopencl as cl
except ImportError as e:
    logger.error(f"Error importing required libraries: {e}")
    logger.error("\nThis benchmark requires NumPy and PyOpenCL.")
    logger.error("Please install them with:")
    logger.error("  pip install numpy pyopencl")
    logger.error("\nOn some systems you may need OpenCL development files:")
    logger.error("  - Ubuntu/Debian: apt install opencl-headers ocl-icd-opencl-dev")
    logger.error("  - Fedora/RHEL: dnf install ocl-icd-devel opencl-headers")
    logger.error("  - Windows: Install GPU vendor's OpenCL SDK")
    sys.exit(1)

# OpenCL Kernel for copy test
COPY_KERNEL = """
__kernel void copy_kernel(__global const float *src, __global float *dst) {
    int gid = get_global_id(0);
    dst[gid] = src[gid];
}
"""

# OpenCL Kernel for read test
READ_KERNEL = """
__kernel void read_kernel(__global const float *src, __global float *sum) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    
    __local float local_sum[256];
    local_sum[lid] = src[gid];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce within work-group
    for (int stride = group_size/2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_sum[lid] += local_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Only first thread in group writes result
    if (lid == 0) {
        sum[get_group_id(0)] = local_sum[0];
    }
}
"""

# OpenCL Kernel for write test
WRITE_KERNEL = """
__kernel void write_kernel(__global float *dst) {
    int gid = get_global_id(0);
    dst[gid] = (float)gid; // Write a simple pattern
}
"""

def get_opencl_devices():
    """Get available OpenCL devices with names and compute details"""
    platforms = cl.get_platforms()
    devices = []
    
    for platform in platforms:
        for device in platform.get_devices():
            device_info = {
                "platform": platform.name,
                "device": device.name,
                "type": cl.device_type.to_string(device.type),
                "max_compute_units": device.max_compute_units,
                "global_mem_size": device.global_mem_size,
                "max_work_group_size": device.max_work_group_size
            }
            devices.append((device, device_info))
    
    return devices

def select_device():
    """Select the best OpenCL device for benchmarking (GPU preferred)"""
    devices = get_opencl_devices()
    
    if not devices:
        logger.error("No OpenCL devices found!")
        sys.exit(1)
    
    # Print available devices
    logger.info("Available OpenCL devices:")
    for i, (_, device_info) in enumerate(devices):
        logger.info(f"  [{i}] {device_info['device']} ({device_info['type']}, {device_info['platform']})")
    
    # First try to find a GPU
    for device, device_info in devices:
        if "GPU" in device_info['type']:
            logger.info(f"\nSelected GPU device: {device_info['device']}")
            return device, device_info
    
    # If no GPU, use first available device
    logger.info(f"\nNo GPU found, using: {devices[0][1]['device']}")
    return devices[0]

def measure_bandwidth(device, mem_size_mb=256, iterations=5):
    """
    Measure memory bandwidth of the selected OpenCL device.
    Returns results in GB/s.
    """
    # Create a context and queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Compile kernels
    program = cl.Program(context, COPY_KERNEL + READ_KERNEL + WRITE_KERNEL).build()
    
    # Allocate memory (in MB)
    mem_size = mem_size_mb * 1024 * 1024  # Convert to bytes
    elem_size = np.dtype(np.float32).itemsize
    num_elements = mem_size // elem_size
    
    # Check if memory size exceeds device capacity
    if mem_size > device.global_mem_size * 0.8:  # Use at most 80% of available memory
        logger.warning(f"Reducing buffer size to fit in device memory")
        mem_size = int(device.global_mem_size * 0.8)
        num_elements = mem_size // elem_size
    
    logger.info(f"Testing with buffer size: {mem_size/1024/1024:.1f} MB")
    
    # Create arrays and buffers
    np.random.seed(42)  # For reproducibility
    src_data = np.random.rand(num_elements).astype(np.float32)
    dst_data = np.zeros_like(src_data)
    
    # Create OpenCL buffers
    src_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=src_data)
    dst_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, dst_data.nbytes)
    
    # Define benchmarks
    benchmarks = [
        {
            "name": "Copy (Read+Write)",
            "kernel": program.copy_kernel,
            "args": [src_buf, dst_buf],
            "global_size": (num_elements,),
            "local_size": None,
            "direction": "bidirectional"
        },
        {
            "name": "Read",
            "kernel": program.read_kernel,
            "args": [src_buf, cl.Buffer(context, cl.mem_flags.WRITE_ONLY, 4 * ((num_elements + 255) // 256))],
            "global_size": (num_elements,),
            "local_size": (256,),
            "direction": "read"
        },
        {
            "name": "Write",
            "kernel": program.write_kernel,
            "args": [dst_buf],
            "global_size": (num_elements,),
            "local_size": None,
            "direction": "write"
        }
    ]
    
    results = {}
    
    for benchmark in benchmarks:
        # Warmup
        event = benchmark["kernel"](
            queue, benchmark["global_size"], benchmark["local_size"], *benchmark["args"]
        )
        event.wait()
        
        # Measure multiple iterations
        bandwidths = []
        for i in range(iterations):
            # Make sure previous operations are done
            queue.finish()
            
            # Start timing
            start_time = time.time()
            
            # Launch kernel
            event = benchmark["kernel"](
                queue, benchmark["global_size"], benchmark["local_size"], *benchmark["args"]
            )
            event.wait()
            
            # End timing
            end_time = time.time()
            
            # Calculate time and bandwidth
            elapsed_ns = (end_time - start_time) * 1e9  # convert to nanoseconds
            
            # For bidirectional operations, count both read and write
            scale_factor = 2.0 if benchmark["direction"] == "bidirectional" else 1.0
            
            # Calculate bandwidth in GB/s
            # (mem_size * scale_factor) bytes / elapsed_ns nanoseconds * 10^9 ns/s / 10^9 B/GB
            bandwidth_gb_per_s = (mem_size * scale_factor) / elapsed_ns
            
            bandwidths.append(bandwidth_gb_per_s)
            
            logger.debug(f"{benchmark['name']} - Iteration {i+1}: {bandwidth_gb_per_s:.2f} GB/s")
        
        # Calculate average and standard deviation
        avg_bandwidth = np.mean(bandwidths)
        std_bandwidth = np.std(bandwidths)
        
        logger.info(f"{benchmark['name']} Bandwidth: {avg_bandwidth:.2f} GB/s (±{std_bandwidth:.2f})")
        
        results[benchmark["name"]] = {
            "value": avg_bandwidth,
            "unit": "GB/s",
            "std_dev": std_bandwidth
        }
    
    # Calculate average bandwidth across all tests
    avg_overall = np.mean([results[b["name"]]["value"] for b in benchmarks])
    results["Average"] = {
        "value": avg_overall,
        "unit": "GB/s"
    }
    
    return results

def fallback_system_info():
    """Get GPU info using other methods if OpenCL fails"""
    import subprocess
    import platform
    
    gpu_info = {
        "device": "Unknown",
        "platform": "Unknown",
        "global_mem_size": 0
    }
    
    system = platform.system()
    
    try:
        if system == "Windows":
            # Try using WMIC on Windows
            try:
                import wmi
                w = wmi.WMI()
                for gpu in w.Win32_VideoController():
                    gpu_info["device"] = gpu.Name
                    if hasattr(gpu, "AdapterRAM"):
                        gpu_info["global_mem_size"] = gpu.AdapterRAM
                    break
            except:
                pass
                
        elif system == "Linux":
            # Try using lspci on Linux
            try:
                output = subprocess.check_output(["lspci", "-v"], text=True)
                for line in output.split('\n'):
                    if "VGA" in line or "3D controller" in line:
                        gpu_info["device"] = line.split(":")[-1].strip()
                        break
            except:
                pass
                
            # Try using nvidia-smi for NVIDIA GPUs
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True)
                parts = output.strip().split(",")
                if len(parts) >= 2:
                    gpu_info["device"] = parts[0].strip()
                    mem_str = parts[1].strip()
                    if "MiB" in mem_str:
                        mem_mb = int(mem_str.split(" ")[0])
                        gpu_info["global_mem_size"] = mem_mb * 1024 * 1024
            except:
                pass
                
        elif system == "Darwin":  # macOS
            try:
                output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
                for line in output.split('\n'):
                    if "Chipset Model" in line:
                        gpu_info["device"] = line.split(":")[-1].strip()
                    if "VRAM" in line:
                        vram_str = line.split(":")[-1].strip()
                        if "MB" in vram_str:
                            mem_mb = int(vram_str.split(" ")[0])
                            gpu_info["global_mem_size"] = mem_mb * 1024 * 1024
                        elif "GB" in vram_str:
                            mem_gb = float(vram_str.split(" ")[0])
                            gpu_info["global_mem_size"] = int(mem_gb * 1024 * 1024 * 1024)
            except:
                pass
    except Exception as e:
        logger.warning(f"Error getting fallback system info: {e}")
    
    return gpu_info

def main():
    parser = argparse.ArgumentParser(description="GPU Memory Bandwidth Benchmark")
    parser.add_argument("--size", type=int, default=256, 
                        help="Buffer size in MB for bandwidth test")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of test iterations")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    logger.info("GPU Memory Bandwidth Benchmark")
    logger.info("=" * 40)
    
    try:
        # Select OpenCL device
        device, device_info = select_device()
        
        # Memory measurements
        bandwidth_results = measure_bandwidth(
            device, 
            mem_size_mb=args.size,
            iterations=args.iterations
        )
        
        # Prepare output
        results = {
            "Bandwidth": bandwidth_results["Average"]["value"],
            "BandwidthUnit": "GB/s",
            "CopyBandwidth": bandwidth_results["Copy (Read+Write)"]["value"],
            "CopyBandwidthUnit": "GB/s",
            "ReadBandwidth": bandwidth_results["Read"]["value"],
            "ReadBandwidthUnit": "GB/s",
            "WriteBandwidth": bandwidth_results["Write"]["value"],
            "WriteBandwidthUnit": "GB/s",
            "DeviceInfo": device_info
        }
        
        # Output results
        if args.json:
            # Only print the JSON result
            print(json.dumps(results, indent=2))
        else:
            # Print detailed results to console
            logger.info("\nSummary:")
            logger.info(f"Device: {device_info['device']} ({device_info['platform']})")
            logger.info(f"Memory Size: {device_info['global_mem_size'] / (1024*1024*1024):.2f} GB")
            logger.info(f"Compute Units: {device_info['max_compute_units']}")
            logger.info("-" * 40)
            logger.info(f"Average Bandwidth: {results['Bandwidth']:.2f} GB/s")
            logger.info(f"Copy Bandwidth: {results['CopyBandwidth']:.2f} GB/s")
            logger.info(f"Read Bandwidth: {results['ReadBandwidth']:.2f} GB/s")
            logger.info(f"Write Bandwidth: {results['WriteBandwidth']:.2f} GB/s")
            
            # Print JSON at the end for the runner to parse
            print("\n" + json.dumps(results))
        
        return 0
    
    except cl.Error as e:
        logger.error(f"OpenCL error: {e}")
        
        # Try to get basic info about the GPU without OpenCL
        gpu_info = fallback_system_info()
        logger.warning("Falling back to system detection without running bandwidth tests")
        logger.info(f"Detected GPU: {gpu_info['device']}")
        
        fallback_results = {
            "error": "OpenCL error: Unable to run benchmark",
            "DeviceInfo": {
                "device": gpu_info["device"],
                "platform": gpu_info["platform"],
                "global_mem_size": gpu_info["global_mem_size"]
            }
        }
        
        if args.json:
            print(json.dumps(fallback_results, indent=2))
        else:
            print("\n" + json.dumps(fallback_results))
        
        return 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        
        error_results = {
            "error": f"Benchmark error: {str(e)}"
        }
        
        if args.json:
            print(json.dumps(error_results, indent=2))
        else:
            print("\n" + json.dumps(error_results))
            
        return 1

if __name__ == "__main__":
    sys.exit(main()) 