#!/usr/bin/env python3
"""
GPU Compute Performance Benchmark

This script measures the floating-point computation performance of the GPU,
including single-precision (FP32) and double-precision (FP64) operations.
Results are reported in GFLOPS (Giga Floating-Point Operations Per Second).
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
logger = logging.getLogger("Compute-Benchmark")

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

# OpenCL Kernel for single precision compute (FP32)
FP32_KERNEL = """
__kernel void compute_fp32(__global float *result, const int iterations) {
    // Each work item does many FLOPs in a loop to maximize computation vs memory transfer
    float a = 1.0f;
    float b = 0.99999f;
    
    // Each iteration performs 8 floating-point operations (4 multiplies, 4 adds)
    for (int i = 0; i < iterations; ++i) {
        a = a * a + b;
        b = b * b + a;
        a = a * a + b;
        b = b * b + a;
    }
    
    // Store the result to prevent the compiler from optimizing away the calculation
    result[get_global_id(0)] = a + b;
}
"""

# OpenCL Kernel for double precision compute (FP64)
FP64_KERNEL = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void compute_fp64(__global double *result, const int iterations) {
    // Each work item does many FLOPs in a loop to maximize computation vs memory transfer
    double a = 1.0;
    double b = 0.99999;
    
    // Each iteration performs 8 floating-point operations (4 multiplies, 4 adds)
    for (int i = 0; i < iterations; ++i) {
        a = a * a + b;
        b = b * b + a;
        a = a * a + b;
        b = b * b + a;
    }
    
    // Store the result to prevent the compiler from optimizing away the calculation
    result[get_global_id(0)] = a + b;
}
"""

def get_opencl_devices():
    """Get available OpenCL devices with information about their capabilities"""
    platforms = cl.get_platforms()
    devices = []
    
    for platform in platforms:
        for device in platform.get_devices():
            # Gather device info
            device_info = {
                "platform": platform.name,
                "device": device.name,
                "type": cl.device_type.to_string(device.type),
                "max_compute_units": device.max_compute_units,
                "max_work_group_size": device.max_work_group_size,
                "supports_fp64": False,  # Will check below
                "global_mem_size": device.global_mem_size,
                "local_mem_size": device.local_mem_size
            }
            
            # Check for double precision support
            try:
                extensions = device.extensions.strip().split()
                device_info["supports_fp64"] = "cl_khr_fp64" in extensions
            except:
                pass
                
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
        fp64_support = "Yes" if device_info["supports_fp64"] else "No"
        logger.info(f"  [{i}] {device_info['device']} ({device_info['type']}, {device_info['platform']})")
        logger.info(f"      Compute Units: {device_info['max_compute_units']}, FP64: {fp64_support}")
    
    # First try to find a GPU
    for device, device_info in devices:
        if "GPU" in device_info['type']:
            logger.info(f"\nSelected GPU device: {device_info['device']}")
            return device, device_info
    
    # If no GPU, use first available device
    logger.info(f"\nNo GPU found, using: {devices[0][1]['device']}")
    return devices[0]

def measure_flops(device, device_info, test_seconds=2.0, iterations_per_kernel=1000):
    """
    Measure FLOPS (Floating-Point Operations Per Second) for the GPU.
    Returns results for both single (FP32) and double (FP64) precision.
    """
    # Create context and command queue
    context = cl.Context([device])
    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    # Build the kernel program
    try:
        program = cl.Program(context, FP32_KERNEL + FP64_KERNEL).build()
    except cl.RuntimeError as e:
        logger.error(f"Error building OpenCL program: {e}")
        logger.error("Trying to build single-precision kernel only...")
        program = cl.Program(context, FP32_KERNEL).build()
    
    # Determine work sizes based on device capabilities
    global_size = min(device_info["max_compute_units"] * 2048, 1024 * 1024)  # Heuristic
    local_size = min(device_info["max_work_group_size"], 256)  # Conservative
    
    # Ensure global_size is a multiple of local_size
    global_size = (global_size // local_size) * local_size
    
    logger.info(f"Using global work size: {global_size}, local work size: {local_size}")
    
    # Results dictionary
    results = {}
    
    # ---- Single Precision (FP32) Test ----
    
    # Create buffer for results (one float per work item)
    result_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, global_size * np.dtype(np.float32).itemsize)
    
    # Warm-up run
    program.compute_fp32(queue, (global_size,), (local_size,), result_buffer, np.int32(10))
    queue.finish()
    
    # Start with a reasonable number of iterations
    iterations = iterations_per_kernel
    
    # First do a calibration run to adjust iterations for desired test duration
    start_time = time.time()
    program.compute_fp32(queue, (global_size,), (local_size,), result_buffer, np.int32(iterations))
    queue.finish()
    end_time = time.time()
    
    # Calculate elapsed time and adjust iterations
    elapsed = end_time - start_time
    iterations = int(iterations * (test_seconds / elapsed))
    iterations = max(iterations, 1)  # Ensure at least 1 iteration
    
    logger.info(f"Calibrated to {iterations} iterations for FP32 test")
    
    # Main measurement
    fp32_times = []
    
    for i in range(3):  # Run 3 times for consistency
        # Run the kernel
        start_time = time.time()
        program.compute_fp32(queue, (global_size,), (local_size,), result_buffer, np.int32(iterations))
        queue.finish()
        end_time = time.time()
        
        elapsed = end_time - start_time
        fp32_times.append(elapsed)
        
        # Calculate FLOPS:
        # - Each iteration does 8 floating-point operations
        # - This is done by global_size work items
        # - For iterations times
        total_flops = global_size * iterations * 8
        gflops = total_flops / elapsed / 1e9
        
        logger.info(f"  Run {i+1}: {gflops:.2f} GFLOPS (FP32)")
    
    # Calculate average FP32 GFLOPS
    avg_elapsed = sum(fp32_times) / len(fp32_times)
    total_flops = global_size * iterations * 8
    avg_fp32_gflops = total_flops / avg_elapsed / 1e9
    
    results["FP32"] = {
        "value": avg_fp32_gflops,
        "unit": "GFLOPS",
        "raw_values": [total_flops / t / 1e9 for t in fp32_times]
    }
    
    # ---- Double Precision (FP64) Test ----
    if device_info["supports_fp64"]:
        try:
            # Create buffer for double-precision results
            result_buffer_dp = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, global_size * np.dtype(np.float64).itemsize)
            
            # Warm-up run
            program.compute_fp64(queue, (global_size,), (local_size,), result_buffer_dp, np.int32(10))
            queue.finish()
            
            # Calibration run - double precision is usually much slower
            # Start with half the iterations of single precision
            dp_iterations = max(iterations // 4, 1)
            
            start_time = time.time()
            program.compute_fp64(queue, (global_size,), (local_size,), result_buffer_dp, np.int32(dp_iterations))
            queue.finish()
            end_time = time.time()
            
            elapsed = end_time - start_time
            dp_iterations = int(dp_iterations * (test_seconds / elapsed))
            dp_iterations = max(dp_iterations, 1)
            
            logger.info(f"Calibrated to {dp_iterations} iterations for FP64 test")
            
            # Main measurement
            fp64_times = []
            
            for i in range(3):  # Run 3 times for consistency
                start_time = time.time()
                program.compute_fp64(queue, (global_size,), (local_size,), result_buffer_dp, np.int32(dp_iterations))
                queue.finish()
                end_time = time.time()
                
                elapsed = end_time - start_time
                fp64_times.append(elapsed)
                
                # Calculate FLOPS (same formula as FP32)
                total_flops = global_size * dp_iterations * 8
                gflops = total_flops / elapsed / 1e9
                
                logger.info(f"  Run {i+1}: {gflops:.2f} GFLOPS (FP64)")
            
            # Calculate average FP64 GFLOPS
            avg_elapsed = sum(fp64_times) / len(fp64_times)
            total_flops = global_size * dp_iterations * 8
            avg_fp64_gflops = total_flops / avg_elapsed / 1e9
            
            results["FP64"] = {
                "value": avg_fp64_gflops,
                "unit": "GFLOPS",
                "raw_values": [total_flops / t / 1e9 for t in fp64_times]
            }
            
            # Calculate FP64/FP32 ratio (useful for architecture comparison)
            ratio = avg_fp64_gflops / avg_fp32_gflops
            logger.info(f"FP64/FP32 ratio: {ratio:.4f}")
            results["FP64_FP32_Ratio"] = ratio
            
        except cl.RuntimeError as e:
            logger.warning(f"Error running double-precision test: {e}")
            logger.warning("Device claims to support FP64 but the test failed.")
            results["FP64"] = {"error": "Test failed"}
    else:
        logger.warning("Device does not support double precision (FP64).")
        results["FP64"] = {"error": "Not supported"}
    
    return results

def main():
    parser = argparse.ArgumentParser(description="GPU Compute Performance Benchmark")
    parser.add_argument("--duration", type=float, default=2.0, 
                        help="Duration in seconds for each test")
    parser.add_argument("--iterations", type=int, default=1000,
                        help="Base number of iterations per kernel (will be auto-adjusted)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    logger.info("GPU Compute Performance Benchmark")
    logger.info("=" * 40)
    
    try:
        # Select OpenCL device
        device, device_info = select_device()
        
        # Run FLOPS measurements
        flops_results = measure_flops(
            device,
            device_info,
            test_seconds=args.duration,
            iterations_per_kernel=args.iterations
        )
        
        # Prepare output
        results = {
            "FP32": flops_results["FP32"]["value"] if "FP32" in flops_results else 0,
            "FP32Unit": "GFLOPS",
        }
        
        # Add FP64 results if available
        if "FP64" in flops_results and "value" in flops_results["FP64"]:
            results["FP64"] = flops_results["FP64"]["value"]
            results["FP64Unit"] = "GFLOPS"
            
            if "FP64_FP32_Ratio" in flops_results:
                results["FP64_FP32_Ratio"] = flops_results["FP64_FP32_Ratio"]
        
        # Add device info
        results["DeviceInfo"] = device_info
        
        # Output results
        if args.json:
            # Only print the JSON result
            print(json.dumps(results, indent=2))
        else:
            # Print detailed results to console
            logger.info("\nSummary:")
            logger.info(f"Device: {device_info['device']} ({device_info['platform']})")
            logger.info(f"Compute Units: {device_info['max_compute_units']}")
            logger.info("-" * 40)
            logger.info(f"Single Precision: {results['FP32']:.2f} GFLOPS (FP32)")
            
            if "FP64" in results:
                logger.info(f"Double Precision: {results['FP64']:.2f} GFLOPS (FP64)")
                if "FP64_FP32_Ratio" in results:
                    logger.info(f"FP64/FP32 Ratio: {results['FP64_FP32_Ratio']:.4f}")
            else:
                logger.info("Double Precision: Not supported/available")
            
            # Print JSON at the end for the runner to parse
            print("\n" + json.dumps(results))
        
        return 0
        
    except cl.Error as e:
        logger.error(f"OpenCL error: {e}")
        
        error_results = {
            "error": f"OpenCL error: {str(e)}"
        }
        
        if args.json:
            print(json.dumps(error_results, indent=2))
        else:
            print("\n" + json.dumps(error_results))
        
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