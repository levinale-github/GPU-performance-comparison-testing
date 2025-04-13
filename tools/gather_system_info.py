#!/usr/bin/env python3
"""
System Information Gatherer

This script collects detailed system and GPU information from various sources:
- CPU information from /proc/cpuinfo
- Memory information from /proc/meminfo
- GPU information from PCIe config space
- Additional system information from various sources
"""

import os
import re
import json
import subprocess
from pathlib import Path
import platform
import psutil
import pynvml  # For NVIDIA GPUs
import pyopencl as cl  # For OpenCL devices
from datetime import datetime

def get_cpu_info():
    """Get CPU information"""
    cpu_info = {}
    try:
        if platform.system() == 'Darwin':  # macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info['model'] = result.stdout.strip()
            
            result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info['processor_count'] = int(result.stdout.strip())
        elif os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read()
                
            # Extract processor information
            processors = content.split('\n\n')
            cpu_info['processor_count'] = len([p for p in processors if p.startswith('processor')])
            
            # Get model name from first processor
            model_match = re.search(r'model name\s*:\s*(.*)', processors[0])
            if model_match:
                cpu_info['model'] = model_match.group(1).strip()
            
            # Get CPU flags
            flags_match = re.search(r'flags\s*:\s*(.*)', processors[0])
            if flags_match:
                cpu_info['flags'] = flags_match.group(1).strip().split()
    except Exception as e:
        print(f"Error getting CPU info: {e}")
        cpu_info['error'] = str(e)
    
    return cpu_info

def get_memory_info():
    """Get memory information"""
    memory_info = {}
    try:
        if platform.system() == 'Darwin':  # macOS
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                memory_info['MemTotal'] = f"{int(result.stdout.strip()) // 1024} kB"
        elif os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        memory_info[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error getting memory info: {e}")
        memory_info['error'] = str(e)
    
    return memory_info

def get_apple_gpu_info():
    """Get Apple Silicon GPU information"""
    apple_info = []
    try:
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            # Get GPU info from system profiler
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = {}
                for line in result.stdout.split('\n'):
                    if 'Chipset Model:' in line:
                        gpu_info['name'] = line.split(':', 1)[1].strip()
                    elif 'VRAM (Total):' in line:
                        gpu_info['memory_total'] = line.split(':', 1)[1].strip()
                
                if gpu_info:
                    apple_info.append(gpu_info)
    except Exception as e:
        print(f"Error getting Apple GPU info: {e}")
        apple_info.append({'error': str(e)})
    
    return apple_info

def get_opencl_info():
    """Get OpenCL device information"""
    opencl_info = []
    try:
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                device_info = {
                    'platform': platform.name,
                    'name': device.name,
                    'vendor': device.vendor,
                    'version': device.version,
                    'type': str(device.type),
                    'max_compute_units': device.max_compute_units,
                    'max_clock_frequency': device.max_clock_frequency,
                    'global_mem_size': device.global_mem_size,
                    'local_mem_size': device.local_mem_size
                }
                opencl_info.append(device_info)
    except Exception as e:
        print(f"Error getting OpenCL info: {e}")
        opencl_info.append({'error': str(e)})
    
    return opencl_info

def get_system_info():
    """Get comprehensive system information"""
    system_info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation()
        },
        'cpu': get_cpu_info(),
        'memory': get_memory_info(),
        'apple_gpus': get_apple_gpu_info(),
        'opencl_devices': get_opencl_info(),
        'timestamp': datetime.now().isoformat()
    }
    
    return system_info

if __name__ == "__main__":
    # Print system information as JSON
    print(json.dumps(get_system_info(), indent=2)) 