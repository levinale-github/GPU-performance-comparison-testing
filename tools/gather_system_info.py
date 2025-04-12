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
import pyamdgpuinfo  # For AMD GPUs
import pyopencl as cl  # For OpenCL devices

def get_cpu_info():
    """Get CPU information from /proc/cpuinfo"""
    cpu_info = {}
    try:
        if os.path.exists('/proc/cpuinfo'):
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
    """Get memory information from /proc/meminfo"""
    memory_info = {}
    try:
        if os.path.exists('/proc/meminfo'):
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        memory_info[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error getting memory info: {e}")
        memory_info['error'] = str(e)
    
    return memory_info

def get_pcie_info():
    """Get PCIe device information"""
    pcie_info = []
    try:
        # Use lspci to get PCIe device information
        result = subprocess.run(['lspci', '-v'], capture_output=True, text=True)
        if result.returncode == 0:
            devices = result.stdout.split('\n\n')
            for device in devices:
                if 'VGA' in device or '3D' in device:
                    device_info = {}
                    lines = device.split('\n')
                    device_info['pci_id'] = lines[0].split()[0]
                    device_info['description'] = ' '.join(lines[0].split()[1:])
                    
                    # Extract additional information
                    for line in lines[1:]:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            device_info[key.strip()] = value.strip()
                    
                    pcie_info.append(device_info)
    except Exception as e:
        print(f"Error getting PCIe info: {e}")
        pcie_info.append({'error': str(e)})
    
    return pcie_info

def get_nvidia_gpu_info():
    """Get detailed NVIDIA GPU information"""
    nvidia_info = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            device_info = {
                'name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                'uuid': pynvml.nvmlDeviceGetUUID(handle).decode('utf-8'),
                'memory_total': info.total,
                'memory_free': info.free,
                'memory_used': info.used,
                'temperature': pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                'power_usage': pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,  # Convert to watts
                'pcie_gen': pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle),
                'pcie_width': pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            }
            nvidia_info.append(device_info)
    except Exception as e:
        print(f"Error getting NVIDIA GPU info: {e}")
        nvidia_info.append({'error': str(e)})
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    
    return nvidia_info

def get_amd_gpu_info():
    """Get detailed AMD GPU information"""
    amd_info = []
    try:
        for i in range(pyamdgpuinfo.detect_gpus()):
            gpu = pyamdgpuinfo.get_gpu(i)
            device_info = {
                'name': gpu.name,
                'memory_total': gpu.memory_info['vram_size'],
                'memory_used': gpu.memory_info['vram_used'],
                'temperature': gpu.query_temperature(),
                'power_usage': gpu.query_power(),
                'pcie_gen': gpu.query_pcie_gen(),
                'pcie_width': gpu.query_pcie_width()
            }
            amd_info.append(device_info)
    except Exception as e:
        print(f"Error getting AMD GPU info: {e}")
        amd_info.append({'error': str(e)})
    
    return amd_info

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
        'pcie_devices': get_pcie_info(),
        'nvidia_gpus': get_nvidia_gpu_info(),
        'amd_gpus': get_amd_gpu_info(),
        'opencl_devices': get_opencl_info(),
        'timestamp': platform.datetime.datetime.now().isoformat()
    }
    
    return system_info

if __name__ == "__main__":
    # Print system information as JSON
    print(json.dumps(get_system_info(), indent=2)) 