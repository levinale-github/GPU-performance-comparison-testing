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
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType', '-json'], capture_output=True, text=True)
            if result.returncode == 0:
                try:
                    # Try to parse the JSON output for more structured data
                    data = json.loads(result.stdout)
                    displays = data.get('SPDisplaysDataType', [])
                    for display in displays:
                        if 'spdisplays_mtlgpufamilysupport' in display:  # Metal GPU
                            gpu_info = {
                                'GPU': display.get('spdisplays_device-name', ''),
                                'GPUVendor': 'Apple',
                                'VRAM': display.get('spdisplays_vram', ''),
                                'MetalFamily': display.get('spdisplays_mtlgpufamilysupport', ''),
                                'DisplayType': display.get('spdisplays_display_type', ''),
                                'CoreCount': display.get('spdisplays_gpu_cores', ''),
                                'Resolution': display.get('spdisplays_resolution', ''),
                                'DeviceID': display.get('spdisplays_device-id', '')
                            }
                            
                            # Add additional details for Apple Silicon GPUs
                            # Try to get GPU frequency
                            try:
                                freq_result = subprocess.run(['sysctl', '-n', 'hw.gpufrequency'], capture_output=True, text=True)
                                if freq_result.returncode == 0 and freq_result.stdout.strip():
                                    gpu_info['ClockFrequency'] = f"{int(freq_result.stdout.strip()) / 1000000} MHz"
                            except:
                                pass
                                
                            if 'Apple' in gpu_info['GPU'] or (not gpu_info['GPU'] and 'Apple' in platform.processor()):
                                # Try to identify chip model and extract info
                                chip_model = ''
                                try:
                                    chip_result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                                              capture_output=True, text=True)
                                    if chip_result.returncode == 0:
                                        chip_model = chip_result.stdout.strip()
                                        # Extract the chip identifier (M1, M2, etc.)
                                        chip_match = re.search(r'Apple (M\d+)( [A-Za-z]+)?', chip_model)
                                        if chip_match:
                                            gpu_info['ChipModel'] = chip_match.group(0)
                                            
                                            # Estimate core count if not provided
                                            if not gpu_info['CoreCount']:
                                                # Estimated GPU cores based on chip model
                                                chip_type = chip_match.group(1)  # M1, M2, etc.
                                                chip_variant = chip_match.group(2).strip() if chip_match.group(2) else ''
                                                
                                                # Rough estimates based on chip type
                                                core_estimates = {
                                                    'M1': {'': 8, 'Pro': 16, 'Max': 32, 'Ultra': 64},
                                                    'M2': {'': 10, 'Pro': 19, 'Max': 38, 'Ultra': 76},
                                                    'M3': {'': 10, 'Pro': 20, 'Max': 40, 'Ultra': 80},
                                                    'M4': {'': 12, 'Pro': 24, 'Max': 48, 'Ultra': 96}
                                                }
                                                
                                                if chip_type in core_estimates and chip_variant in core_estimates[chip_type]:
                                                    gpu_info['CoreCount'] = str(core_estimates[chip_type][chip_variant])
                                except:
                                    pass
                                
                            apple_info.append(gpu_info)
                except json.JSONDecodeError:
                    # Fall back to text parsing if JSON fails
                    gpu_info = {'GPUVendor': 'Apple'}
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            if 'Chipset Model' in key:
                                gpu_info['GPU'] = value
                            elif 'Vendor' in key:
                                gpu_info['GPUVendor'] = value
                            elif 'VRAM' in key:
                                gpu_info['VRAM'] = value
                            elif 'Metal Family' in key:
                                gpu_info['MetalFamily'] = value
                            elif 'Resolution' in key:
                                gpu_info['Resolution'] = value
                            elif 'Device ID' in key:
                                gpu_info['DeviceID'] = value
                    
                    if gpu_info:
                        apple_info.append(gpu_info)
                        
            # Get additional thermal/power info for more comprehensive data
            try:
                powermetrics_result = subprocess.run(['sudo', 'powermetrics', '-n', '1', '-i', '1000', '--samplers', 'gpu_power'], 
                                                    capture_output=True, text=True, timeout=2)
                if powermetrics_result.returncode == 0:
                    for line in powermetrics_result.stdout.split('\n'):
                        if 'GPU power:' in line:
                            power_value = line.split(':', 1)[1].strip()
                            if apple_info and isinstance(apple_info[0], dict):
                                apple_info[0]['PowerUsage'] = power_value
            except:
                # Powermetrics requires sudo, so this might fail - that's okay
                pass
                
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