#!/usr/bin/env python3
"""
GPU Benchmarking & Performance Testing - Main Runner Script

This script runs all available benchmarks and collects their results
into a single JSON file for submission.
"""

import os
import sys
import json
import time
import datetime
import argparse
import logging
import platform
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GPU-Benchmark")

# Base directories
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR / "benchmarks"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_SUBMISSIONS_DIR = RESULTS_DIR / "submissions"
RESULTS_LOGS_DIR = RESULTS_DIR / "logs"

# Ensure directories exist
RESULTS_SUBMISSIONS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_LOGS_DIR.mkdir(exist_ok=True, parents=True)

# Add tools directory to path
sys.path.append(str(Path(__file__).parent / 'tools'))

from gather_system_info import get_system_info

def detect_gpu():
    """
    Attempts to detect the GPU model and driver version.
    Returns a dictionary with GPU information.
    """
    gpu_info = {
        "GPU": "Unknown",
        "GPUVendor": "Unknown",
        "Driver": "Unknown",
        "VRAM": "Unknown"
    }
    
    try:
        if platform.system() == "Windows":
            # On Windows, use WMIC to get GPU info
            import wmi
            w = wmi.WMI()
            gpu_devices = w.Win32_VideoController()
            
            if gpu_devices:
                gpu = gpu_devices[0]  # Get the first GPU
                gpu_info["GPU"] = gpu.Name
                
                # Determine vendor from name
                if "NVIDIA" in gpu.Name.upper():
                    gpu_info["GPUVendor"] = "NVIDIA"
                elif "AMD" in gpu.Name.upper() or "RADEON" in gpu.Name.upper():
                    gpu_info["GPUVendor"] = "AMD"
                elif "INTEL" in gpu.Name.upper():
                    gpu_info["GPUVendor"] = "Intel"
                
                gpu_info["Driver"] = gpu.DriverVersion
                gpu_info["VRAM"] = f"{int(gpu.AdapterRAM / (1024**2))} MB" if hasattr(gpu, "AdapterRAM") else "Unknown"
                
        elif platform.system() == "Linux":
            # On Linux, try lspci and glxinfo
            try:
                lspci_output = subprocess.check_output(["lspci", "-v"], text=True)
                
                # Look for GPU in lspci output
                for line in lspci_output.split("\n"):
                    if "VGA" in line or "3D controller" in line:
                        if "NVIDIA" in line:
                            gpu_info["GPUVendor"] = "NVIDIA"
                            gpu_info["GPU"] = line.split(":")[-1].strip()
                        elif "AMD" in line or "ATI" in line:
                            gpu_info["GPUVendor"] = "AMD"
                            gpu_info["GPU"] = line.split(":")[-1].strip()
                        elif "Intel" in line:
                            gpu_info["GPUVendor"] = "Intel"
                            gpu_info["GPU"] = line.split(":")[-1].strip()
                
                # Try to get driver info with glxinfo
                glxinfo_output = subprocess.check_output(["glxinfo"], text=True)
                for line in glxinfo_output.split("\n"):
                    if "OpenGL version string" in line:
                        gpu_info["Driver"] = line.split(":")[-1].strip()
                    if "Video memory" in line:
                        gpu_info["VRAM"] = line.split(":")[-1].strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("Failed to get GPU info using lspci/glxinfo")
                
        elif platform.system() == "Darwin":  # macOS
            try:
                # Use system_profiler to get GPU info on macOS
                output = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
                for line in output.split("\n"):
                    line = line.strip()
                    if "Chipset Model" in line:
                        gpu_info["GPU"] = line.split(":")[-1].strip()
                    if "Vendor" in line:
                        gpu_info["GPUVendor"] = line.split(":")[-1].strip()
                    if "VRAM" in line:
                        gpu_info["VRAM"] = line.split(":")[-1].strip()
                    if "Metal" in line and "family" in line:
                        gpu_info["Driver"] = line.split(":")[-1].strip()
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("Failed to get GPU info on macOS")
    
    except Exception as e:
        logger.error(f"Error detecting GPU: {e}")
    
    return gpu_info

def run_benchmark_script(script_path, device=None):
    """
    Runs an individual benchmark script and returns its results.
    """
    logger.info(f"Running benchmark: {script_path}")
    
    try:
        # Determine if script is Python or shell
        script_path_str = str(script_path)
        if script_path_str.endswith('.py'):
            cmd = [sys.executable, script_path_str]  # Convert Path to string
            
            # Always add --json flag for Python scripts
            cmd.append("--json")
            
            # Only add device argument for AI benchmarks that support it
            if device and "ai" in str(script_path):
                cmd.extend(["--device", device])
                
            # Add specific arguments for PCIe bandwidth benchmark
            if script_path.name == "pcie_bandwidth.py":
                cmd.extend([
                    "--sizes", "16,64,256,1024",  # Test with different transfer sizes
                    "--iterations", "5"           # Number of iterations per size
                ])
                
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero exit
            )
            
            # Log any stderr output
            if result.stderr:
                logger.warning(f"Stderr output: {result.stderr}")
            
            # Check return code
            if result.returncode != 0:
                return {
                    "error": f"Benchmark failed with exit code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                
        else:
            # Make sure the script is executable
            os.chmod(script_path, 0o755)
            result = subprocess.run(
                [script_path_str],  # Convert Path to string
                capture_output=True,
                text=True,
                check=False
            )
            
        logger.info(f"Benchmark completed: {script_path}")
        
        # Try to parse JSON output if available
        try:
            # Try to find JSON in the output
            output = result.stdout.strip()
            # Look for the last occurrence of a JSON-like structure
            json_start = output.rfind("{")
            json_end = output.rfind("}")
            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end + 1]
                return json.loads(json_str)
            else:
                logger.warning(f"No JSON found in output: {output}")
                return {"error": "No JSON found in output", "raw_output": output}
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON from benchmark output: {script_path}")
            logger.warning(f"JSON error: {e}")
            logger.warning(f"Raw output: {result.stdout}")
            return {"error": f"Failed to parse JSON output: {str(e)}", "raw_output": result.stdout.strip()}
            
    except Exception as e:
        logger.error(f"Error running benchmark {script_path}: {str(e)}")
        return {"error": str(e)}

def run_all_benchmarks(device=None):
    """
    Runs all benchmarks in the benchmark directory.
    Returns a dictionary with results from all benchmarks.
    """
    results = {
        "memory": {},
        "compute": {},
        "graphics": {},
        "ai": {},
        "pcie": {}  # Add PCIe category to collect PCIe-specific benchmarks
    }
    
    # Dictionary mapping directory names to result categories
    category_mapping = {
        "memory": "memory",
        "compute": "compute",
        "graphics": "graphics",
        "ai": "ai",
        "pcie": "pcie"  # Add mapping for PCIe directory
    }
    
    # Find all benchmark scripts
    benchmark_scripts = []
    processed_scripts = set()  # Keep track of processed scripts to avoid duplicates
    
    # First, add the PCIe bandwidth benchmark if it exists
    pcie_bandwidth_script = BENCHMARK_DIR / "memory" / "pcie_bandwidth.py"
    if pcie_bandwidth_script.exists() and os.access(pcie_bandwidth_script, os.X_OK):
        benchmark_scripts.append(("pcie", pcie_bandwidth_script))
        processed_scripts.add(pcie_bandwidth_script)
    
    # Then process other benchmarks
    for category in category_mapping:
        category_dir = BENCHMARK_DIR / category
        if category_dir.exists():
            for file in category_dir.glob("**/*"):
                if (file.is_file() and 
                    (file.name.endswith(".py") or file.name.endswith(".sh")) and
                    not file.name.startswith("_") and
                    os.access(file, os.X_OK) and
                    file not in processed_scripts):  # Skip if already processed
                    benchmark_scripts.append((category, file))
                    processed_scripts.add(file)
    
    # Run each benchmark
    for category, script in benchmark_scripts:
        result = run_benchmark_script(script, device)
        
        # Add result to appropriate category
        result_category = category_mapping.get(category, "other")
        
        # If the result is a dictionary, update the category
        if isinstance(result, dict):
            results[result_category].update(result)
        else:
            # Otherwise store as a raw result
            results[result_category][script.stem] = result
    
    return results

def main():
    parser = argparse.ArgumentParser(description="GPU Benchmark Suite Runner")
    parser.add_argument("--output", "-o", help="Output file name", default=None)
    parser.add_argument("--contributor", help="Your name or identifier for the result submission", default="")
    parser.add_argument("--retail-price", type=float, help="Retail price of your GPU in USD", default=None)
    parser.add_argument("--purchase-date", help="Purchase date of your GPU (YYYY-MM-DD)", default=None)
    parser.add_argument("--cloud-equivalent", help="Cloud instance equivalent", default=None)
    parser.add_argument("--cloud-cost", type=float, help="Hourly cost of cloud instance in USD", default=None)
    parser.add_argument("--notes", help="Additional notes about your submission", default="")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                      help="Device to run benchmarks on (auto, cuda, mps, or cpu)")
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Log start of benchmark run
    logger.info("Starting GPU benchmark suite")
    
    # Get system information
    sys_info = get_system_info()
    
    # Detect GPU
    gpu_info = detect_gpu()
    logger.info(f"Detected GPU: {gpu_info['GPU']}")
    
    # Log system information
    logger.info(f"System: {sys_info['platform']['system']} {sys_info['platform']['release']}, {sys_info['cpu']['model']}, {sys_info['memory'].get('MemTotal', 'Unknown RAM')}")
    
    # Run all benchmarks
    results = run_all_benchmarks(args.device)
    
    # Calculate total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    
    # Prepare metadata
    metadata = {
        "system": {
            "os": f"{sys_info['platform']['system']} {sys_info['platform']['release']}",
            "cpu": sys_info['cpu'],
            "memory": sys_info['memory'],
            "gpu": gpu_info
        },
        "contributor": args.contributor,
        "timestamp": datetime.datetime.now().isoformat(),
        "runtime": total_runtime,
        "notes": args.notes
    }
    
    # Add cost information if provided
    if args.retail_price or args.cloud_cost:
        metadata["cost"] = {
            "retail_price_usd": args.retail_price,
            "purchase_date": args.purchase_date,
            "cloud_equivalent": args.cloud_equivalent,
            "cloud_cost_per_hour": args.cloud_cost
        }
    
    # Combine results and metadata
    final_results = {
        "metadata": metadata,
        "results": results
    }
    
    # Generate output filename if not provided
    if not args.output:
        # Create a filename based on GPU and date
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        gpu_name = gpu_info["GPU"].replace(" ", "_")
        args.output = RESULTS_SUBMISSIONS_DIR / f"{date_str}_{gpu_name}.json"
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")
    
    # Copy log file to results/logs with matching name
    log_filename = output_path.stem + ".log"
    log_path = RESULTS_LOGS_DIR / log_filename
    if os.path.exists("benchmark_run.log"):
        import shutil
        shutil.copy2("benchmark_run.log", log_path)
        logger.info(f"Log file copied to: {log_path}")
    
    return 0

if __name__ == "__main__":
    main() 