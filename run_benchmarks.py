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

def run_benchmark_script(script_path):
    """
    Runs an individual benchmark script and returns its results.
    """
    logger.info(f"Running benchmark: {script_path}")
    
    try:
        # Determine if script is Python or shell
        script_path_str = str(script_path)
        if script_path_str.endswith('.py'):
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True
            )
        else:
            # Make sure the script is executable
            os.chmod(script_path, 0o755)
            result = subprocess.run(
                [script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
        logger.info(f"Benchmark completed: {script_path}")
        
        # Try to parse JSON output if available
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from benchmark output: {script_path}")
            return {"raw_output": result.stdout.strip()}
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Benchmark failed: {script_path}")
        logger.error(f"Exit code: {e.returncode}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return {"error": f"Benchmark failed with exit code {e.returncode}"}
        
    except Exception as e:
        logger.error(f"Error running benchmark {script_path}: {str(e)}")
        return {"error": str(e)}

def run_all_benchmarks():
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
    for category in category_mapping:
        category_dir = BENCHMARK_DIR / category
        if category_dir.exists():
            for file in category_dir.glob("**/*"):
                if (file.is_file() and 
                    (file.name.endswith(".py") or file.name.endswith(".sh")) and
                    not file.name.startswith("_") and
                    os.access(file, os.X_OK)):
                    benchmark_scripts.append((category, file))
    
    # Special case for PCIe bandwidth benchmark
    # Include it even if the pcie directory doesn't exist yet
    pcie_bandwidth_script = BENCHMARK_DIR / "memory" / "pcie_bandwidth.py"
    if pcie_bandwidth_script.exists() and os.access(pcie_bandwidth_script, os.X_OK):
        benchmark_scripts.append(("pcie", pcie_bandwidth_script))
    
    # Run each benchmark
    for category, script in benchmark_scripts:
        result = run_benchmark_script(script)
        
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
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    # Log start of benchmark run
    logger.info("Starting GPU benchmark suite")
    
    # Get GPU and system info
    gpu_info = detect_gpu()
    sys_info = get_system_info()
    
    logger.info(f"Detected GPU: {gpu_info['GPU']}")
    logger.info(f"System: {sys_info['OS']}, {sys_info['CPUModel']}, {sys_info['RAMSize']}")
    
    # Run all benchmarks
    benchmark_results = run_all_benchmarks()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"All benchmarks completed in {elapsed_time:.2f} seconds")
    
    # Prepare results object
    timestamp = datetime.datetime.now().isoformat()
    
    results = {
        "metadata": {
            **gpu_info,
            **sys_info,
            "Contributor": args.contributor,
            "Notes": args.notes,
            "Timestamp": timestamp
        },
        "results": benchmark_results
    }
    
    # Add cost information if provided
    if any([args.retail_price, args.purchase_date, args.cloud_equivalent, args.cloud_cost]):
        results["metadata"]["Cost"] = {}
        
        if args.retail_price:
            results["metadata"]["Cost"]["RetailPrice"] = args.retail_price
            results["metadata"]["Cost"]["Currency"] = "USD"
            
        if args.purchase_date:
            results["metadata"]["Cost"]["PurchaseDate"] = args.purchase_date
            
        if args.cloud_equivalent:
            results["metadata"]["Cost"]["CloudEquivalent"] = args.cloud_equivalent
            
        if args.cloud_cost:
            results["metadata"]["Cost"]["CloudCostPerHour"] = args.cloud_cost
    
    # Generate output filename if not provided
    if not args.output:
        # Clean the GPU name to use in filename
        gpu_name_safe = gpu_info["GPU"].replace(" ", "_").replace("/", "_")
        gpu_name_safe = ''.join(c for c in gpu_name_safe if c.isalnum() or c in '-_')
        
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        output_filename = f"{date_str}_{gpu_info['GPUVendor']}_{gpu_name_safe}.json"
        output_path = RESULTS_SUBMISSIONS_DIR / output_filename
    else:
        output_path = Path(args.output)
    
    # Write results to file
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results written to {output_path}")
    
    # Copy log file to logs directory
    log_filename = output_path.stem + ".log"
    log_dest_path = RESULTS_LOGS_DIR / log_filename
    
    # Wait a moment to ensure all logs are flushed
    time.sleep(0.5)
    
    try:
        import shutil
        shutil.copy("benchmark_run.log", log_dest_path)
        logger.info(f"Log copied to {log_dest_path}")
    except Exception as e:
        logger.error(f"Failed to copy log file: {e}")
    
    print("\nBenchmark run completed!")
    print(f"Results saved to: {output_path}")
    print(f"Log saved to: {log_dest_path}")
    
    print("\nTo contribute these results to the repository:")
    print("1. Fork the repository on GitHub")
    print("2. Add your results file to the 'results/submissions/' directory")
    print("3. Add your log file to the 'results/logs/' directory")
    print("4. Create a pull request")
    print("See CONTRIBUTING.md for detailed instructions.")

if __name__ == "__main__":
    main() 