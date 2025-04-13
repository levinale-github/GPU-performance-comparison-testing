#!/usr/bin/env python3
"""
GpuTest benchmark for the GPU benchmark suite.
This benchmark uses GpuTest to measure graphics performance across multiple tests.

GpuTest is a cross-platform GPU stress test and benchmark tool:
http://www.geeks3d.com/gputest/
"""

import os
import sys
import json
import time
import platform
import subprocess
import argparse
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GpuTest download URLs by platform
GPUTEST_URLS = {
    "Windows": "https://www.geeks3d.com/dl/get/587",  # GpuTest_Windows_x64_0.7.0.zip
    "Darwin": "https://www.geeks3d.com/dl/get/383",   # GpuTest_OSX_x64_0.7.0.zip
    "Linux": "https://www.geeks3d.com/dl/get/585"     # GpuTest_Linux_x64_0.7.0.zip
}

# GpuTest executable names by platform
GPUTEST_EXECUTABLES = {
    "Windows": "GpuTest.exe",
    "Darwin": "GpuTest.app/Contents/MacOS/GpuTest",
    "Linux": "GpuTest"
}

# Test names available in GpuTest
AVAILABLE_TESTS = [
    "fur",          # FurMark test
    "fillrate",     # Fill rate test
    "triangle",     # Triangle test
    "plot3d",       # 3D plot test
    "tess_x8",      # Tessellation test (factor 8)
    "tess_x16",     # Tessellation test (factor 16)
    "tess_x32",     # Tessellation test (factor 32)
    "tess_x64"      # Tessellation test (factor 64)
]

def detect_gputest():
    """Check if GpuTest is installed or can be installed."""
    system = platform.system()
    
    # Check common installation paths
    common_paths = []
    
    if system == "Windows":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        common_paths = [
            Path(program_files) / "GpuTest",
            Path(program_files_x86) / "GpuTest",
            Path.home() / "GpuTest"
        ]
    elif system == "Darwin":
        common_paths = [
            Path("/Applications/GpuTest.app"),
            Path.home() / "Applications/GpuTest.app"
        ]
    elif system == "Linux":
        common_paths = [
            Path("/opt/gputest"),
            Path("/usr/local/gputest"),
            Path.home() / "gputest"
        ]
    
    # Check for the executable in common paths
    for path in common_paths:
        executable_path = path / GPUTEST_EXECUTABLES.get(system, "")
        if executable_path.exists():
            return str(executable_path)
    
    return None

def download_gputest():
    """
    Download and extract GpuTest for the current platform.
    Returns the path to the executable if successful, None otherwise.
    """
    system = platform.system()
    if system not in GPUTEST_URLS:
        logger.error(f"Unsupported platform: {system}")
        return None
    
    url = GPUTEST_URLS[system]
    download_dir = Path(tempfile.mkdtemp())
    zip_path = download_dir / "gputest.zip"
    
    try:
        logger.info(f"Downloading GpuTest from {url}")
        # Download using curl or wget
        if system == "Windows":
            subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri '{url}' -OutFile '{zip_path}'"], check=True)
        else:
            if subprocess.run(["which", "curl"], capture_output=True).returncode == 0:
                subprocess.run(["curl", "-L", "-o", str(zip_path), url], check=True)
            else:
                subprocess.run(["wget", "-O", str(zip_path), url], check=True)
        
        # Extract the zip file
        logger.info(f"Extracting GpuTest to {download_dir}")
        if system == "Windows":
            subprocess.run(["powershell", "-Command", f"Expand-Archive -Path '{zip_path}' -DestinationPath '{download_dir}'"], check=True)
        else:
            if subprocess.run(["which", "unzip"], capture_output=True).returncode == 0:
                subprocess.run(["unzip", str(zip_path), "-d", str(download_dir)], check=True)
            else:
                logger.error("unzip command not found. Please install unzip.")
                return None
        
        # Find the executable
        executable_name = GPUTEST_EXECUTABLES[system]
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file == os.path.basename(executable_name):
                    executable_path = os.path.join(root, file)
                    # Make executable
                    if system != "Windows":
                        os.chmod(executable_path, 0o755)
                    return executable_path
        
        logger.error(f"Could not find GpuTest executable in the extracted archive.")
        return None
        
    except Exception as e:
        logger.error(f"Error downloading or extracting GpuTest: {e}")
        return None

def run_gputest_benchmark(executable, test_name="fur", width=1280, height=720, duration=10):
    """
    Run a specific GpuTest benchmark.
    
    Args:
        executable: Path to the GpuTest executable
        test_name: Name of the test to run (e.g., "fur", "fillrate", "triangle")
        width: Screen width
        height: Screen height
        duration: Test duration in seconds
        
    Returns:
        Dictionary with benchmark results
    """
    system = platform.system()
    result_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    result_file.close()
    
    try:
        cmd = [
            executable,
            f"/test={test_name}",
            f"/width={width}",
            f"/height={height}",
            f"/benchmark_duration={duration}",
            f"/benchmark_result_file={result_file.name}"
        ]
        
        # Run GpuTest
        logger.info(f"Running GpuTest benchmark: {test_name}")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"GpuTest failed with return code {process.returncode}")
            logger.error(f"Error output: {stderr}")
            return {"error": stderr}
        
        # Parse the result file (XML format)
        if os.path.exists(result_file.name):
            results = parse_gputest_result(result_file.name)
            return results
        else:
            logger.error(f"Result file not found: {result_file.name}")
            return {"error": "Result file not found"}
            
    except Exception as e:
        logger.error(f"Error running GpuTest: {e}")
        return {"error": str(e)}
    finally:
        # Clean up
        if os.path.exists(result_file.name):
            os.unlink(result_file.name)

def parse_gputest_result(result_file):
    """
    Parse the GpuTest XML result file.
    
    Args:
        result_file: Path to the XML result file
        
    Returns:
        Dictionary with parsed results
    """
    import xml.etree.ElementTree as ET
    
    try:
        tree = ET.parse(result_file)
        root = tree.getroot()
        
        results = {}
        
        # Parse basic benchmark information
        benchmark_elem = root.find("benchmark")
        if benchmark_elem is not None:
            results["test_name"] = benchmark_elem.get("name", "unknown")
            
            # Parse score
            score_elem = benchmark_elem.find("score")
            if score_elem is not None:
                score = float(score_elem.text)
                results["score"] = score
                
            # Parse fps
            fps_elem = benchmark_elem.find("fps")
            if fps_elem is not None:
                min_fps = float(fps_elem.get("min", "0"))
                max_fps = float(fps_elem.get("max", "0"))
                avg_fps = float(fps_elem.text)
                results["fps"] = {
                    "min": min_fps,
                    "max": max_fps,
                    "avg": avg_fps
                }
                
        # Parse renderer information
        renderer_elem = root.find("renderer")
        if renderer_elem is not None:
            renderer_info = {}
            for child in renderer_elem:
                renderer_info[child.tag] = child.text
            results["renderer"] = renderer_info
            
        return results
        
    except Exception as e:
        logger.error(f"Error parsing GpuTest result file: {e}")
        return {"error": f"Error parsing result file: {str(e)}"}

def gputest_installation_instructions():
    """Return platform-specific installation instructions for GpuTest."""
    system = platform.system()
    
    if system == "Windows":
        return """
To install GpuTest on Windows:
1. Download GpuTest from https://www.geeks3d.com/gputest/
2. Extract the ZIP file to a folder of your choice
3. Run GpuTest.exe
"""
    elif system == "Darwin":
        return """
To install GpuTest on macOS:
1. Download GpuTest from https://www.geeks3d.com/gputest/
2. Extract the ZIP file
3. Move GpuTest.app to your Applications folder
"""
    elif system == "Linux":
        return """
To install GpuTest on Linux:
1. Download GpuTest from https://www.geeks3d.com/gputest/
2. Extract the ZIP file to a folder of your choice
3. Make the GpuTest file executable: chmod +x GpuTest
"""
    else:
        return f"GpuTest installation instructions not available for {system}."

def run_all_tests(executable, duration=10):
    """
    Run all available GpuTest tests and combine the results.
    
    Args:
        executable: Path to the GpuTest executable
        duration: Duration of each test in seconds
        
    Returns:
        Dictionary with results from all tests
    """
    results = {}
    
    # Run each test
    for test in AVAILABLE_TESTS:
        try:
            logger.info(f"Running {test} test...")
            test_results = run_gputest_benchmark(executable, test_name=test, duration=duration)
            results[test] = test_results
        except Exception as e:
            logger.error(f"Error running {test} test: {e}")
            results[test] = {"error": str(e)}
    
    # Calculate overall score (geometric mean of all test scores)
    scores = []
    for test_name, test_result in results.items():
        if "score" in test_result:
            scores.append(test_result["score"])
    
    if scores:
        import math
        overall_score = math.exp(sum(math.log(score) for score in scores) / len(scores))
        results["overall_score"] = overall_score
    
    return results

def main():
    """Main function to run the GpuTest benchmark."""
    parser = argparse.ArgumentParser(description="GPU Graphics Performance Benchmark using GpuTest")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--test", choices=AVAILABLE_TESTS + ["all"], default="all",
                      help="Specific GpuTest to run (default: all)")
    parser.add_argument("--duration", type=int, default=10,
                      help="Duration of each test in seconds (default: 10)")
    args = parser.parse_args()
    
    # Initialize results
    results = {
        "error": None,
        "GpuTest": {
            "installed": False,
            "version": "0.7.0"  # Hardcoded for now
        }
    }
    
    try:
        # Check if GpuTest is installed
        executable = detect_gputest()
        
        # If not installed, try to download it
        if executable is None:
            logger.info("GpuTest not found. Attempting to download...")
            executable = download_gputest()
        
        if executable is None:
            # If we still don't have it, return an error
            results["error"] = "GpuTest not found and could not be downloaded."
            results["installation_instructions"] = gputest_installation_instructions()
        else:
            results["GpuTest"]["installed"] = True
            logger.info(f"Using GpuTest at {executable}")
            
            # Run the requested test(s)
            if args.test == "all":
                test_results = run_all_tests(executable, duration=args.duration)
                results["tests"] = test_results
                if "overall_score" in test_results:
                    results["GpuTestScore"] = test_results["overall_score"]
            else:
                test_results = run_gputest_benchmark(executable, test_name=args.test, duration=args.duration)
                results["tests"] = {args.test: test_results}
                if "score" in test_results:
                    results["GpuTestScore"] = test_results["score"]
    
    except Exception as e:
        results["error"] = f"Error during benchmark: {str(e)}"
    
    # Output results
    if args.json:
        print(json.dumps(results))
    else:
        if "error" in results and results["error"]:
            logger.error(f"Error: {results['error']}")
            if "installation_instructions" in results:
                logger.info(results["installation_instructions"])
        else:
            logger.info("\nGpuTest Benchmark Results:")
            if "GpuTestScore" in results:
                logger.info(f"Overall Score: {results['GpuTestScore']:.2f}")
            
            if "tests" in results:
                for test_name, test_result in results["tests"].items():
                    if test_name != "overall_score":
                        if "score" in test_result:
                            logger.info(f"{test_name} test score: {test_result['score']:.2f}")
                        if "fps" in test_result:
                            logger.info(f"{test_name} FPS: {test_result['fps']['avg']:.2f} (min: {test_result['fps']['min']:.2f}, max: {test_result['fps']['max']:.2f})")
    
    return 0 if not results.get("error") else 1

if __name__ == "__main__":
    sys.exit(main()) 