#!/usr/bin/env python3
"""
LuxMark benchmark for the GPU benchmark suite.
This benchmark uses LuxMark to measure OpenCL compute performance.

LuxMark is an OpenCL benchmark tool: https://github.com/LuxCoreRender/LuxMark
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
import shutil
import re
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache directory for downloaded files
CACHE_DIR = Path.home() / ".gpu_benchmark_cache" / "luxmark"

# LuxMark download URLs by platform
LUXMARK_URLS = {
    "Windows": "https://github.com/LuxCoreRender/LuxMark/releases/download/luxmark_v4.0alpha0/luxmark-v4.0alpha0-win64.zip",
    "Darwin": "https://github.com/LuxCoreRender/LuxMark/releases/download/luxmark_v4.0alpha0/luxmark-v4.0alpha0-mac64.dmg",
    "Linux": "https://github.com/LuxCoreRender/LuxMark/releases/download/luxmark_v4.0alpha0/luxmark-v4.0alpha0-linux64.tar.bz2"
}

# LuxMark executable names by platform
LUXMARK_EXECUTABLES = {
    "Windows": "luxmark.exe",
    "Darwin": "LuxMark.app/Contents/MacOS/luxmark",
    "Linux": "luxmark"
}

# Available scenes in LuxMark
AVAILABLE_SCENES = ["food", "hotel", "microphone"]

# LuxMark scene names to human-readable names mapping
SCENE_DISPLAY_NAMES = {
    "food": "Food",
    "hotel": "Hotel Lobby",
    "microphone": "Microphone"
}

def detect_luxmark():
    """Check if LuxMark is installed or can be installed."""
    system = platform.system()
    
    # Check common installation paths
    common_paths = []
    
    if system == "Windows":
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        common_paths = [
            Path(program_files) / "LuxMark",
            Path(program_files_x86) / "LuxMark",
            Path.home() / "LuxMark"
        ]
    elif system == "Darwin":
        common_paths = [
            Path("/Applications/LuxMark.app"),
            Path.home() / "Applications/LuxMark.app"
        ]
    elif system == "Linux":
        common_paths = [
            Path("/opt/luxmark"),
            Path("/usr/local/bin"),
            Path.home() / "luxmark"
        ]
    
    # Check for the executable in common paths
    for path in common_paths:
        executable_path = path / LUXMARK_EXECUTABLES.get(system, "")
        if executable_path.exists():
            return str(executable_path)
    
    return None

def download_luxmark():
    """
    Download and extract LuxMark for the current platform.
    Returns the path to the executable if successful, None otherwise.
    """
    system = platform.system()
    if system not in LUXMARK_URLS:
        logger.error(f"Unsupported platform: {system}")
        return None
    
    url = LUXMARK_URLS[system]
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if we have a cached version
    cached_file = CACHE_DIR / os.path.basename(url)
    cached_extract_dir = CACHE_DIR / "extracted"
    cached_executable = None
    
    # Check if we have already extracted the executable
    if cached_extract_dir.exists():
        executable_name = LUXMARK_EXECUTABLES[system]
        for root, dirs, files in os.walk(cached_extract_dir):
            for file in files:
                if file == os.path.basename(executable_name) or (system == "Darwin" and root.endswith("MacOS") and file == "luxmark"):
                    cached_executable = os.path.join(root, file)
                    if os.path.exists(cached_executable):
                        logger.info(f"Using cached LuxMark executable at {cached_executable}")
                        return cached_executable
    
    # If no cached executable was found but the file exists, try extracting it again
    if cached_file.exists():
        logger.info(f"Archive file found in cache. Using {cached_file}")
        download_file = cached_file
    else:
        # Need to download the file
        # Use a temporary directory for downloading
        download_dir = Path(tempfile.mkdtemp())
        download_file = download_dir / os.path.basename(url)
        
        logger.info(f"Downloading LuxMark from {url}")
        
        # Download using curl or wget
        try:
            if system == "Windows":
                subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri '{url}' -OutFile '{download_file}'"], check=True)
            else:
                if subprocess.run(["which", "curl"], capture_output=True).returncode == 0:
                    subprocess.run(["curl", "-L", "-o", str(download_file), url], check=True)
                else:
                    subprocess.run(["wget", "-O", str(download_file), url], check=True)
            
            # Copy to cache directory
            shutil.copy2(download_file, cached_file)
            download_file = cached_file
        except Exception as e:
            logger.error(f"Error downloading LuxMark: {e}")
            return None
    
    # Clear any previous extraction directory
    if cached_extract_dir.exists():
        shutil.rmtree(cached_extract_dir)
    cached_extract_dir.mkdir(exist_ok=True)
    
    try:
        # Extract based on file type
        if system == "Windows" or (system == "Linux" and url.endswith(".zip")):
            # Extract ZIP
            if system == "Windows":
                subprocess.run(["powershell", "-Command", f"Expand-Archive -Path '{download_file}' -DestinationPath '{cached_extract_dir}'"], check=True)
            else:
                if subprocess.run(["which", "unzip"], capture_output=True).returncode == 0:
                    subprocess.run(["unzip", str(download_file), "-d", str(cached_extract_dir)], check=True)
                else:
                    logger.error("unzip command not found. Please install unzip.")
                    return None
        
        elif system == "Darwin" and url.endswith(".dmg"):
            # Mount DMG
            mount_point = "/Volumes/LuxMark"
            try:
                # Verify DMG file is valid
                verify_cmd = subprocess.run(["hdiutil", "verify", str(download_file)], capture_output=True, text=True)
                if verify_cmd.returncode != 0:
                    logger.error(f"Invalid DMG file: {verify_cmd.stderr}")
                    # Try redownloading using wget instead if curl didn't work well
                    if os.path.exists(cached_file):
                        os.remove(cached_file)  # Remove corrupted file
                    
                    if os.path.exists("/usr/local/bin/wget") or os.path.exists("/opt/homebrew/bin/wget"):
                        logger.info("Retrying download with wget...")
                        subprocess.run(["wget", "-O", str(cached_file), url], check=True)
                    else:
                        logger.info("Retrying download with curl -L option...")
                        subprocess.run(["curl", "-L", "-o", str(cached_file), url], check=True)
                    
                    download_file = cached_file
                
                # Attempt to mount
                subprocess.run(["hdiutil", "attach", str(download_file)], check=True)
                
                # Copy app to cache directory
                shutil.copytree(f"{mount_point}/LuxMark.app", cached_extract_dir / "LuxMark.app")
                
                # Unmount DMG
                subprocess.run(["hdiutil", "detach", mount_point], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error mounting DMG file: {e}")
                logger.info("Trying to mount with nobrowse option...")
                try:
                    # Try an alternative mounting approach
                    subprocess.run(["hdiutil", "attach", "-nobrowse", str(download_file)], check=True)
                    
                    # Check where it was mounted
                    volumes = subprocess.run(["ls", "/Volumes"], capture_output=True, text=True).stdout.strip().split('\n')
                    for volume in volumes:
                        if "LuxMark" in volume:
                            mount_point = f"/Volumes/{volume}"
                            
                    if not os.path.exists(f"{mount_point}/LuxMark.app"):
                        logger.error(f"LuxMark.app not found in {mount_point}")
                        return None
                        
                    # Copy app to cache directory
                    shutil.copytree(f"{mount_point}/LuxMark.app", cached_extract_dir / "LuxMark.app")
                    
                    # Unmount DMG
                    subprocess.run(["hdiutil", "detach", mount_point], check=True)
                except Exception as inner_e:
                    logger.error(f"Failed to mount DMG with alternative approach: {inner_e}")
                    return None
        
        elif system == "Linux" and (url.endswith(".tar.gz") or url.endswith(".tar.bz2")):
            # Extract TAR
            if url.endswith(".tar.gz"):
                subprocess.run(["tar", "-xzf", str(download_file), "-C", str(cached_extract_dir)], check=True)
            elif url.endswith(".tar.bz2"):
                subprocess.run(["tar", "-xjf", str(download_file), "-C", str(cached_extract_dir)], check=True)
        
        # Find the executable
        executable_name = LUXMARK_EXECUTABLES[system]
        for root, dirs, files in os.walk(cached_extract_dir):
            for file in files:
                if file == os.path.basename(executable_name) or (system == "Darwin" and root.endswith("MacOS") and file == "luxmark"):
                    executable_path = os.path.join(root, file)
                    # Make executable
                    if system != "Windows":
                        os.chmod(executable_path, 0o755)
                    return executable_path
        
        logger.error(f"Could not find LuxMark executable in the extracted archive.")
        return None
        
    except Exception as e:
        logger.error(f"Error extracting LuxMark: {e}")
        return None

def run_luxmark_benchmark(executable, scene="microphone", mode="opencl", gpu_index=0):
    """
    Run LuxMark benchmark in command-line mode.
    
    Args:
        executable: Path to LuxMark executable
        scene: Scene to render (microphone, hotel, food)
        mode: Benchmark mode (opencl, cpu)
        gpu_index: GPU index to use (for multi-GPU systems)
        
    Returns:
        Dictionary with benchmark results
    """
    system = platform.system()
    results = {}
    
    # Map our scene names to LuxMark scene names
    scene_map = {
        "food": "FOOD",
        "hotel": "HALLBENCH",
        "microphone": "WALLPAPER"
    }
    
    # Map our mode to LuxMark mode
    mode_map = {
        "opencl": "BENCHMARK_OCL_GPU",
        "cpu": "BENCHMARK_OCL_CPU"
    }
    
    luxmark_scene = scene_map.get(scene, "WALLPAPER")
    luxmark_mode = mode_map.get(mode, "BENCHMARK_OCL_GPU")
    
    try:
        cmd = [
            executable,
            f"--scene={luxmark_scene}",
            f"--mode={luxmark_mode}",
            "--single-run"
        ]
        
        logger.info(f"Running LuxMark benchmark with scene: {scene}, mode: {mode}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception for non-zero exit
        )
        
        if result.returncode != 0:
            logger.warning(f"LuxMark exited with non-zero code: {result.returncode}")
        
        # Parse the output
        output = result.stdout + result.stderr
        logger.debug(f"LuxMark output: {output}")
        
        # Extract score using regex
        score_pattern = r"Score: (\d+(\.\d+)?)"
        score_match = re.search(score_pattern, output)
        
        if score_match:
            score = float(score_match.group(1))
            results["score"] = score
            
            # Extract device information
            device_pattern = r"Device: (.+)"
            device_match = re.search(device_pattern, output)
            if device_match:
                results["device"] = device_match.group(1)
                
            return results
        else:
            logger.error("Could not find score in LuxMark output")
            return {"error": "Score not found in output", "output": output}
            
    except Exception as e:
        logger.error(f"Error running LuxMark: {e}")
        return {"error": str(e)}

def luxmark_installation_instructions():
    """Return platform-specific installation instructions for LuxMark."""
    system = platform.system()
    
    if system == "Windows":
        return """
To install LuxMark on Windows:
1. Download LuxMark from https://github.com/LuxCoreRender/LuxMark/releases
2. Extract the ZIP file to a folder of your choice
3. Run luxmark.exe
"""
    elif system == "Darwin":
        return """
To install LuxMark on macOS:
1. Download LuxMark from https://github.com/LuxCoreRender/LuxMark/releases
2. Mount the DMG file
3. Copy LuxMark.app to your Applications folder
"""
    elif system == "Linux":
        return """
To install LuxMark on Linux:
1. Download LuxMark from https://github.com/LuxCoreRender/LuxMark/releases
2. Extract the archive to a folder of your choice
3. Make the luxmark file executable: chmod +x luxmark
"""
    else:
        return f"LuxMark installation instructions not available for {system}."

def run_all_scenes(executable, mode="opencl", gpu_index=0):
    """
    Run all LuxMark scenes and combine the results.
    
    Args:
        executable: Path to LuxMark executable
        mode: Benchmark mode (opencl, cpu)
        gpu_index: GPU index to use (for multi-GPU systems)
        
    Returns:
        Dictionary with results from all scenes
    """
    results = {}
    
    for scene in AVAILABLE_SCENES:
        try:
            logger.info(f"Running {scene} scene...")
            scene_results = run_luxmark_benchmark(executable, scene=scene, mode=mode, gpu_index=gpu_index)
            results[scene] = scene_results
        except Exception as e:
            logger.error(f"Error running {scene} scene: {e}")
            results[scene] = {"error": str(e)}
    
    # Calculate overall score (average of all scene scores)
    total_score = 0
    count = 0
    
    for scene_name, scene_result in results.items():
        if "score" in scene_result:
            total_score += scene_result["score"]
            count += 1
    
    if count > 0:
        results["overall_score"] = total_score / count
    
    return results

def main():
    """Main function to run the LuxMark benchmark."""
    parser = argparse.ArgumentParser(description="GPU Compute Performance Benchmark using LuxMark")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--scene", choices=AVAILABLE_SCENES + ["all"], default="all",
                      help="Specific scene to run (default: all)")
    parser.add_argument("--mode", choices=["opencl", "cuda", "cpu"], default="opencl",
                      help="Benchmark mode (default: opencl)")
    parser.add_argument("--gpu-index", type=int, default=0,
                      help="GPU index to use (default: 0)")
    parser.add_argument("--force-download", action="store_true", 
                      help="Force re-download of LuxMark even if cached version exists")
    args = parser.parse_args()
    
    # Initialize results
    results = {
        "error": None,
        "LuxMark": {
            "installed": False,
            "version": "4.0alpha0"  # Updated to match downloaded version
        }
    }
    
    try:
        # Check if LuxMark is installed
        executable = detect_luxmark()
        
        # If not installed or force download is requested, try to download it
        if executable is None or args.force_download:
            if args.force_download:
                logger.info("Force download requested. Clearing cache...")
                if CACHE_DIR.exists():
                    shutil.rmtree(CACHE_DIR)
                    
            logger.info("LuxMark not found. Attempting to download...")
            executable = download_luxmark()
        
        if executable is None:
            # If we still don't have it, return an error
            results["error"] = "LuxMark not found and could not be downloaded."
            results["installation_instructions"] = luxmark_installation_instructions()
        else:
            results["LuxMark"]["installed"] = True
            logger.info(f"Using LuxMark at {executable}")
            
            # Run the requested scene(s)
            if args.scene == "all":
                test_results = run_all_scenes(executable, mode=args.mode, gpu_index=args.gpu_index)
                results["scenes"] = test_results
                if "overall_score" in test_results:
                    results["LuxMarkScore"] = test_results["overall_score"]
            else:
                scene_results = run_luxmark_benchmark(executable, scene=args.scene, mode=args.mode, gpu_index=args.gpu_index)
                results["scenes"] = {args.scene: scene_results}
                if "score" in scene_results:
                    results["LuxMarkScore"] = scene_results["score"]
    
    except Exception as e:
        results["error"] = f"Error during benchmark: {str(e)}"
    
    # Output results
    if args.json:
        print(json.dumps(results))
    else:
        if results.get("error"):
            logger.error(f"Error: {results['error']}")
            if "installation_instructions" in results:
                logger.info(results["installation_instructions"])
        else:
            logger.info("\nLuxMark Benchmark Results:")
            if "LuxMarkScore" in results:
                logger.info(f"Overall Score: {results['LuxMarkScore']:.2f}")
            
            if "scenes" in results:
                for scene_name, scene_result in results["scenes"].items():
                    if scene_name != "overall_score" and "score" in scene_result:
                        logger.info(f"{scene_name} scene score: {scene_result['score']:.2f}")
    
    return 0 if not results.get("error") else 1

if __name__ == "__main__":
    sys.exit(main()) 