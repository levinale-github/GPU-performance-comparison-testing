#!/usr/bin/env python3
"""
GPU AI Inference Benchmark

This script measures the performance of the GPU for AI inference tasks
using a pre-trained ResNet-50 model and PyTorch.
"""

import os
import sys
import json
import time
import argparse
import logging
import platform
from pathlib import Path
import torch
import torchvision.models as models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AI-Inference-Benchmark")

# Add parent directory to path to import common utilities
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Try to import PyTorch and related libraries
try:
    import torchvision
    import numpy as np
    from PIL import Image
except ImportError as e:
    logger.error(f"Error importing required libraries: {e}")
    logger.error("\nThis benchmark requires PyTorch, torchvision, Pillow, and NumPy.")
    logger.error("Please install them with:")
    logger.error("  pip install torch torchvision pillow numpy")
    sys.exit(1)

def get_device_info():
    """Get information about the available GPU device"""
    # Check for Apple Silicon MPS first
    if torch.backends.mps.is_available():
        return {
            "device": "MPS GPU",
            "device_name": "Apple Silicon GPU",
            "mps_available": True,
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__
        }
    # Then check for CUDA
    elif torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        cuda_version = torch.version.cuda
        
        return {
            "device": "CUDA GPU",
            "device_name": device_name,
            "device_count": device_count,
            "cuda_version": cuda_version,
            "cuda_available": True,
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__
        }
    # Fall back to CPU
    else:
        return {
            "device": "CPU",
            "device_name": platform.processor(),
            "cuda_available": False,
            "mps_available": False,
            "pytorch_version": torch.__version__,
            "torchvision_version": torchvision.__version__
        }

def download_model(model_name="resnet50", pretrained=True):
    """Download a pre-trained model"""
    logger.info(f"Loading pre-trained {model_name} model...")
    
    try:
        # Load model with pretrained weights
        if model_name == "resnet50":
            # Use weights enum in newer versions of torchvision
            if hasattr(torchvision.models, 'ResNet50_Weights'):
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                model = torchvision.models.resnet50(weights=weights)
            else:
                model = torchvision.models.resnet50(pretrained=pretrained)
        elif model_name == "mobilenet_v2":
            if hasattr(torchvision.models, 'MobileNet_V2_Weights'):
                weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
                model = torchvision.models.mobilenet_v2(weights=weights)
            else:
                model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return None

def prepare_input(batch_size, device):
    """Prepare input tensor on the specified device."""
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    if device.type == 'mps':
        input_tensor = input_tensor.float()
    return input_tensor.to(device)

def load_model(device):
    """Load and move model to the specified device."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    if device.type == 'mps':
        # For MPS, we need to ensure everything is float32
        model = model.float()
        for param in model.parameters():
            param.data = param.data.float()
    model = model.to(device)
    model.eval()
    return model

def warmup(model, input_tensor, device):
    """Warm up the model."""
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_tensor)
        if device.type == 'mps':
            torch.mps.synchronize()

def benchmark_inference(model, input_tensor, num_iterations=100):
    """Benchmark inference performance."""
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(input_tensor)
        if input_tensor.device.type == 'mps':
            torch.mps.synchronize()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        images_per_second = (num_iterations * input_tensor.size(0)) / elapsed_time
        return images_per_second

def benchmark_batch_sizes(model, device, sizes=[1, 4, 8, 16, 32, 64]):
    """Benchmark performance with different batch sizes"""
    results = {}
    
    for batch_size in sizes:
        try:
            # Check if we have enough memory for this batch size
            if device.type == "cuda":
                # Get available memory in bytes
                try:
                    free_mem = torch.cuda.mem_get_info()[0]  # available memory
                    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
                    
                    # Estimate input size
                    input_size = batch_size * 3 * 224 * 224 * 4  # batch x channels x height x width x bytes
                    
                    # If we don't have enough memory, skip this batch size
                    if input_size + model_size > free_mem * 0.8:  # Use 80% as safety factor
                        logger.warning(f"Skipping batch size {batch_size} - insufficient memory")
                        continue
                except:
                    # If we can't check memory, we'll just try and catch OOM errors
                    pass
                
            logger.info(f"Testing batch size: {batch_size}")
            input_tensor = prepare_input(batch_size, device)
            
            # Shorter warmup and fewer runs for larger batches
            warmup_count = max(5, 20 // batch_size)
            run_count = max(10, 100 // batch_size)
            
            warmup(model, input_tensor, device)
            result = benchmark_inference(model, input_tensor)
            
            # Calculate images per second (batch_size * inferences_per_second)
            result["images_per_second"] = batch_size * result
            
            # Add to results dictionary
            results[f"batch_{batch_size}"] = result
            
            logger.info(f"  Time per inference: {result['time_per_inference']:.2f} ms")
            logger.info(f"  Images per second: {result['images_per_second']:.2f}")
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"Batch size {batch_size} caused CUDA out of memory error")
            break
        except Exception as e:
            logger.error(f"Error benchmarking batch size {batch_size}: {e}")
            break
    
    return results

def benchmark_models(device, models=["resnet50", "mobilenet_v2"]):
    """Benchmark different model architectures"""
    results = {}
    
    for model_name in models:
        try:
            logger.info(f"Benchmarking {model_name}...")
            model = download_model(model_name)
            
            if model is None:
                logger.error(f"Failed to load {model_name}. Skipping.")
                continue
            
            # Use smaller batch sizes for the benchmark run
            model_results = benchmark_batch_sizes(model, device, sizes=[1, 4, 8, 16])
            results[model_name] = model_results
            
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="GPU AI Inference Benchmark")
    parser.add_argument("--model", choices=["resnet50", "mobilenet_v2", "all"], default="resnet50",
                        help="Model architecture to benchmark")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Fixed batch size for simple benchmarks")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="Device to run benchmark on")
    args = parser.parse_args()
    
    logger.info("GPU AI Inference Benchmark")
    logger.info("=" * 40)
    
    # Set device based on args and availability
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    # Get device info
    device_info = get_device_info()
    device_name = device_info["device_name"]
    
    logger.info(f"Running on device: {device}")
    logger.info(f"Device name: {device_name}")
    
    if device.type == "cuda":
        logger.info(f"CUDA version: {device_info.get('cuda_version', 'Unknown')}")
    elif device.type == "mps":
        logger.info("Using Apple Metal Performance Shaders (MPS)")
    
    logger.info(f"PyTorch version: {device_info.get('pytorch_version', 'Unknown')}")
    
    # Determine which models to benchmark
    models_to_benchmark = []
    if args.model == "all":
        models_to_benchmark = ["resnet50", "mobilenet_v2"]
    else:
        models_to_benchmark = [args.model]
    
    # Run benchmarks
    try:
        results = {}
        
        if args.batch_size > 1:
            # Single model with specified batch size
            model_name = models_to_benchmark[0]
            model = download_model(model_name)
            
            if model is None:
                raise ValueError(f"Failed to load {model_name}")
            
            input_tensor = prepare_input(args.batch_size, device)
            warmup(model, input_tensor, device)
            
            logger.info(f"Benchmarking {model_name} with batch size {args.batch_size}...")
            inference_result = benchmark_inference(model, input_tensor)
            
            inference_result["images_per_second"] = args.batch_size * inference_result
            results[model_name] = {f"batch_{args.batch_size}": inference_result}
            
        else:
            # Full benchmark suite
            results = benchmark_models(device, models=models_to_benchmark)
        
        # Process results for output
        processed_results = {}
        
        # Find highest images per second across all tested configurations
        max_images_per_second = 0
        optimal_batch_size = 1
        optimal_model = models_to_benchmark[0]
        
        for model_name, model_results in results.items():
            for batch_key, batch_result in model_results.items():
                images_per_second = batch_result.get("images_per_second", 0)
                if images_per_second > max_images_per_second:
                    max_images_per_second = images_per_second
                    optimal_batch_size = int(batch_key.split("_")[1])
                    optimal_model = model_name
        
        # Create the final result
        ai_results = {
            "ResNet50Inference": int(max_images_per_second) if "resnet50" in results else None,
            "ResNet50InferenceUnit": "images/sec",
            "OptimalBatchSize": optimal_batch_size,
            "OptimalModel": optimal_model,
            "DetailedResults": results,
            "DeviceInfo": device_info
        }
        
        # If we have MobileNetV2 results, include those too
        if "mobilenet_v2" in results:
            # Find best MobileNetV2 performance
            mobilenet_max_ips = 0
            for batch_key, batch_result in results["mobilenet_v2"].items():
                images_per_second = batch_result.get("images_per_second", 0)
                if images_per_second > mobilenet_max_ips:
                    mobilenet_max_ips = images_per_second
            
            ai_results["MobileNetV2Inference"] = int(mobilenet_max_ips)
            ai_results["MobileNetV2InferenceUnit"] = "images/sec"
        
        # Output results
        if args.json:
            # Only print the JSON result
            print(json.dumps(ai_results, indent=2))
        else:
            # Print detailed results to console
            logger.info("\nSummary:")
            logger.info(f"Device: {device_info['device_name']}")
            
            if "ResNet50Inference" in ai_results and ai_results["ResNet50Inference"]:
                logger.info(f"ResNet-50 Performance: {ai_results['ResNet50Inference']} images/sec")
            
            if "MobileNetV2Inference" in ai_results and ai_results["MobileNetV2Inference"]:
                logger.info(f"MobileNet-V2 Performance: {ai_results['MobileNetV2Inference']} images/sec")
            
            logger.info(f"Optimal batch size: {optimal_batch_size}")
            
            # Print JSON at the end for the runner to parse
            print("\n" + json.dumps(ai_results))
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during benchmark: {e}")
        
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