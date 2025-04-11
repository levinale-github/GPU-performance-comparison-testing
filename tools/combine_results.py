#!/usr/bin/env python3
"""
GPU Benchmark Results Aggregator

This script combines all individual benchmark submissions from results/submissions/
into a single aggregated results file used by the dashboard.
"""

import os
import sys
import json
import glob
import datetime
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SUBMISSIONS_DIR = REPO_ROOT / "results" / "submissions"
AGGREGATED_FILE = REPO_ROOT / "results" / "aggregated_results.json"

def clean_string_for_id(s):
    """Convert a GPU name to a valid ID for use in the aggregated file"""
    # Convert to lowercase, replace spaces with underscores
    s = s.lower().replace(' ', '_')
    # Remove parentheses, brackets, etc.
    s = s.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    # Remove any special characters
    s = ''.join(c for c in s if c.isalnum() or c == '_')
    return s

def load_submission(file_path):
    """Load and validate a submission file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Basic validation
        if not isinstance(data, dict):
            print(f"Error: {file_path} does not contain a valid JSON object.")
            return None
        
        if 'metadata' not in data or 'results' not in data:
            print(f"Error: {file_path} is missing required fields (metadata or results).")
            return None
        
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_gpu_summary(submission, file_path):
    """
    Extract the key performance metrics from a submission into a summary
    for the aggregated results file.
    """
    metadata = submission.get('metadata', {})
    results = submission.get('results', {})
    
    # Get GPU name and vendor
    gpu_name = metadata.get('GPU', 'Unknown GPU')
    vendor = metadata.get('GPUVendor', 'Unknown')
    
    # Create a unique ID for this GPU
    gpu_id = f"{clean_string_for_id(vendor)}_{clean_string_for_id(gpu_name)}"
    
    # Extract memory info
    memory = metadata.get('VRAM', 'Unknown')
    
    # Extract memory bandwidth (if available)
    memory_bandwidth = None
    if 'memory' in results and 'Bandwidth' in results['memory']:
        memory_bandwidth = results['memory']['Bandwidth']
    
    # Extract compute performance metrics
    fp32_performance = None
    fp16_performance = None
    fp64_performance = None
    fp64_fp32_ratio = None
    
    if 'compute' in results:
        compute = results['compute']
        if 'FP32' in compute:
            fp32_performance = compute['FP32']
        if 'FP16' in compute:
            fp16_performance = compute['FP16']
        if 'FP64' in compute:
            fp64_performance = compute['FP64']
        if 'FP64_FP32_Ratio' in compute:
            fp64_fp32_ratio = compute['FP64_FP32_Ratio']
    
    # Extract graphics score
    graphics_score = None
    if 'graphics' in results and 'GLMark2Score' in results['graphics']:
        graphics_score = results['graphics']['GLMark2Score']
    
    # Extract AI inference performance
    ai_inference = None
    if 'ai' in results and 'ResNet50Inference' in results['ai']:
        ai_inference = results['ai']['ResNet50Inference']
    
    # Extract cost information
    retail_price = None
    cloud_cost_per_hour = None
    break_even_hours = None
    
    if 'Cost' in metadata:
        cost = metadata['Cost']
        if 'RetailPrice' in cost:
            retail_price = cost['RetailPrice']
        if 'CloudCostPerHour' in cost:
            cloud_cost_per_hour = cost['CloudCostPerHour']
            
            # Calculate break-even hours if both retail price and cloud cost are available
            if retail_price is not None and cloud_cost_per_hour is not None and cloud_cost_per_hour > 0:
                break_even_hours = retail_price / cloud_cost_per_hour
    
    # Get contributor and timestamp
    contributor = metadata.get('Contributor', 'Anonymous')
    timestamp = metadata.get('Timestamp', datetime.datetime.now().isoformat())
    
    # Create a relative path to the detailed results
    relative_path = str(Path(file_path).relative_to(REPO_ROOT))
    
    # Create a summary object for the aggregated results
    summary = {
        "id": gpu_id,
        "name": gpu_name,
        "vendor": vendor,
        "memory": memory,
        "memory_bandwidth": memory_bandwidth,
        "fp32_performance": fp32_performance,
        "fp16_performance": fp16_performance,
        "fp64_performance": fp64_performance,
        "fp64_fp32_ratio": fp64_fp32_ratio,
        "graphics_score": graphics_score,
        "ai_inference": ai_inference,
        "retail_price": retail_price,
        "cloud_cost_per_hour": cloud_cost_per_hour,
        "break_even_hours": break_even_hours,
        "contributor": contributor,
        "submission_date": timestamp,
        "detailed_results": relative_path
    }
    
    # Filter out None values for cleaner JSON
    return {k: v for k, v in summary.items() if v is not None}

def update_aggregated_results():
    """
    Process all submission files and update the aggregated results JSON file.
    """
    # Ensure the submissions directory exists
    if not SUBMISSIONS_DIR.exists():
        print(f"Error: Submissions directory {SUBMISSIONS_DIR} does not exist.")
        return False
    
    # Get all JSON files in the submissions directory
    submission_files = glob.glob(str(SUBMISSIONS_DIR / "*.json"))
    
    if not submission_files:
        print("No submission files found.")
        return False
    
    # Load existing aggregated file if it exists
    if AGGREGATED_FILE.exists():
        try:
            with open(AGGREGATED_FILE, 'r') as f:
                aggregated = json.load(f)
                existing_gpus = {gpu['id']: gpu for gpu in aggregated.get('gpus', [])}
        except Exception as e:
            print(f"Error loading existing aggregated results: {e}")
            print("Starting with a fresh aggregated file.")
            existing_gpus = {}
            aggregated = {
                "gpus": [],
                "metadata": {
                    "schema_version": "1.0",
                    "unit_definitions": {
                        "memory_bandwidth": "GB/s",
                        "fp32_performance": "TFLOPS",
                        "fp16_performance": "TFLOPS",
                        "fp64_performance": "TFLOPS",
                        "ai_inference": "images/sec with ResNet50",
                        "retail_price": "USD",
                        "cloud_cost_per_hour": "USD/hour"
                    }
                }
            }
    else:
        # Create a new aggregated file
        existing_gpus = {}
        aggregated = {
            "gpus": [],
            "metadata": {
                "schema_version": "1.0",
                "unit_definitions": {
                    "memory_bandwidth": "GB/s",
                    "fp32_performance": "TFLOPS",
                    "fp16_performance": "TFLOPS",
                    "fp64_performance": "TFLOPS",
                    "ai_inference": "images/sec with ResNet50",
                    "retail_price": "USD",
                    "cloud_cost_per_hour": "USD/hour"
                }
            }
        }
    
    # Process each submission
    processed_count = 0
    updated_count = 0
    error_count = 0
    
    for file_path in submission_files:
        submission = load_submission(file_path)
        if submission is None:
            error_count += 1
            continue
        
        # Extract summary data
        summary = extract_gpu_summary(submission, file_path)
        
        # Check if this GPU already exists in the aggregated results
        if summary['id'] in existing_gpus:
            # Update existing entry with new data
            existing_gpus[summary['id']].update(summary)
            updated_count += 1
        else:
            # Add as a new entry
            existing_gpus[summary['id']] = summary
            processed_count += 1
    
    # Update the aggregated data
    aggregated['gpus'] = list(existing_gpus.values())
    aggregated['last_updated'] = datetime.datetime.now().isoformat()
    
    # Write the aggregated file
    try:
        with open(AGGREGATED_FILE, 'w') as f:
            json.dump(aggregated, f, indent=2)
        
        print(f"Aggregated results updated successfully:")
        print(f"  - {processed_count} new entries added")
        print(f"  - {updated_count} existing entries updated")
        print(f"  - {error_count} files skipped due to errors")
        print(f"Results written to {AGGREGATED_FILE}")
        return True
    except Exception as e:
        print(f"Error writing aggregated results: {e}")
        return False

def main():
    """Main function"""
    print("GPU Benchmark Results Aggregator")
    print("=" * 30)
    
    result = update_aggregated_results()
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 