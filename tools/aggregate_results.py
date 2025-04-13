#!/usr/bin/env python3
"""
Aggregate benchmark results for the dashboard
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def aggregate_results():
    """Aggregate all benchmark results into a single file"""
    results_dir = Path('results')
    submissions_dir = results_dir / 'submissions'
    dashboard_dir = Path('dashboard')
    
    # Initialize aggregated data
    aggregated = {
        'last_updated': datetime.now().isoformat(),
        'gpus': []
    }
    
    # Process each submission
    for submission_file in submissions_dir.glob('*.json'):
        try:
            with open(submission_file, 'r') as f:
                data = json.load(f)
            
            # Extract GPU name from filename if not in the data
            filename = submission_file.stem
            parts = filename.split('_')
            
            # Extract relevant information based on different possible formats
            gpu_info = {
                'name': '',
                'vendor': '',
                'timestamp': '',
                'system': {
                    'cpu': {},
                    'memory': {},
                    'os': ''
                },
                'benchmarks': {}
            }
            
            # Extract metadata from different formats
            if 'metadata' in data:
                metadata = data['metadata']
                
                # Handle the new format with system key
                if 'system' in metadata:
                    system = metadata['system']
                    gpu_info['name'] = system.get('gpu', {}).get('GPU', parts[-1] if len(parts) > 0 else 'Unknown')
                    gpu_info['vendor'] = system.get('gpu', {}).get('GPUVendor', parts[-2] if len(parts) > 1 else 'Unknown')
                    gpu_info['timestamp'] = metadata.get('timestamp', datetime.now().isoformat())
                    gpu_info['system']['cpu'] = system.get('cpu', {})
                    gpu_info['system']['memory'] = system.get('memory', {})
                    gpu_info['system']['os'] = system.get('os', 'Unknown')
                    
                    # Preserve all GPU information fields
                    if 'gpu' in system:
                        # Copy all GPU fields as-is
                        gpu_info['system']['gpu'] = system['gpu']
                    
                    gpu_info['contributor'] = metadata.get('contributor', 'Unknown')
                    gpu_info['notes'] = metadata.get('notes', '')
                # Handle the old format without system key
                else:
                    gpu_info['name'] = metadata.get('GPU', parts[-1] if len(parts) > 0 else 'Unknown')
                    gpu_info['vendor'] = metadata.get('GPUVendor', parts[-2] if len(parts) > 1 else 'Unknown')
                    gpu_info['timestamp'] = metadata.get('Timestamp', datetime.now().isoformat())
                    gpu_info['system']['cpu'] = {'model': metadata.get('CPUModel', 'Unknown')}
                    gpu_info['system']['memory'] = {'total': metadata.get('RAMSize', 'Unknown')}
                    gpu_info['system']['os'] = metadata.get('OS', 'Unknown')
                    gpu_info['contributor'] = metadata.get('Contributor', 'Unknown')
                    gpu_info['notes'] = metadata.get('Notes', '')
            
            # Extract benchmark results
            if 'results' in data:
                gpu_info['benchmarks'] = data['results']
            
            aggregated['gpus'].append(gpu_info)
            print(f"Processed: {submission_file}")
        except Exception as e:
            print(f"Error processing {submission_file}: {e}")
    
    # Save aggregated results to results directory
    output_path = results_dir / 'aggregated_results.json'
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    # Also save to dashboard directory
    dashboard_output_path = dashboard_dir / 'aggregated_results.json'
    with open(dashboard_output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Aggregated results saved to: {output_path} and {dashboard_output_path}")

if __name__ == "__main__":
    aggregate_results() 