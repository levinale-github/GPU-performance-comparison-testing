#!/usr/bin/env python3
"""
Aggregate benchmark results for the dashboard
"""

import json
from pathlib import Path
from datetime import datetime

def aggregate_results():
    """Aggregate all benchmark results into a single file"""
    results_dir = Path('results')
    submissions_dir = results_dir / 'submissions'
    
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
                
            # Extract relevant information
            gpu_info = {
                'name': data['metadata']['system'].get('nvidia_gpus', [{}])[0].get('name', 'Unknown'),
                'vendor': data['metadata']['system'].get('nvidia_gpus', [{}])[0].get('vendor', 'Unknown'),
                'timestamp': data['metadata']['timestamp'],
                'system': {
                    'cpu': data['metadata']['system'].get('cpu', {}),
                    'memory': data['metadata']['system'].get('memory', {}),
                    'os': data['metadata']['system'].get('os', 'Unknown')
                },
                'benchmarks': data['results']
            }
            
            aggregated['gpus'].append(gpu_info)
        except Exception as e:
            print(f"Error processing {submission_file}: {e}")
    
    # Save aggregated results
    output_path = results_dir / 'aggregated_results.json'
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Aggregated results saved to: {output_path}")

if __name__ == "__main__":
    aggregate_results() 