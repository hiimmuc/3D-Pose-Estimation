#!/usr/bin/env python
"""
Utilities for saving pose estimation results.

This module handles saving prediction results to disk in various formats,
ensuring consistent output structure and naming conventions.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import json_tricks as json

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
END = '\033[0m'


def save_predictions(pred_instances_list, output_path, dataset_meta):
    """Save prediction results to a JSON file.
    
    Args:
        pred_instances_list: List of prediction instances to save
        output_path: Path to save the JSON file
        dataset_meta: Metadata about the dataset
        
    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(
            dict(
                meta_info=dataset_meta,
                instance_info=pred_instances_list
            ),
            f,
            indent='\t'
        )
    print(f"{GREEN}✓{END} Prediction results saved to {BOLD}{output_path}{END}")


def setup_output_paths(args):
    """Setup output paths for saving results.
    
    Args:
        args: Command line arguments
        
    Returns:
        None (modifies args in-place)
    """
    if args.output_root:
        os.makedirs(args.output_root, exist_ok=True)
        
        # Setup prediction save path if enabled
        if args.save_predictions:
            if args.input and args.input not in ['webcam', 'realsense']:
                input_filename = os.path.splitext(os.path.basename(args.input))[0]
                args.pred_save_path = f'{args.output_root}/results_{input_filename}.json'
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                args.pred_save_path = f'{args.output_root}/results_{timestamp}.json'
