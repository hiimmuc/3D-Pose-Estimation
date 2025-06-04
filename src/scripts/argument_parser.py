#!/usr/bin/env python
"""
Command-line argument parser for pose estimation.

This module handles command-line argument parsing and configuration merging,
providing a clean interface for accessing user-specified options.
"""

import argparse
import os
from typing import Any, Dict, Optional

import yaml
from decoration import *


def parse_arguments():
    """Parse command line arguments for pose estimation.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Pose Estimation with 3D coordinates visualization')
    
    # Configuration files
    parser.add_argument('--config', 
                        type=str, 
                        default=None,
                        help='Path to config file (.yaml)')
    parser.add_argument('--det-config', 
                        type=str, 
                        default=None,
                        help='Config file for detection')
    parser.add_argument('--det-checkpoint', 
                        type=str, 
                        default=None,
                        help='Checkpoint file for detection')
    parser.add_argument('--pose-config', 
                        type=str, 
                        default=None,
                        help='Config file for pose')
    parser.add_argument('--pose-checkpoint', 
                        type=str, 
                        default=None,
                        help='Checkpoint file for pose')
    
    # Input/Output options
    parser.add_argument('--input', 
                        type=str, 
                        default='',
                        help='Path to input image/video, or use "webcam" or "realsense"')
    parser.add_argument('--show',
                        action='store_true',
                        default=False,
                        help='whether to show img')
    parser.add_argument('--output-root',
                        type=str,
                        default='',
                        help='root of the output img file.')
    parser.add_argument('--save-predictions',
                        action='store_true',
                        default=False,
                        help='whether to save predicted results')
    parser.add_argument('--show-interval',
                        type=float,
                        default=0,
                        help='sleep seconds per frame')
    
    # Model parameters
    parser.add_argument('--device', 
                        default='cuda:0', 
                        help='Device used for inference')
    parser.add_argument('--det-cat-id',
                        type=int,
                        default=0,
                        help='Category id for bounding box detection model')
    parser.add_argument('--bbox-thr',
                        type=float,
                        default=0.3,
                        help='Bounding box score threshold')
    parser.add_argument('--nms-thr',
                        type=float,
                        default=0.3,
                        help='NMS threshold')
    parser.add_argument('--kpt-thr',
                        type=float,
                        default=0.3,
                        help='Keypoint score threshold')
    
    # Visualization options
    parser.add_argument('--draw-heatmap', 
                        action='store_true', 
                        default=False,
                        help='Draw heatmap of keypoints')
    parser.add_argument('--draw-bbox', 
                        action='store_true', 
                        default=False,
                        help='Draw bounding boxes')
    parser.add_argument('--show-kpt-idx', 
                        action='store_true', 
                        default=False,
                        help='Show keypoint indices')
    parser.add_argument('--skeleton-style', 
                        default='mmpose',
                        choices=['mmpose', 'openpose'],
                        help='Skeleton style')
    parser.add_argument('--radius',
                        type=int,
                        default=3,
                        help='Keypoint radius for visualization')
    parser.add_argument('--thickness',
                        type=int,
                        default=1,
                        help='Link thickness for visualization')
    
    # Calibration file
    parser.add_argument('--calibration-file',
                        default='calibration_data/calib_hd_pro_webcam_c920__046d_082d__1920.json',
                        help='camera calibration file for 3D coordinate deprojection')
    
    return parser.parse_args()


def merge_configs(args, config: Dict[str, Any]):
    """Merge command line arguments with configuration file.
    
    Command line arguments take precedence over configuration file.
    
    Args:
        args: Command line arguments
        config: Configuration dictionary
        
    Returns:
        None (modifies args in-place)
    """
    # Detection parameters
    if not args.det_config and 'detection' in config:
        args.det_config = config['detection']['config']
    if not args.det_checkpoint and 'detection' in config:
        args.det_checkpoint = config['detection']['checkpoint']
    if not args.device or args.device == 'cuda:0' and 'detection' in config:
        args.device = config['detection']['device'] 
    if not args.det_cat_id and 'detection' in config:
        args.det_cat_id = config['detection']['cat_id']
    if not args.bbox_thr and 'detection' in config:
        args.bbox_thr = config['detection']['bbox_thr']
    if not args.nms_thr and 'detection' in config:
        args.nms_thr = config['detection']['nms_thr']
        
    # Pose parameters
    if not args.pose_config and 'pose' in config:
        args.pose_config = config['pose']['config']
    if not args.pose_checkpoint and 'pose' in config:
        args.pose_checkpoint = config['pose']['checkpoint']
    if not args.kpt_thr and 'pose' in config:
        args.kpt_thr = config['pose']['kpt_thr']
    if not args.radius and 'pose' in config:
        args.radius = config['pose']['radius']
    if not args.thickness and 'pose' in config:
        args.thickness = config['pose']['thickness']
        
    # Visualization parameters
    if not args.draw_bbox and 'visualization' in config:
        args.draw_bbox = config['visualization']['draw_bbox']
    if not args.draw_heatmap and 'visualization' in config:
        args.draw_heatmap = config['visualization']['draw_heatmap']
    if not args.show_kpt_idx and 'visualization' in config:
        args.show_kpt_idx = config['visualization']['show_kpt_idx']
    if args.skeleton_style == 'mmpose' and 'visualization' in config:
        args.skeleton_style = config['visualization']['skeleton_style']
        
    # IO parameters
    if not args.show and 'io' in config:
        args.show = config['io']['show']
    if not args.output_root and 'io' in config:
        args.output_root = config['io']['output_root']
    if not args.save_predictions and 'io' in config:
        args.save_predictions = config['io']['save_predictions']
    if not args.show_interval and 'io' in config:
        args.show_interval = config['io']['show_interval']
