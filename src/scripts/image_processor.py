#!/usr/bin/env python
"""
Image processing for pose estimation.

This module handles all image processing operations needed for pose estimation,
including detection, pose estimation, and visualization of results.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from mmdet.apis import inference_detector

from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances

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


def process_one_image(args, img, detector, pose_estimator, visualizer=None,
                      depth_img=None, depth_scale=None, show_interval=0):
    """Process one image frame with enhanced visualization and 3D coordinates.
    
    Args:
        args: Command line arguments
        img: Input image (path string or numpy array)
        detector: Detection model (MMDetection model)
        pose_estimator: Pose estimation model (MMPose model)
        visualizer: Visualization object from MMPose
        depth_img: Depth image as numpy array (optional for 3D coordinates)
        depth_scale: Depth scale factor (optional)
        show_interval: Sleep time between frames visualization (optional)
        
    Returns:
        Tuple: (img_vis, pred_instances) - Visualized image and prediction instances
    """
    
    start_time = time.time()
    
    # Predict bboxes
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[
        np.logical_and(
            pred_instance.labels == args.det_cat_id,
            pred_instance.scores > args.bbox_thr
        )
    ]
    
    # Non-maximum suppression
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
    
    # Predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)
    
    # Get the pred_instances for later processing
    pred_instances = data_samples.get('pred_instances', None)
    
    # Calculate 3D coordinates if depth information is available
    if pred_instances is not None and depth_img is not None and depth_scale is not None:
        calculate_3d_keypoints(args, pred_instances, depth_img, depth_scale)
    
    # Convert image if needed
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)
    
    # Visualize results
    if visualizer is not None:
        # Add visualization parameters from args
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=False,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr
        )
        
        # Get the visualized image
        img_vis = visualizer.get_image()
        img_vis = mmcv.rgb2bgr(img_vis)
        
        # Draw 3D coordinates or 2D coordinates
        draw_keypoint_annotations(img_vis, pred_instances, args)
        
        # Calculate and display FPS and timestamp
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Draw performance information
        draw_performance_info(img_vis, fps, timestamp)
    else:
        img_vis = None
    
    return img_vis, pred_instances


def calculate_3d_keypoints(args, pred_instances, depth_img, depth_scale):
    """Calculate 3D coordinates for keypoints using depth information.
    
    Args:
        args: Command line arguments
        pred_instances: Prediction instances
        depth_img: Depth image
        depth_scale: Depth scale
        
    Returns:
        None (modifies pred_instances in-place)
    """
    # Extract keypoints for 3D coordinate calculation
    keypoints = pred_instances.keypoints
    keypoint_scores = pred_instances.keypoint_scores
    
    # Initialize 3D keypoints
    keypoints_3d = np.zeros((keypoints.shape[0], keypoints.shape[1], 3))
    
    # For each person and each keypoint
    for person_idx, (person_keypoints, person_scores) in enumerate(zip(keypoints, keypoint_scores)):
        for kpt_idx, (kpt, score) in enumerate(zip(person_keypoints, person_scores)):
            if score > args.kpt_thr:
                x, y = int(kpt[0]), int(kpt[1])
                
                # Ensure point is within frame bounds
                if (0 <= x < depth_img.shape[1] and 
                    0 <= y < depth_img.shape[0]):
                    
                    # Get depth at keypoint location (in meters)
                    depth_value = depth_img[y, x] * depth_scale
                    
                    # Store 3D coordinates (x, y, depth)
                    keypoints_3d[person_idx, kpt_idx] = [x, y, depth_value]
                else:
                    keypoints_3d[person_idx, kpt_idx] = [x, y, 0]
            else:
                keypoints_3d[person_idx, kpt_idx] = [0, 0, 0]
    
    # Add 3D keypoints to pred_instances
    pred_instances.keypoints_3d = keypoints_3d


def draw_keypoint_annotations(img_vis, pred_instances, args):
    """Draw keypoint annotations (3D coordinates or indices) on the image.
    
    Args:
        img_vis: Visualized image (numpy array)
        pred_instances: Prediction instances from pose estimator
        args: Command line arguments with visualization settings
        
    Returns:
        None (modifies img_vis in-place)
    """
    if pred_instances is None or img_vis is None:
        return
    
    # Get configuration values or use defaults
    text_scale = getattr(args, 'text_scale', 0.5)
    text_thickness = getattr(args, 'text_thickness', 1)
    font = getattr(args, 'font', cv2.FONT_HERSHEY_SIMPLEX)
    kpt_thr = getattr(args, 'kpt_thr', 0.3)
    
    # Extract keypoints data
    keypoints = pred_instances.keypoints
    keypoint_scores = pred_instances.keypoint_scores
    keypoints_3d = getattr(pred_instances, 'keypoints_3d', None)
    
    # Draw 3D coordinates if available
    if keypoints_3d is not None:
        # For each person and keypoint
        for person_idx, (person_keypoints, person_scores, person_keypoints_3d) in enumerate(
                zip(keypoints, keypoint_scores, keypoints_3d)):
            for kpt_idx, (kpt, score, kpt_3d) in enumerate(
                    zip(person_keypoints, person_scores, person_keypoints_3d)):
                if score > kpt_thr:
                    x, y = int(kpt[0]), int(kpt[1])
                    depth = kpt_3d[2]
                    
                    # Format text with 3D coordinates
                    text = f"{kpt_idx}: ({x},{y},{depth:.3f}m)"
                    
                    # Calculate text background for better readability
                    
                    # Calculate text size and position
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, font, text_scale, text_thickness)
                    
                    # Draw semi-transparent background
                    cv2.rectangle(
                        img_vis,
                        (x, y - text_height - 2),
                        (x + text_width, y),
                        (0, 0, 0), -1
                    )
                    
                    # Apply alpha blending for transparency
                    alpha = 0.5
                    roi = img_vis[y - text_height - 2:y, x:x + text_width]
                    if roi.size > 0:  # Check if ROI is valid
                        cv2.addWeighted(roi, alpha, roi, 1 - alpha, 0, roi)
                    
                    # Draw text
                    cv2.putText(
                        img_vis, text, (x, y - 2),
                        font, text_scale, (255, 255, 255), text_thickness
                    )
    else:
        # Draw keypoints without depth information (2D only)
        for person_idx, (person_keypoints, person_scores) in enumerate(
                zip(keypoints, keypoint_scores)):
            for kpt_idx, (kpt, score) in enumerate(zip(person_keypoints, person_scores)):
                if score > args.kpt_thr:
                    x, y = int(kpt[0]), int(kpt[1])
                    
                    # Format text with 2D coordinates and zero depth
                    text = f"{kpt_idx}: ({x},{y})"
                    
                    # Draw text with small font
                    cv2.putText(
                        img_vis, text, (x, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
                    )


def draw_performance_info(img_vis, fps, timestamp):
    """Draw performance metrics (FPS, timestamp) on the image.
    
    Args:
        img_vis: Visualized image (numpy array)
        fps: Frames per second (float)
        timestamp: Current timestamp (string)
        
    Returns:
        None (modifies img_vis in-place)
    """
    if img_vis is None:
        return
        
    height, width = img_vis.shape[:2]
    
    # Create a semi-transparent overlay for performance display
    overlay = img_vis.copy()
    info_width = 300
    info_height = 40
    
    # Ensure overlay doesn't exceed image dimensions
    info_width = min(info_width, width - 10)
    
    # Draw background rectangle
    cv2.rectangle(
        img=overlay, 
        pt1=(5, 5), 
        pt2=(5 + info_width, 5 + info_height), 
        color=(0, 0, 0), 
        thickness=-1
    )
    cv2.addWeighted(src1=overlay, alpha=0.6, src2=img_vis, beta=0.4, gamma=0, dst=img_vis)
    
    # Draw FPS with adaptive color (green to red based on FPS)
    # Higher FPS = green, lower FPS = red
    fps_color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 5 else (0, 0, 255)
    
    # Draw FPS with shadow for better visibility
    cv2.putText(
        img_vis, f"FPS: {fps:.1f}", 
        (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2  # Shadow
    )
    cv2.putText(
        img_vis, f"FPS: {fps:.1f}", 
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2
    )
    
    cv2.putText(
        img_vis, f"Time: {timestamp}", 
        (131, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
    )
    cv2.putText(
        img_vis, f"Time: {timestamp}", 
        (130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
