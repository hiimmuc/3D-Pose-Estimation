#!/usr/bin/env python
"""
Image processing for pose estimation.

This module handles all image processing operations needed for pose estimation,
including detection, pose estimation, and visualization of results.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
from decoration import *
from mmdet.apis import inference_detector

from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances


def process_single_frame(args, img, detector, pose_estimator, visualizer=None,
                         depth_img=None, depth_scale=None, show_interval=0,
                         camera_matrix=None):
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
        camera_matrix: Camera intrinsics matrix for 3D deprojection
        
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
    
    # Calculate 3D coordinates
    if pred_instances is not None:
        calculate_3d_keypoints(args, pred_instances, depth_img, depth_scale, camera_matrix)
    
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
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Draw performance information
        draw_performance_info(img_vis, fps, timestamp)
    else:
        img_vis = None
    
    return img_vis, pred_instances


def calculate_3d_keypoints(args, pred_instances, depth_img, depth_scale, camera_matrix=None):
    """Calculate 3D coordinates for keypoints using depth information.
    
    Args:
        args: Command line arguments
        pred_instances: Prediction instances
        depth_img: Depth image
        depth_scale: Depth scale
        camera_matrix: Camera intrinsics matrix for 3D deprojection
        
    Returns:
        None (modifies pred_instances in-place)
    """
    # Extract keypoints for 3D coordinate calculation
    keypoints = pred_instances.keypoints
    keypoint_scores = pred_instances.keypoint_scores
    
    # Initialize 3D keypoints (camera coordinates)
    keypoints_3d = np.zeros((keypoints.shape[0], keypoints.shape[1], 3))
    
    # Initialize 3D keypoints (world coordinates after deprojection)
    keypoints_3d_world = np.zeros((keypoints.shape[0], keypoints.shape[1], 3))
    
    # For each person and each keypoint
    for person_idx, (person_keypoints, person_scores) in enumerate(zip(keypoints, keypoint_scores)):
        for kpt_idx, (kpt, score) in enumerate(zip(person_keypoints, person_scores)):
            if score > args.kpt_thr:
                x, y = int(kpt[0]), int(kpt[1])
                
                # Get depth value if depth image is available
                depth_value = np.nan
                if depth_img is not None and depth_scale is not None:
                    if (0 <= x < depth_img.shape[1] and 0 <= y < depth_img.shape[0]):
                        depth_value = depth_img[y, x] * depth_scale
                
                # Store 3D coordinates (x, y, depth)
                keypoints_3d[person_idx, kpt_idx] = [x, y, depth_value]
                
                # Calculate world coordinates if camera matrix is available
                if camera_matrix is not None:
                    # Use unit depth if depth is not available
                    actual_depth = 1.0 if np.isnan(depth_value) else depth_value
                    point_3d = camera_deprojection(
                        np.array([[x], [y], [actual_depth]], dtype=np.float32),
                        camera_matrix
                    )
                    # Store world coordinates
                    keypoints_3d_world[person_idx, kpt_idx] = [
                        point_3d[0, 0], point_3d[1, 0], point_3d[2, 0]
                    ]
            else:
                keypoints_3d[person_idx, kpt_idx] = [np.nan, np.nan, np.nan]
                keypoints_3d_world[person_idx, kpt_idx] = [np.nan, np.nan, np.nan]
    
    # Add 3D keypoints to pred_instances
    pred_instances.keypoints_3d = keypoints_3d
    pred_instances.keypoints_3d_world = keypoints_3d_world
    
    #* not implemented yet
    # Transform coordinates to be attached to a specific keypoint (e.g., nose)
    # pred_instances.keypoints_3d_transformed = keypoints_3d_transformed
    keypoints_3d_transformed = coordinates_transformation(
        keypoints_3d_world, kpt_idx=0  # Assuming 0 is the index of the nose keypoint
    )
    if keypoints_3d_transformed.shape == keypoints_3d_world.shape:
        pred_instances.keypoints_3d_transformed = keypoints_3d_transformed
    else:
        pred_instances.keypoints_3d_transformed = keypoints_3d_world
    
def camera_deprojection(keypoints_3d, camera_matrix):
    """Deproject 3D keypoints from camera coordinates to world coordinates.
    
    Args:
        keypoints_3d: 3D keypoints in camera coordinates (numpy array)
        camera_matrix: Camera intrinsics matrix (numpy array)
        
    Returns:
        Deprojected 3D keypoints in world coordinates (numpy array)
    """
    if camera_matrix is None:
        return keypoints_3d  # Return as is if no camera matrix provided
    # Invert the camera matrix
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    
    # Deproject each keypoint
    keypoints_world = np.dot(inv_camera_matrix, keypoints_3d)
    
    return keypoints_world

def coordinates_transformation(list_keypoints_3d_world, kpt_idx):
    """Given 3D keypoints (world), transform them to be attached to a specific key point, defined by kpt_idx. 
    Transformation using Homogeneous matrix as per the mathematical formulation in the document.
    
    Args:
        list_keypoints_3d_world: 3D keypoints in world coordinates (numpy array of shape [N, 3])
        kpt_idx: Index of the key point to attach coordinates to (e.g., nose)
        
    Returns:
        transformed_keypoints_3d: Transformed 3D keypoints in local frame (numpy array of shape [N, 3])
    """
    
    # Convert input to numpy array if not already
    keypoints_world = np.array(list_keypoints_3d_world).reshape(-1, 3)  # Ensure shape is (N, 3)
    
    
    # Validate input shape
    if keypoints_world.ndim != 2 or keypoints_world.shape[1] != 3:
        raise ValueError("Input keypoints must have shape (N, 3)")
    
    # Get the reference keypoint (the one we're transforming to)
    reference_point = keypoints_world[kpt_idx, :]  # [x, y, z] of reference keypoint
    
    # Create the transformation matrix T = [R | t]
    #                                     [0 | 1]
    # For coordinate frame transformation, we use:
    # - R = I (identity matrix, no rotation - assuming same orientation)
    # - t = -reference_point (translation to move reference point to origin)
    
    # Create 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, 3] = -reference_point  # Set translation part
    
    # Convert keypoints to homogeneous coordinates (add column of ones)
    N = keypoints_world.shape[0]
    keypoints_homo = np.hstack([keypoints_world, np.ones((N, 1))])  # Shape: (N, 4)
    
    # Apply transformation: X_local = T * X_world
    # We need to transpose for matrix multiplication: (4x4) * (4xN) = (4xN)
    transformed_homo = T @ keypoints_homo.T  # Shape: (4, N)
    
    # Transpose back and extract 3D coordinates (remove homogeneous coordinate)
    transformed_keypoints_3d = transformed_homo.T[:, :3]  # Shape: (N, 3)
    
    return np.expand_dims(transformed_keypoints_3d, axis=0)


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
    text_scale = getattr(args, 'text_scale', 0.4)
    text_thickness = getattr(args, 'text_thickness', 1)
    font = getattr(args, 'font', cv2.FONT_HERSHEY_SIMPLEX)
    kpt_thr = getattr(args, 'kpt_thr', 0.3)
    
    # Extract keypoints data
    keypoints = pred_instances.keypoints
    keypoint_scores = pred_instances.keypoint_scores
    keypoints_3d = getattr(pred_instances, 'keypoints_3d', None)
    keypoints_3d_world = getattr(pred_instances, 'keypoints_3d_world', None)
    keypoints_3d_transformed = getattr(pred_instances, 'keypoints_3d_transformed', None)
    
    
    # Draw 3D coordinates if available
    if keypoints_3d is not None:
        # For each person and keypoint
        for person_idx, (person_keypoints, person_scores, person_keypoints_3d) in enumerate(
                zip(keypoints, keypoint_scores, keypoints_3d)):
            
            # Get world coordinates if available
            person_world_coords = None
            if keypoints_3d_world is not None:
                person_world_coords = keypoints_3d_world[person_idx]
            
            kpt_body_coords = None
            if keypoints_3d_transformed is not None:
                kpt_body_coords = keypoints_3d_transformed[person_idx]
                
            
            for kpt_idx, (kpt, score, kpt_3d) in enumerate(
                    zip(person_keypoints, person_scores, person_keypoints_3d)):
                if score > kpt_thr:
                    x, y = int(kpt[0]), int(kpt[1])
                    depth = kpt_3d[2]
                    
                    # Get world coordinates if available
                    world_x, world_y, world_z = 0, 0, 0
                    if person_world_coords is not None:
                        world_x, world_y, world_z = person_world_coords[kpt_idx]
                        
                    transformed_x, transformed_y, transformed_z = 0, 0, 0
                    if kpt_body_coords is not None: 
                        transformed_x, transformed_y, transformed_z = kpt_body_coords[kpt_idx]
                    
                    # Format text with pixel and 3D coordinates
                    text_pixel = f"P{kpt_idx} ({x}, {y})"
                    text_world = f"[{world_x:.3f}, {world_y:.3f}, {world_z:.3f}]"
                    text_transformed = f"[{transformed_x:.3f}, {transformed_y:.3f},{transformed_z:.3f}]"
                    
                    # Calculate text size and position
                    (text_pixel_width, text_pixel_height), _ = cv2.getTextSize(
                        text_pixel, font, text_scale, text_thickness)

                    (text_world_width, _), _ = cv2.getTextSize(
                        text_world, font, text_scale, text_thickness)
                    
                    (text_transformed_width, _), _ = cv2.getTextSize(
                        text_transformed, font, text_scale, text_thickness)
                    
                    # Calculate the maximum width needed
                    max_width = max(text_pixel_width, text_world_width, text_transformed_width)
                    total_height = text_pixel_height * 3 + 4  # 3 lines of text plus padding
                    
                    # Draw semi-transparent background
                    # cv2.rectangle(
                    #     img_vis,
                    #     (x, y - total_height - 2),
                    #     (x + max_width, y),
                    #     (0, 0, 0), -1
                    # )
                    
                    # Apply alpha blending for transparency
                    alpha = 0.5
                    roi = img_vis[y - total_height - 2:y, x:x + max_width]
                    if roi.size > 0:  # Check if ROI is valid
                        cv2.addWeighted(roi, alpha, roi, 1 - alpha, 0, roi)
                    
                    # Draw text lines
                    # First line: pixel coordinates (white)
                    cv2.putText(
                        img_vis, text_pixel, (x, y - total_height + text_pixel_height),
                        font, text_scale, (255, 255, 255), text_thickness
                    )
                    
                    # Second line: camera coordinates (green)
      
                    cv2.putText(
                        img_vis, text_world, (x, y - 2),
                        font, text_scale, (0, 255, 0), text_thickness
                    )
                    
                    # Third line: transformed coordinates (blue)
                    cv2.putText(
                        img_vis, text_transformed, (x, y + text_pixel_height),
                        font, text_scale, (255, 0, 0), text_thickness
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


def load_camera_intrinsics(calib_file_path):
    """Load camera intrinsics matrix from calibration file.
    
    Args:
        calib_file_path: Path to the calibration JSON file
        
    Returns:
        Camera intrinsics matrix as numpy array
    """
    try:
        with open(calib_file_path, 'r') as f:
            calib_data = json.load(f)
            
        # Extract camera matrix from calibration data
        camera_matrix = np.array(calib_data['camera_matrix'], dtype=np.float32)
        print(success(f"Loaded camera intrinsics from {calib_file_path}"))
        return camera_matrix
    except Exception as e:
        print(error(f"Error loading camera intrinsics: {str(e)}"))
        # Return a default matrix as fallback
        return np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
