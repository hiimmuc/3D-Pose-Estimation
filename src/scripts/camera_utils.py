#!/usr/bin/env python
"""
Camera initialization and utilities.

This module handles all camera-related operations including RealSense and webcam
initialization, frame processing, and video capture. It provides a clean interface
for the main application to access camera data.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import mmengine
import numpy as np
from decoration import *
from mmengine.logging import print_log

from mmpose.structures import split_instances


def init_realsense():
    """Initialize RealSense camera and return pipeline and other objects.
    
    Returns:
        Tuple: (pipeline, profile, depth_scale, depth_intrinsics, color_intrinsics, align)
              or (None, None, None, None, None, None) if realsense not available
    """
    try:
        import pyrealsense2 as rs
        has_realsense = True
    except ImportError:
        print(f"{RED}✗ Error: pyrealsense2 library not found{END}")
        return None, None, None, None, None, None
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams with default settings
    # These will be overridden with config values later
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"{RED}✗ Error starting RealSense camera: {e}{END}")
        return None, None, None, None, None, None
    
    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # Get camera intrinsics
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    # Get extrinsics (transformation between depth and color)
    depth_to_color_extrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(
        profile.get_stream(rs.stream.color))
    
    # Create align object
    align = rs.align(rs.stream.color)
    
    print(f"{GREEN}✓{END} {BOLD}RealSense camera initialized{END} with depth scale {GREEN}{depth_scale}{END}")
    
    return pipeline, profile, depth_scale, depth_intrinsics, color_intrinsics, align


def process_camera(args, detector, pose_estimator, visualizer, 
                   is_realsense=False, realsense_objects=None,
                   process_frame_func=None, camera_matrix=None):
    """Process camera input (webcam or realsense).
    
    Args:
        args: Command line arguments
        detector: Detection model
        pose_estimator: Pose estimation model
        visualizer: Visualization object
        is_realsense: Whether to use RealSense camera
        realsense_objects: RealSense objects (pipeline, etc.)
        process_frame_func: Function to process a single frame
        
    Returns:
        None
    """


    # ANSI color codes for status messages
    camera_type = "RealSense" if is_realsense else "Webcam"
    
    print(f"{WHITE}{BOLD}STREAM CONSOLE{END}")
    
    print(f"\n{BLUE}┌─ {MAGENTA if is_realsense else CYAN}{camera_type}{BLUE} ─{'─' * (47 - len(camera_type))}┐{END}")
    
    
    if is_realsense:
        if not has_realsense():
            print(f"{BLUE}│{END} {RED}✗ Error: pyrealsense2 library not found{END}")
            print(f"{BLUE}└─{'─' * 50}┘{END}")
            return
        
        if realsense_objects is None or realsense_objects[0] is None:
            print(f"{BLUE}│{END} {RED}✗ Error: RealSense pipeline not initialized{END}")
            print(f"{BLUE}└─{'─' * 50}┘{END}")
            return
        
        pipeline, _, depth_scale, _, color_intrinsics, align = realsense_objects
        # NOTE: get camera matrix from color intrinsics
        camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        print(f"{BLUE}│{END} {YELLOW}Processing RealSense camera input...{END}")
        
    else:
        print(f"{BLUE}│{END} {YELLOW}Initializing webcam...{END}")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{BLUE}│{END} {RED}✗ Error: Could not open webcam{END}")
            print(f"{BLUE}└─{'─' * 50}┘{END}")
            return
        
        # NOTE: get camera matrix from webcam properties
        camera_matrix = camera_matrix
        
        print(f"{BLUE}│{END} {GREEN}✓ Webcam opened successfully{END}")
    
    # Initialize video writer
    video_writer = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_root, f"{camera_type.lower()}_{timestamp}.mp4")
    
    pred_instances_list = []
    frame_idx = 0
    
    try:
        while True:
            if is_realsense:
                # Wait for frames from RealSense
                frames = pipeline.wait_for_frames()
                
                # Align depth to color frame
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert frames to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Process frame
                img_vis, pred_instances = process_frame_func(
                    args, color_image, detector, pose_estimator, visualizer, 
                    depth_img=depth_image, depth_scale=depth_scale, 
                    show_interval=args.show_interval,
                    camera_matrix=camera_matrix
                )
                
            else:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame (no depth for standard webcam)
                img_vis, pred_instances = process_frame_func(
                    args, frame, detector, pose_estimator, visualizer, 
                    show_interval=args.show_interval,
                    camera_matrix=camera_matrix
                )
            
            frame_idx += 1
            
            # Initialize video writer if not done yet
            if video_writer is None and args.output_root:
                frame_height, frame_width = img_vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    output_file, fourcc, 25, (frame_width, frame_height)
                )
            
            # Save prediction results
            if args.save_predictions and pred_instances is not None:
                pred_instances_list.append(
                    dict(frame_id=frame_idx, instances=split_instances(pred_instances))
                )
            
            # Display result
            if args.show:
                cv2.imshow(f'Pose Estimation ({camera_type})', img_vis)
                
                # Press ESC to exit
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                # Sleep for display interval
                time.sleep(args.show_interval)
            
            # Write frame to output video
            if video_writer:
                video_writer.write(img_vis)
    
    except KeyboardInterrupt:
        print(f"\n{BLUE}│{END} {YELLOW}Camera capture interrupted by user{END}")
    except Exception as e:
        print(f"\n{BLUE}│{END} {RED}Error during camera capture: {e}{END}")
    finally:
        print(f"\n{BLUE}│{END} {GREEN}Stopping {camera_type.lower()} capture...{END}")
        
        # Clean up
        if is_realsense:
            if pipeline:
                pipeline.stop()
                print(f"{BLUE}│{END} {GREEN}✓ RealSense pipeline stopped{END}")
        else:
            cap.release()
            print(f"{BLUE}│{END} {GREEN}✓ Webcam released{END}")
        
        if video_writer:
            video_writer.release()
            print(f"{BLUE}│{END} {GREEN}✓ Video writer released{END}")
            
        cv2.destroyAllWindows()
        print(f"{BLUE}└─{'─' * 50}┘{END}")
        
        # Save predictions
        if args.save_predictions and pred_instances_list:
            from save_utils import save_predictions
            pred_save_path = os.path.join(
                args.output_root,
                f"{camera_type.lower()}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            save_predictions(pred_instances_list, pred_save_path, pose_estimator.dataset_meta)
        
        if args.output_root and video_writer:
            print_log(
                f'The output video has been saved at {output_file}',
                logger='current',
                level=logging.INFO
            )


def has_realsense() -> bool:
    """Check if pyrealsense2 is available.
    
    Returns:
        bool: True if pyrealsense2 is available, False otherwise
    """
    try:
        import pyrealsense2 as rs
        return True
    except ImportError:
        return False
