#!/usr/bin/env python
"""
Main script for Pose Estimation with 3D coordinates visualization using MMPose and MMDetection.
This script serves as the entry point for the pose estimation system, integrating all the modular components.
"""

import mimetypes
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmengine.logging import print_log

# Import MMPose modules
from mmpose.apis import init_model as init_pose_estimator
from mmpose.registry import VISUALIZERS
from mmpose.utils import adapt_mmdet_pipeline

# Import MMDetection modules
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# Import custom modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from argument_parser import merge_configs, parse_arguments
from camera_utils import has_realsense, init_realsense, process_camera
from config_utils import ensure_models_exist, load_config, print_model_info
from image_processor import process_one_image
from save_utils import save_predictions, setup_output_paths

# Filter out warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ANSI color codes for console output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
BOLD = '\033[1m'
END = '\033[0m'

def process_image(args, img_path, detector, pose_estimator, visualizer):
    """Process a single image file.
    
    Args:
        args: Command line arguments
        img_path: Path to the image file
        detector: Detection model
        pose_estimator: Pose estimation model
        visualizer: Visualization object
        
    Returns:
        pred_instances: The prediction instances or None if processing failed
    """
    try:
        # Load image
        img = mmcv.imread(img_path, channel_order='rgb')
        if img is None:
            print(f"{RED}✗ Error: Failed to load image: {img_path}{END}")
            return None
            
        # Process image
        print(f"{BLUE}Processing image: {os.path.basename(img_path)}{END}")
        img_vis, pred_instances = process_one_image(
            args, img, detector, pose_estimator, visualizer
        )
        
        # Show result
        if args.show:
            cv2.imshow('Pose Estimation Result', img_vis)
            print(f"{YELLOW}Displaying result. Press any key to continue...{END}")
            cv2.waitKey(0)
        
        # Save result
        if args.output_root:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            output_file = os.path.join(
                args.output_root, 
                f"pose_estimation_{file_name}_{timestamp}.jpg"
            )
            
            # Save image
            mmcv.imwrite(img_vis, output_file)
            print(f"{GREEN}✓{END} Result saved to {BOLD}{output_file}{END}")
            
            # Save predictions if requested
            if args.save_predictions and pred_instances is not None:
                pred_instances_list = [pred_instances.to_dict()]
                pred_save_path = os.path.join(
                    args.output_root, 
                    f"predictions_{file_name}_{timestamp}.json"
                )
                save_predictions(pred_instances_list, pred_save_path, pose_estimator.dataset_meta)
        
        return pred_instances
        
    except Exception as e:
        print(f"{RED}✗ Error processing image: {e}{END}")
        return None


def process_video(args, video_path, detector, pose_estimator, visualizer):
    """Process video file for pose estimation.
    
    Args:
        args: Command line arguments
        video_path: Path to the video file
        detector: Detection model
        pose_estimator: Pose estimation model
        visualizer: Visualization object
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{RED}✗ Error: Failed to open video: {video_path}{END}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cv2.__version__ >= "3.0" else 0
        
        # Initialize output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.splitext(os.path.basename(video_path))[0]
        output_file = None
        
        # Initialize video writer
        video_writer = None
        if args.output_root:
            mmcv.mkdir_or_exist(args.output_root)
            output_file = os.path.join(args.output_root, f"{filename}_{timestamp}.mp4")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_file, fourcc, fps, (frame_width, frame_height))
        
        # Initialize prediction collection
        pred_instances_list = []
        frame_idx = 0
        processing_times = []
        
        # Print video information
        print(f"{BLUE}┌──────────────── Processing Video ─────────────────┐{END}")
        print(f"{BLUE}│{END} {YELLOW}Video:{END} {os.path.basename(video_path)}")
        print(f"{BLUE}│{END} {YELLOW}Resolution:{END} {frame_width}x{frame_height}, {YELLOW}FPS:{END} {fps:.2f}")
        if total_frames > 0:
            print(f"{BLUE}│{END} {YELLOW}Total frames:{END} {total_frames}")
        
        # Process frames
        try:
            while cap.isOpened():
                # Read frame
                success, frame = cap.read()
                if not success:
                    break
                    
                # Process frame
                start_time = time.time()
                frame_vis, pred_instances = process_one_image(
                    args, frame, detector, pose_estimator, visualizer
                )
                process_time = time.time() - start_time
                processing_times.append(process_time)
                
                # Display progress
                if frame_idx % 10 == 0:
                    if total_frames > 0:
                        progress = frame_idx / total_frames * 100
                        print(f"{BLUE}│{END} Frame {frame_idx}/{total_frames} ({progress:.1f}%)", end='')
                    else:
                        print(f"{BLUE}│{END} Frame {frame_idx}", end='')
                        
                    if processing_times:
                        avg_time = sum(processing_times[-10:]) / min(10, len(processing_times[-10:]))
                        print(f", processing at {1/avg_time:.1f} FPS")
                    else:
                        print()
                
                # Save to video
                if video_writer:
                    video_writer.write(frame_vis)
                
                # Display frame
                if args.show:
                    cv2.imshow('Pose Estimation', frame_vis)
                    wait_time = max(1, int(1000 / fps * args.show_interval)) if fps > 0 else 1
                    key = cv2.waitKey(wait_time)
                    
                    # Exit on ESC key
                    if key & 0xFF == 27:  # ESC
                        print(f"{BLUE}│{END} {YELLOW}Processing interrupted by user{END}")
                        break
                
                # Save predictions
                if args.save_predictions and pred_instances is not None:
                    pred_instances_list.append({
                        'frame_id': frame_idx,
                        'instances': pred_instances.to_dict()
                    })
                
                frame_idx += 1
                
        except KeyboardInterrupt:
            print(f"{BLUE}│{END} {YELLOW}Processing interrupted by user{END}")
        except Exception as e:
            print(f"{BLUE}│{END} {RED}Error processing frame {frame_idx}: {e}{END}")
        
        # Compute and display statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            avg_fps = 1 / avg_time if avg_time > 0 else 0
            print(f"{BLUE}│{END} {YELLOW}Average processing speed:{END} {avg_fps:.1f} FPS")
        
        # Release resources
        cap.release()
        if video_writer:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()
        
        # Save predictions
        if args.save_predictions and pred_instances_list:
            pred_save_path = os.path.join(
                args.output_root, 
                f'predictions_{filename}_{timestamp}.json'
            )
            save_predictions(pred_instances_list, pred_save_path, pose_estimator.dataset_meta)
            print(f"{BLUE}│{END} {GREEN}✓{END} Prediction results saved to {BOLD}{os.path.basename(pred_save_path)}{END}")
        
        print(f"{BLUE}└─{'─' * 50}┘{END}")
        
        if video_writer and output_file:
            print(f"{GREEN}✓{END} The output video has been saved at {BOLD}{output_file}{END}")
        
        return True
        
    except Exception as e:
        print(f"{RED}✗ Error processing video: {e}{END}")
        return False


def initialize_models(config, workspace_dir, args):
    """Initialize detection and pose estimation models.
    
    Args:
        config: Configuration dictionary
        workspace_dir: Path to workspace directory
        args: Command-line arguments
        
    Returns:
        Tuple: (detector, pose_estimator, visualizer) or (None, None, None) if initialization fails
    """
    print(f"\n{BLUE}➤ Initializing models...{END}")
    
    try:
        # Get model paths
        model_paths = ensure_models_exist(config, workspace_dir)
        det_config_path = model_paths[0]
        det_checkpoint_path = model_paths[1]
        pose_config_path = model_paths[2]
        pose_checkpoint_path = model_paths[3]
        
        # Initialize detector
        detector = init_detector(det_config_path, det_checkpoint_path, device=args.device)
        # Adapt MMDetection pipeline for compatibility with MMPose
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        
        # Initialize pose estimator
        pose_estimator = init_pose_estimator(pose_config_path, pose_checkpoint_path, device=args.device)
        
        # Print model information
        print_model_info(detector, pose_estimator)
        
        # Initialize visualizer
        visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
        visualizer.set_dataset_meta(pose_estimator.dataset_meta)
        
        return detector, pose_estimator, visualizer
    except Exception as e:
        print(f"{RED}✗ Error initializing models: {e}{END}")
        return None, None, None


def main():
    """Main function for pose estimation."""
    # Get the base directory
    workspace_dir = str(Path(__file__).resolve().parents[2])
    print(f"{BLUE}Working directory set to: {workspace_dir}{END}")
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    if args.config is None:
        args.config = 'config.yaml'
    
    config = load_config(args.config, workspace_dir)
    if config is None:
        print(f"{RED}✗ Error: Failed to load configuration{END}")
        return
    
    # Merge configuration and arguments
    merge_configs(args, config)
    
    # Setup output paths
    setup_output_paths(args)
    
    # Check MMDetection installation
    if not has_mmdet:
        print(f"{RED}✗ Error: MMDetection not found. This script requires MMDetection for person detection.{END}")
        return
    
    # Initialize models
    detector, pose_estimator, visualizer = initialize_models(config, workspace_dir, args)
    if detector is None or pose_estimator is None or visualizer is None:
        return
    
    # Define a function to process input sources
    def process_input_source(input_source):
        """Process different input sources (webcam, realsense, image, video).
        
        Args:
            input_source: Input source specification
        """
        print(f"\n{BLUE}➤ Processing input: {input_source if input_source else 'webcam'}{END}")
        
        # Handle camera inputs
        if not input_source or input_source == 'webcam':
            process_camera(args, detector, pose_estimator, visualizer, 
                          is_realsense=False, process_frame_func=process_one_image)
            return
        
        if input_source == 'realsense':
            if not has_realsense():
                print(f"{RED}✗ Error: RealSense library not available{END}")
                return
                
            realsense_objects = init_realsense()
            if realsense_objects[0] is not None:
                process_camera(args, detector, pose_estimator, visualizer,
                              is_realsense=True, realsense_objects=realsense_objects,
                              process_frame_func=process_one_image)
            return
        
        # Handle file inputs (image/video)
        if not os.path.exists(input_source):
            print(f"{RED}✗ Error: Input file not found: {input_source}{END}")
            return
            
        # Determine input file type
        input_type = mimetypes.guess_type(input_source)[0]
        if input_type is None:
            # Try to determine type from extension
            ext = os.path.splitext(input_source)[1].lower()
            if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                input_type = 'video'
            elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                input_type = 'image'
            else:
                print(f"{RED}✗ Error: Cannot determine input type for {input_source}{END}")
                return
        
        # Process based on input type
        if 'image' in input_type:
            process_image(args, input_source, detector, pose_estimator, visualizer)
        elif 'video' in input_type:
            process_video(args, input_source, detector, pose_estimator, visualizer)
        else:
            print(f"{RED}✗ Error: Unsupported input type: {input_type}{END}")
    
    # Process the input source
    process_input_source(args.input)


if __name__ == '__main__':
    main()
