#!/usr/bin/env python
# Pose Estimation with MMPose and MMDetection
# Supports image, video, webcam, and RealSense D435i camera
# Refactored to follow the structure of MMPose's topdown_demo_with_mmdet.py

import argparse
import logging
import mimetypes
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import torch
import yaml
from mmengine.logging import print_log

# Import MMPose modules
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

# Import MMDetection modules
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    print("\033[91m✗ Error:\033[0m MMDetection not found. This script requires MMDetection for person detection.")
    exit(1)

# Import RealSense modules
try:
    import pyrealsense2 as rs
    has_realsense = True
except ImportError:
    has_realsense = False
    print("\033[93m⚠ Warning:\033[0m pyrealsense2 not found. RealSense camera will not be available.")
    
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

WORKING_DIR = str(Path(__file__).resolve().parents[2])
print(f"{BLUE}Working directory set to: {WORKING_DIR}{END}")

def ensure_models_exist(config, workspace_dir):
    """Ensure that model files exist by checking and downloading if necessary."""
    # Get the configuration parameters for detection and pose models
    det_config_name = config['detection']['config']
    det_checkpoint_name = config['detection']['checkpoint']
    pose_config_name = config['pose']['config']
    pose_checkpoint_name = config['pose']['checkpoint']
    
    print(f"{BLUE}➤ Checking for model files...{END}")
    print(f"  {CYAN}Detection config:{END}    {det_config_name}")
    print(f"  {CYAN}Detection checkpoint:{END} {det_checkpoint_name}")
    print(f"  {MAGENTA}Pose config:{END}        {pose_config_name}")
    print(f"  {MAGENTA}Pose checkpoint:{END}    {pose_checkpoint_name}")
    
    # Define paths for detection files
    det_config = os.path.join(
        workspace_dir, 'checkpoints', 'detection',
        det_config_name, 
        f"{det_config_name}.py"
    )

    det_checkpoint = os.path.join(
        workspace_dir, 'checkpoints', 'detection',
        det_checkpoint_name
    )
    
    # Define paths for pose files
    pose_config = os.path.join(
        workspace_dir, 'checkpoints', 'pose-estimation',
        pose_config_name, 
        f"{pose_config_name}.py"
    )
    pose_checkpoint = os.path.join(
        workspace_dir, 'checkpoints', 'pose-estimation',
        pose_checkpoint_name
    )
    
    # Check if files exist in alternative locations (backward compatibility)
    old_det_config = os.path.join(
        workspace_dir, 'src', 'detection', 'configs',
        f"{det_config_name}.py"
    )
    old_pose_config = os.path.join(
        workspace_dir, 'src', 'pose-estimation', 'configs',
        f"{pose_config_name}.py"
    )
    
    # Check each file separately and determine download needs
    download_files = {"detection_config": False, "detection_checkpoint": False, 
                     "pose_config": False, "pose_checkpoint": False}
    
    # Check detection config
    if os.path.exists(det_config) or os.path.exists(old_det_config):
        print(f"  {GREEN}✓ Detection config exists: {det_config_name}{END}")
        if os.path.exists(det_config):
            det_config_path = det_config
        else:
            det_config_path = old_det_config
    else:
        print(f"  {YELLOW}⚠ Missing detection config: {det_config_name}{END}")
        download_files["detection_config"] = True
        det_config_path = det_config  # Default to new path
    
    # Check detection checkpoint
    if os.path.exists(det_checkpoint):
        print(f"  {GREEN}✓ Detection checkpoint exists: {det_checkpoint_name}{END}")
    else:
        print(f"  {YELLOW}⚠ Missing detection checkpoint: {det_checkpoint_name}{END}")
        download_files["detection_checkpoint"] = True
    
    # Check pose config
    if os.path.exists(pose_config) or os.path.exists(old_pose_config):
        print(f"  {GREEN}✓ Pose config exists: {pose_config_name}{END}")
        if os.path.exists(pose_config):
            pose_config_path = pose_config
        else:
            pose_config_path = old_pose_config
    else:
        print(f"  {YELLOW}⚠ Missing pose config: {pose_config_name}{END}")
        download_files["pose_config"] = True
        pose_config_path = pose_config  # Default to new path
    
    # Check pose checkpoint
    if os.path.exists(pose_checkpoint):
        print(f"  {GREEN}✓ Pose checkpoint exists: {pose_checkpoint_name}{END}")
    else:
        print(f"  {YELLOW}⚠ Missing pose checkpoint: {pose_checkpoint_name}{END}")
        download_files["pose_checkpoint"] = True
    
    # If any files are missing, attempt to download only the needed ones
    if any(download_files.values()):
        print(f"\n{YELLOW}⚠ Some model files are missing. Will attempt to download only what's needed.{END}")
        download_script = os.path.join(workspace_dir, 'src', 'scripts', 'download_checkpoints.sh')
        
        if os.path.exists(download_script):
            # Find the config file path
            config_path = os.path.join(workspace_dir, 'src', 'config.yaml')
            if 'config_path' in config and os.path.exists(config['config_path']):
                config_path = config['config_path']
            
            # Create command with explicit config parameter
            download_cmd = f"bash {download_script} --config {config_path}"
            
            print(f"{BLUE}➤ Running:{END} {download_cmd}")
            return_code = os.system(download_cmd)
            
            if return_code != 0:
                print(f"{RED}✗ Error: Automatic download failed with exit code {return_code}.{END}")
                print(f"{YELLOW}Try running manually:{END} {download_cmd}")
                raise RuntimeError(f"Failed to download model files. Exit code: {return_code}")
            
            # Verify the files again after download
            if download_files["detection_config"] and not (os.path.exists(det_config) or os.path.exists(old_det_config)):
                print(f"{YELLOW}⚠ Warning: Detection config file still missing after download{END}")
            
            if download_files["detection_checkpoint"] and not os.path.exists(det_checkpoint):
                print(f"{YELLOW}⚠ Warning: Detection checkpoint file still missing after download{END}")
            
            if download_files["pose_config"] and not (os.path.exists(pose_config) or os.path.exists(old_pose_config)):
                print(f"{YELLOW}⚠ Warning: Pose config file still missing after download{END}")
            
            if download_files["pose_checkpoint"] and not os.path.exists(pose_checkpoint):
                print(f"{YELLOW}⚠ Warning: Pose checkpoint file still missing after download{END}")
        else:
            print(f"{RED}✗ Error: Download script not found: {download_script}{END}")
            raise FileNotFoundError(f"Download script not found: {download_script}")
    
    # Determine which files to use (prefer new structure but fall back to old)
    if not os.path.exists(det_config) and os.path.exists(old_det_config):
        det_config = old_det_config
    if not os.path.exists(pose_config) and os.path.exists(old_pose_config):
        pose_config = old_pose_config
        
    # Verify all files now exist
    if not os.path.exists(det_config):
        print(f"{RED}✗ Error: Detection config file not found: {det_config}{END}")
        raise FileNotFoundError(f"Detection config file not found: {det_config}")
    if not os.path.exists(det_checkpoint):
        print(f"{RED}✗ Error: Detection checkpoint file not found: {det_checkpoint}{END}")
        raise FileNotFoundError(f"Detection checkpoint file not found: {det_checkpoint}")
    if not os.path.exists(pose_config):
        print(f"{RED}✗ Error: Pose config file not found: {pose_config}{END}")
        raise FileNotFoundError(f"Pose config file not found: {pose_config}")
    if not os.path.exists(pose_checkpoint):
        print(f"{RED}✗ Error: Pose checkpoint file not found: {pose_checkpoint}{END}")
        raise FileNotFoundError(f"Pose checkpoint file not found: {pose_checkpoint}")
        
    return det_config, det_checkpoint, pose_config, pose_checkpoint


def init_realsense():
    """Initialize RealSense camera and return the pipeline and other objects."""
    if not has_realsense:
        print(f"{RED}✗ Error: pyrealsense2 library not found{END}")
        return None, None, None, None, None, None
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams with default settings
    # These will be overridden with config values later
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
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


def print_model_info(config, det_config, pose_config, workspace_dir):
    """Print information about the models being used."""
    det_checkpoint_path = os.path.join(
        workspace_dir, 'checkpoints', 'detection',
        config['detection']['checkpoint']
    )
    pose_checkpoint_path = os.path.join(
        workspace_dir, 'checkpoints', 'pose-estimation',
        config['pose']['checkpoint']
    )
    
    det_size = os.path.getsize(det_checkpoint_path) / (1024 * 1024)  # Convert to MB
    pose_size = os.path.getsize(pose_checkpoint_path) / (1024 * 1024)  # Convert to MB
    
    device_str = config['detection']['device']
    device_color = GREEN if 'cuda' in device_str else YELLOW
    
    # Print header with decorative elements
    print(f"\n{BOLD}{'╔' + '═' * 80 + '╗'}{END}")
    print(f"{BOLD}╠{'═' * 30}  MODEL INFORMATION  {'═' * 30}╣{END}")
    print(f"{BOLD}╠{'═' * 80}╣{END}")
    
    # Print detection model info
    print(f"{BOLD}║{END} {CYAN}Detection Model:{END}    {BLUE}{os.path.basename(det_config)}{END}")
    print(f"{BOLD}║{END} {CYAN}Detection Size:{END}     {det_size:.2f} MB")
    print(f"{BOLD}║{END} {CYAN}Detection Path:{END}     {Path(det_checkpoint_path).relative_to(WORKING_DIR)}")
    
    # Print pose model info
    print(f"{BOLD}║{END} {MAGENTA}Pose Model:{END}        {BLUE}{os.path.basename(pose_config)}{END}")
    print(f"{BOLD}║{END} {MAGENTA}Pose Size:{END}         {pose_size:.2f} MB")
    print(f"{BOLD}║{END} {MAGENTA}Pose Path:{END}         {Path(pose_checkpoint_path).relative_to(WORKING_DIR)}")
    
    # Print execution device with conditional coloring
    print(f"{BOLD}║{END} {YELLOW}Running on:{END}        {device_color}{device_str}{END}")
    
    # Additional system info
    print(f"{BOLD}║{END} {GREEN}Time:{END}             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        print(f"{BOLD}║{END} {GREEN}GPU:{END}              {torch.cuda.get_device_name(0)}")
    
    # Print footer
    print(f"{BOLD}{'╚' + '═' * 78 + '╝'}{END}\n")


def process_single_frame(args, img, detector, pose_estimator, visualizer=None, 
                      depth_img=None, depth_scale=None, show_interval=0):
    """Process one image frame with enhanced visualization and 3D coordinates."""
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
        
        # Add 3D keypoints to data_samples
        pred_instances.keypoints_3d = keypoints_3d
    
    # Convert image if needed
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)
    
    # Visualize results
    if visualizer is not None:
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
        
        # Draw 3D coordinates if available
        if pred_instances is not None:
            keypoints = pred_instances.keypoints
            keypoint_scores = pred_instances.keypoint_scores
            keypoints_3d = getattr(pred_instances, 'keypoints_3d', None)
            
            if keypoints_3d is not None:
                # For each person and keypoint
                for person_idx, (person_keypoints, person_scores, person_keypoints_3d) in enumerate(
                        zip(keypoints, keypoint_scores, keypoints_3d)):
                    for kpt_idx, (kpt, score, kpt_3d) in enumerate(
                            zip(person_keypoints, person_scores, person_keypoints_3d)):
                        if score > args.kpt_thr:
                            x, y = int(kpt[0]), int(kpt[1])
                            depth = kpt_3d[2]
                            
                            # Format text with 3D coordinates
                            text = f"{kpt_idx}: ({x},{y},{depth:.3f}m)"
                            
                            # Draw text with small font and transparent background
                            text_scale = 0.5
                            text_thickness = 1
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            
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
        
        # Calculate and display FPS and timestamp
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create a semi-transparent overlay for FPS display
        overlay = img_vis.copy()
        cv2.rectangle(img=overlay, pt1=(5, 5), pt2=(300, 40), color=(0, 0, 0), thickness=-1)
        cv2.addWeighted(src1=overlay, alpha=0.6, src2=img_vis, beta=0.4, gamma=0, dst=img_vis)
        
        # Draw FPS with gradient color (green to red based on FPS)
        fps_color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 5 else (0, 0, 255)
        
        # Draw FPS and timestamp with shadow for better visibility
        cv2.putText(
            img_vis, f"FPS: {fps:.1f}", 
            (11, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
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
    else:
        img_vis = None
    
    return img_vis, pred_instances


def save_predictions(pred_instances_list, output_path, dataset_meta):
    """Save prediction results to a JSON file."""
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


def parse_args():
    """Parse command line arguments with enhanced handling for config."""
    parser = argparse.ArgumentParser(
        description='Pose Estimation with 3D coordinates visualization')
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
    parser.add_argument('--input', 
                        type=str, 
                        default='',
                        help='Path to input image/video, or use "webcam" or "realsense"')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        default=False, 
        help='Draw bboxes of instances')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--show-interval', 
        type=float, 
        default=0.001, 
        help='Sleep seconds per frame')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Link thickness for visualization')
    parser.add_argument(
        '--use-realsense-config',
        action='store_true',
        default=False,
        help='Use RealSense configuration from config file')
    parser.add_argument(
        '--install',
        action='store_true',
        default=False,
        help='Install model files specified in config without running inference')
    
    return parser.parse_args()


def process_camera(args, detector, pose_estimator, visualizer, 
                   is_realsense=False, realsense_objects=None):
    """Process camera input (webcam or realsense)."""
    # ANSI color codes for status messages
    camera_type = "RealSense" if is_realsense else "Webcam"
    
    print(f"\n{BLUE}┌─ {MAGENTA if is_realsense else CYAN}{camera_type}{BLUE} ─{'─' * (45 - len(camera_type))}┐{END}")
    
    if is_realsense:
        if not has_realsense:
            print(f"{BLUE}│{END} {RED}✗ Error: pyrealsense2 library not found{END}")
            print(f"{BLUE}└─{'─' * 50}┘{END}")
            return
        
        if realsense_objects is None or realsense_objects[0] is None:
            print(f"{BLUE}│{END} {RED}✗ Error: RealSense pipeline not initialized{END}")
            print(f"{BLUE}└─{'─' * 50}┘{END}")
            return
        
        pipeline, _, depth_scale, _, _, align = realsense_objects
        print(f"{BLUE}│{END} {YELLOW}Processing RealSense camera input...{END}")
        
    else:
        print(f"{BLUE}│{END} {YELLOW}Initializing webcam...{END}")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{BLUE}│{END} {RED}✗ Error: Could not open webcam{END}")
            print(f"{BLUE}└─{'─' * 50}┘{END}")
            return
        
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
                img_vis, pred_instances = process_single_frame(
                    args, color_image, detector, pose_estimator, visualizer, 
                    depth_img=depth_image, depth_scale=depth_scale, 
                    show_interval=args.show_interval
                )
                
            else:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame (no depth for standard webcam)
                img_vis, pred_instances = process_single_frame(
                    args, frame, detector, pose_estimator, visualizer, 
                    show_interval=args.show_interval
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


def main():
    """Main function."""
    print(f"{MAGENTA}3D Human Pose Estimation with MMPose & MMDetection{END}")
    print(f"{GREEN}{'─' * 60}{END}\n")
    
    # Parse arguments
    args = parse_args()
    workspace_dir = str(Path(__file__).resolve().parent.parent.parent)
    
    # Check if install mode is used without config
    if args.install and not args.config:
        print(f"{RED}✗ Error: Installation mode requires a config file.{END}")
        print(f"{YELLOW}Please provide a config file path with --config parameter.{END}")
        print(f"Example: {CYAN}python {__file__} --config src/config.yaml --install{END}")
        return
    
    # Load config file if specified
    config = None
    config_path = None
    if args.config:
        config_path = args.config
        if not os.path.isabs(config_path):
            # First try the direct path
            if os.path.exists(args.config):
                config_path = args.config
            # Then try relative to src
            elif os.path.exists(os.path.join(workspace_dir, 'src', args.config)):
                config_path = os.path.join(workspace_dir, 'src', args.config)
            # Finally try relative to workspace
            else:
                config_path = os.path.join(workspace_dir, args.config)
        
        # Check if config file exists
        if not os.path.exists(config_path):
            print(f"{RED}✗ Error: Config file not found: {config_path}{END}")
            if args.install:
                print(f"{YELLOW}Installation aborted. Please provide a valid config file path.{END}")
                return
            else:
                print(f"{YELLOW}⚠ Will attempt to use command line arguments instead{END}")
        else:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                # Store the config path for the download script
                config['config_path'] = config_path
                print(f"{GREEN}✓{END} Loaded configuration from {BOLD}{config_path}{END}")
            except Exception as e:
                print(f"{RED}✗ Error loading configuration:{END} {e}")
                if args.install:
                    print(f"{YELLOW}Installation aborted. Config file is invalid.{END}")
                    return
                print(f"{YELLOW}⚠ Using command line arguments instead{END}")
    
    # Initialize model paths based on config or args
    if config:
        # Handle installation mode
        if args.install:
            print(f"{BLUE}➤ Installation mode activated. Checking and downloading model files...{END}")
            
            # Run the download script directly when in installation mode
            download_script = os.path.join(workspace_dir, 'src', 'scripts', 'download_checkpoints.sh')
            if not os.path.exists(download_script):
                print(f"{RED}✗ Error: Download script not found: {download_script}{END}")
                return
                
            # Execute the download script with explicit config parameter
            download_cmd = f"bash {download_script} --config {config_path}"
            print(f"{BLUE}➤ Running:{END} {download_cmd}")
            
            return_code = os.system(download_cmd)
            if return_code == 0:
                # Check if files are now available
                try:
                    det_config, det_checkpoint, pose_config, pose_checkpoint = ensure_models_exist(
                        config, workspace_dir
                    )
                    print(f"\n{GREEN}✓ Model installation complete! The following files are ready:{END}")
                    print(f"  {CYAN}Detection config:{END}    {det_config}")
                    print(f"  {CYAN}Detection checkpoint:{END} {det_checkpoint}")
                    print(f"  {MAGENTA}Pose config:{END}        {pose_config}")
                    print(f"  {MAGENTA}Pose checkpoint:{END}    {pose_checkpoint}")
                except Exception as e:
                    print(f"{RED}✗ Some files could not be verified after download: {e}{END}")
            else:
                print(f"{RED}✗ Installation failed with exit code {return_code}{END}")
                print(f"{YELLOW}Try running manually:{END} {download_cmd}")
            return
        
        # Ensure models exist (and download if necessary)
        det_config, det_checkpoint, pose_config, pose_checkpoint = ensure_models_exist(
            config, workspace_dir
        )
        
        # Override args with config values
        if not args.det_cat_id:
            args.det_cat_id = config['detection']['cat_id']
        if not args.bbox_thr:
            args.bbox_thr = config['detection']['bbox_thr']
        if not args.nms_thr:
            args.nms_thr = config['detection']['nms_thr']
        if not args.kpt_thr:
            args.kpt_thr = config['pose']['kpt_thr']
        if not args.radius:
            args.radius = config['pose']['radius'] 
        if not args.thickness:
            args.thickness = config['pose']['thickness']
        if not args.device:
            args.device = config['pose']['device']
        if not args.draw_bbox and 'visualization' in config:
            args.draw_bbox = config['visualization']['draw_bbox']
        if not args.draw_heatmap and 'visualization' in config:
            args.draw_heatmap = config['visualization']['draw_heatmap']
        if not args.show_kpt_idx and 'visualization' in config:
            args.show_kpt_idx = config['visualization']['show_kpt_idx']
        if args.skeleton_style == 'mmpose' and 'visualization' in config:
            args.skeleton_style = config['visualization']['skeleton_style']
        if not args.show and 'io' in config:
            args.show = config['io']['show']
        if not args.output_root and 'io' in config:
            args.output_root = config['io']['output_root']
        if not args.save_predictions and 'io' in config:
            args.save_predictions = config['io']['save_predictions']
        if args.show_interval == 0.001 and 'io' in config:
            args.show_interval = config['io']['show_interval']
    else:
        # Use args directly
        if not args.det_config or not args.det_checkpoint or not args.pose_config or not args.pose_checkpoint:
            print(f"{RED}✗ Error:{END} When not using a config file, you must specify:")
            print("  --det-config, --det-checkpoint, --pose-config, --pose-checkpoint")
            return
        
        det_config = args.det_config
        det_checkpoint = args.det_checkpoint
        pose_config = args.pose_config
        pose_checkpoint = args.pose_checkpoint
    
    # Convert relative paths to absolute
    if not os.path.isabs(det_config):
        det_config = os.path.join(workspace_dir, det_config)
    if not os.path.isabs(det_checkpoint):
        det_checkpoint = os.path.join(workspace_dir, det_checkpoint)
    if not os.path.isabs(pose_config):
        pose_config = os.path.join(workspace_dir, pose_config)
    if not os.path.isabs(pose_checkpoint):
        pose_checkpoint = os.path.join(workspace_dir, pose_checkpoint)
    
    # Validate input
    if not args.input:
        print(f"{RED}✗ Error:{END} Please specify an input (image path, video path, 'webcam', or 'realsense')")
        return
    
    # Check for output directory
    assert args.show or (args.output_root != '')
    if args.output_root:
        if not os.path.isabs(args.output_root):
            args.output_root = os.path.join(workspace_dir, args.output_root)
        mmengine.mkdir_or_exist(args.output_root)
    
    # Setup prediction save path
    if args.save_predictions:
        assert args.output_root != ''
        if args.input not in ['webcam', 'realsense']:
            args.pred_save_path = f'{args.output_root}/results_' \
                f'{os.path.splitext(os.path.basename(args.input))[0]}.json'
    
    # Check for CUDA and update device if needed
    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f"{YELLOW}⚠ Warning: CUDA not available, using CPU instead of {args.device}{END}")
        args.device = 'cpu'
    
    # Build detector
    print(f"{BLUE}➤ Initializing detection model...{END}")
    detector = init_detector(det_config, str(Path(det_checkpoint).relative_to(WORKING_DIR)), device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    # Build pose estimator
    print(f"{BLUE}➤ Initializing pose estimation model...{END}")
    pose_estimator = init_pose_estimator(
        pose_config,
        str(Path(pose_checkpoint).relative_to(WORKING_DIR)),
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))
        )
    )
    
    # Initialize visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = 0.8  # Default transparency
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    visualizer.set_dataset_meta(
        pose_estimator.dataset_meta, skeleton_style=args.skeleton_style
    )
    
    # Print model information
    print_model_info(config if config else {
        'detection': {'config': os.path.basename(det_config), 'checkpoint': os.path.basename(det_checkpoint), 'device': args.device},
        'pose': {'config': os.path.basename(pose_config), 'checkpoint': os.path.basename(pose_checkpoint), 'device': args.device}
    }, det_config, pose_config, workspace_dir)
    
    # Initialize RealSense if needed
    realsense_objects = None
    use_realsense = False
    if args.input == 'realsense' or (config and 'camera' in config and config['camera'].get('use_realsense', False)):
        use_realsense = True
        realsense_objects = init_realsense()
    
    # Process based on input type
    if args.input == 'webcam':
        print(f"{BLUE}➤ Processing webcam input...{END}")
        process_camera(args, detector, pose_estimator, visualizer)
    
    elif args.input == 'realsense':
        if not has_realsense:
            print(f"{RED}✗ Error:{END} pyrealsense2 library not found")
            return
        
        print(f"{BLUE}➤ Processing RealSense camera input...{END}")
        process_camera(args, detector, pose_estimator, visualizer, True, realsense_objects)
    
    else:
        # Check if input is valid
        if not os.path.exists(args.input):
            print(f"{RED}✗ Error:{END} Input file not found: {BOLD}{args.input}{END}")
            return
        
        # Check if input is image or video
        input_type = mimetypes.guess_type(args.input)[0]
        if input_type is None:
            print(f"{RED}✗ Error:{END} Cannot determine file type for {BOLD}{args.input}{END}")
            return
        
        input_type = input_type.split('/')[0]
        
        # Process input based on type
        if input_type == 'image':
            print(f"{BLUE}➤ Processing image: {BOLD}{args.input}{END}")
            
            # Process the image
            img_vis, pred_instances = process_single_frame(
                args, args.input, detector, pose_estimator, visualizer
            )
            
            # Display result
            if args.show:
                cv2.imshow('Pose Estimation', img_vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Save result
            if args.output_root:
                output_file = os.path.join(args.output_root, os.path.basename(args.input))
                cv2.imwrite(output_file, img_vis)
                print(f"{GREEN}✓{END} Result saved to {BOLD}{Path(output_file).relative_to(WORKING_DIR)}{END}")
            
            # Save predictions
            if args.save_predictions and pred_instances is not None:
                pred_instances_list = [dict(frame_id=0, instances=split_instances(pred_instances))]
                save_predictions(pred_instances_list, args.pred_save_path, pose_estimator.dataset_meta)
        
        elif input_type == 'video':
            print(f"{BLUE}➤ Processing video: {BOLD}{args.input}{END}")
            
            # Open video file
            cap = cv2.VideoCapture(args.input)
            if not cap.isOpened():
                print(f"{RED}✗ Error:{END} Could not open video {BOLD}{args.input}{END}")
                return
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer
            video_writer = None
            if args.output_root:
                output_file = os.path.join(args.output_root, os.path.basename(args.input))
                
                # Use original extension or default to mp4
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_writer = cv2.VideoWriter(
                    output_file, fourcc, fps, (frame_width, frame_height)
                )
            
            pred_instances_list = []
            frame_idx = 0
            
            # ANSI color codes for progress bar
            while cap.isOpened():
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Print progress every 10 frames if not showing the video
                if frame_idx % 10 == 0 and not args.show:
                    progress = int(50 * frame_idx / total_frames)
                    progress_bar = GREEN + "█" * progress + YELLOW + "░" * (50 - progress) + END
                    percent = frame_idx / total_frames * 100
                    print(f"\r{BLUE}Processing: {END}{progress_bar} {percent:.1f}% ({frame_idx}/{total_frames})", end="", flush=True)
                
                # Process frame
                img_vis, pred_instances = process_single_frame(
                    args, frame, detector, pose_estimator, visualizer, 
                    show_interval=args.show_interval
                )
                
                # Save prediction results
                if args.save_predictions and pred_instances is not None:
                    pred_instances_list.append(
                        dict(frame_id=frame_idx, instances=split_instances(pred_instances))
                    )
                
                # Display result
                if args.show:
                    cv2.imshow('Pose Estimation', img_vis)
                    
                    # Press ESC to exit
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    
                    # Sleep for display interval
                    time.sleep(args.show_interval)
                
                # Write frame to output video
                if video_writer:
                    video_writer.write(img_vis)
            
            print(f"\n{BLUE}Finalizing video processing...{END}")
            
            # Clean up
            cap.release()
            print(f"{GREEN}✓{END} Video file released")
            
            if video_writer:
                video_writer.release()
                print(f"{GREEN}✓{END} Output video writer released")
                
            cv2.destroyAllWindows()
            print(f"{GREEN}✓{END} Windows closed")
            
            # Save predictions
            if args.save_predictions and pred_instances_list:
                save_predictions(pred_instances_list, args.pred_save_path, pose_estimator.dataset_meta)
            
            if args.output_root:
                print_log(
                    f'The output video has been saved at {output_file}',
                    logger='current',
                    level=logging.INFO
                )
        
        else:
            print(f"{RED}✗ Error:{END} Unsupported input type: {BOLD}{input_type}{END}")


if __name__ == '__main__':
    main()
