#!/usr/bin/env python
"""
Configuration utilities for pose estimation.

This module handles loading, validating, and processing configuration files,
as well as ensuring required model files exist and are accessible.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from decoration import *


def load_config(config_path: str, workspace_dir: str) -> Optional[Dict]:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to the config file
        workspace_dir: Path to workspace directory
        
    Returns:
        Dict: Configuration dictionary or None if loading failed
    """
    if not os.path.isabs(config_path):
        # First try the direct path
        if os.path.exists(config_path):
            pass
        # Then try relative to src
        elif os.path.exists(os.path.join(workspace_dir, 'src', config_path)):
            config_path = os.path.join(workspace_dir, 'src', config_path)
        # Finally try relative to workspace
        else:
            config_path = os.path.join(workspace_dir, config_path)
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"{RED}✗ Error: Config file not found: {config_path}{END}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Store the config path for the download script
        config['config_path'] = config_path
        print(f"{GREEN}✓{END} Loaded configuration from {BOLD}{config_path}{END}")
        return config
    except Exception as e:
        print(f"{RED}✗ Error loading configuration:{END} {e}")
        return None


def ensure_models_exist(config: Dict, workspace_dir: str) -> Tuple[str, str, str, str]:
    """Ensure that model files exist by checking and downloading if necessary.
    
    Args:
        config: Configuration dictionary
        workspace_dir: Path to workspace directory
        
    Returns:
        Tuple containing paths to detection and pose models (configs and checkpoints)
        
    Raises:
        FileNotFoundError: If any required model file is not found
    """
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


def print_model_info(config: Dict, det_config: str, pose_config: str, workspace_dir: str) -> None:
    """Print information about the models being used.
    
    Args:
        config: Configuration dictionary
        det_config: Path to detection config
        pose_config: Path to pose config
        workspace_dir: Path to workspace directory
    """
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
    print(f"{BOLD}╠{'═' * 30}  c  {'═' * 30}╣{END}")
    print(f"{BOLD}╠{'═' * 80}╣{END}")
    
    # Print detection model info
    print(f"{BOLD}║{END} {CYAN}Detection Model:{END}    {BLUE}{os.path.basename(det_config)}{END}")
    print(f"{BOLD}║{END} {CYAN}Detection Size:{END}     {det_size:.2f} MB")
    print(f"{BOLD}║{END} {CYAN}Detection Path:{END}     {Path(det_checkpoint_path).relative_to(workspace_dir)}")
    
    # Print pose model info
    print(f"{BOLD}║{END} {MAGENTA}Pose Model:{END}        {BLUE}{os.path.basename(pose_config)}{END}")
    print(f"{BOLD}║{END} {MAGENTA}Pose Size:{END}         {pose_size:.2f} MB")
    print(f"{BOLD}║{END} {MAGENTA}Pose Path:{END}         {Path(pose_checkpoint_path).relative_to(workspace_dir)}")
    
    # Print execution device with conditional coloring
    print(f"{BOLD}║{END} {YELLOW}Running on:{END}        {device_color}{device_str}{END}")
    
    # Additional system info
    from datetime import datetime

    import torch
    print(f"{BOLD}║{END} {GREEN}Time:{END}             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if torch.cuda.is_available():
        print(f"{BOLD}║{END} {GREEN}GPU:{END}              {torch.cuda.get_device_name(0)}")
    
    # Print footer
    print(f"{BOLD}{'╚' + '═' * 80 + '╝'}{END}\n")


def print_model_info(detector, pose_estimator):
    """Print information about the initialized models.
    
    Args:
        detector: Initialized detector model
        pose_estimator: Initialized pose estimator model
    """
    # Print header with decorative elements
    print(f"\n{BOLD}{'╔' + '═' * 80 + '╗'}{END}")
    print(f"{BOLD}╠{'═' * 30} MODEL INFORMATION {'═' * 31}╣{END}")
    print(f"{BOLD}╠{'═' * 80}╣{END}")
    
    # Print detection model info
    detector_name = detector.cfg.model.get('type', 'Unknown')
    pose_name = pose_estimator.cfg.model.get('type', 'Unknown')
    
    print(f"{BOLD}║{END} {CYAN}Detection Model:{END}    {BLUE}{detector_name}{END}")
    print(f"{BOLD}║{END} {CYAN}Device:{END}             {GREEN}{str(next(detector.parameters()).device)}{END}")
    
    # Print pose model info
    print(f"\n{BOLD}║{END} {MAGENTA}Pose Model:{END}        {BLUE}{pose_name}{END}")
    print(f"{BOLD}║{END} {MAGENTA}Device:{END}             {GREEN}{str(next(pose_estimator.parameters()).device)}{END}")
    
    # Print keypoint information
    keypoint_names = pose_estimator.dataset_meta.get('keypoint_names', [])
    skeleton = pose_estimator.dataset_meta.get('skeleton_links', [])
    
    print(f"{BOLD}║{END} {MAGENTA}Keypoints:{END}          {len(keypoint_names)} points")
    print(f"{BOLD}║{END} {MAGENTA}Skeleton Links:{END}     {len(skeleton)} connections")
    
    print(f"{BOLD}{'╚' + '═' * 80 + '╝'}{END}")
