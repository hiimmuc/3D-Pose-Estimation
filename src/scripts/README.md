# Pose Estimation System

This is a CLEAN architecture implementation of a pose estimation system using MMPose and MMDetection frameworks. The system is designed to be modular, maintainable, and extensible.

## Architecture Overview

The system follows CLEAN architecture principles, organizing code into specialized modules with clear responsibilities:

```
src/scripts/
├── main.py                # Entry point that ties everything together
├── argument_parser.py     # Command-line argument handling
├── camera_utils.py        # Camera initialization and streaming
├── config_utils.py        # Configuration loading and validation
├── image_processor.py     # Image processing and pose detection
└── save_utils.py          # Utilities for saving results
```

### Module Responsibilities

-   **main.py**: Entry point for the application, orchestrating all components
-   **argument_parser.py**: Handles command-line arguments and merges them with configuration
-   **camera_utils.py**: Manages camera input (webcam, RealSense) and frame acquisition
-   **config_utils.py**: Loads and validates configuration files, ensures required models exist
-   **image_processor.py**: Processes images to detect poses and visualize results
-   **save_utils.py**: Handles saving prediction results and outputs to disk

## Usage

```bash
# Basic usage with a webcam
python src/scripts/main.py

# Process an image file
python src/scripts/main.py --input examples/sample_image.jpg --show

# Process a video file
python src/scripts/main.py --input examples/sample_video.mp4 --show

# Use RealSense camera (if available)
python src/scripts/main.py --input realsense --show

# Save results to output directory
python src/scripts/main.py --input examples/sample_image.jpg --output-root output

# Use a custom configuration file
python src/scripts/main.py --config path/to/config.yaml
```

## Key Features

-   2D and 3D pose estimation with real-time visualization
-   Support for multiple input sources (image, video, webcam, RealSense)
-   Visualization of pose keypoints with optional 3D coordinates
-   Performance metrics display (FPS, timestamp)
-   Saving results as images, videos, and JSON predictions
-   Robust error handling and user feedback

## Dependencies

-   MMDetection and MMPose for detection and pose estimation
-   OpenCV for image processing and visualization
-   PyRealSense2 (optional) for RealSense camera support
-   CUDA-enabled environment recommended for real-time performance

## Camera calibration

1. print the pattern without any distortions
2. check if the printed squares are correct squares
3. glue the pattern to a solid board
4. fix the camera lens zoom, the calibration values change with the lens zoom changes
5. record a video with the pattern moving in front of the camera

-   the pattern should be most of the time completely visible
-   try to move the pattern to cover all parts of the camera view, pay attention to the corners
-   the length of the video should be 1 or 2 minutes

6. run the calibration.py to extract chessboard pattern corners from the video and perform camera calibration
