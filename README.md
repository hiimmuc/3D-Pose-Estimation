# Pose Estimation Project

This project provides a clean and structured implementation of pose estimation for both images and videos, based on MMPose and MMDetection frameworks. It supports regular RGB cameras, webcams, and RealSense D435i depth cameras.

## Features

-   Image, video, webcam, and RealSense D435i camera support
-   Advanced 3D visualization capabilities:
    -   2D pixel coordinates (white)
    -   3D camera coordinates (green)
    -   3D world coordinates (cyan)
    -   Coordinate transformation relative to any keypoint (development...)
-   Both CPU and GPU support
-   FPS and timestamp display
-   Camera calibration support
-   Customizable visualization with color-coded information
-   Robust handling of missing depth data

## Directory Structure

```
├── src/
│   ├── config.yaml             # Main configuration file
│   ├── pose-estimation/        # Pose estimation related files
│   │   ├── configs/            # Pose estimation model configs
│   │   ├── projects/           # Additional pose projects
│   │   └── model-index.yml     # Index to model checkpoints
│   ├── detection/              # Object detection related files
│   │   ├── configs/            # Detection model configs
│   │   ├── projects/           # Additional detection projects
│   │   └── model-index.yml     # Index to model checkpoints
│   └── scripts/                # Python and shell scripts
│       ├── download_checkpoints.sh    # Script to download model weights
│       ├── pose_estimation.py         # Main pose estimation script
│       └── run_pose_estimation.sh     # Helper script to run pose estimation
├── checkpoints/                # Directory to store model checkpoints
│   ├── detection/              # Object detection model weights
│   └── pose-estimation/        # Pose estimation model weights
└── output/                     # Output directory for results
```

## Getting Started

### Prerequisites

-   Python 3.6+
-   PyTorch
-   MMPose
-   MMDetection
-   OpenCV
-   pyrealsense2 (for RealSense camera support)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/hiimmuc/3D-Pose-Estimation.git
cd 3D-Pose-Estimation
```

2. Download the model checkpoints:

```bash
./src/scripts/download_checkpoints.sh
```

### Usage

#### Using the helper script:

```bash
# Process an image:
 python ./src/scripts/main.py--input /path/to/image.jpg

# Process a video:
 python ./src/scripts/main.py--input /path/to/video.mp4

# Use webcam:
 python ./src/scripts/main.py --webcam

# Use RealSense camera:
 python ./src/scripts/main.py--realsense

# Transform coordinates relative to a specific keypoint:
 python ./src/scripts/main.py --input /path/to/video.mp4 --reference-keypoint 0
```

#### Using the Python script directly:

```bash
python ./src/scripts/main.py --config src/config.yaml --input /path/to/image_or_video.jpg --calibration-file /path/to/calibration-file
```

### Configuration

You can customize the behavior by editing the `src/config.yaml` file. The configuration includes:

-   Detection model settings
-   Pose estimation model settings
-   Visualization options (including color schemes)
-   Input/Output settings
-   Camera settings and calibration parameters

## Example Output

When running the script, you will see the pose estimation results with multiple coordinate representations for each keypoint:

-   2D pixel coordinates (u, v) showing the position in the image
-   3D camera coordinates (X, Y, Z) showing the position relative to the camera
-   3D world coordinates showing the position in the global reference frame
-   Optional transformed coordinates relative to a specified keypoint

The visualization uses color coding to distinguish different types of information:

-   White text for 2D pixel coordinates
-   Green text for camera coordinates
-   Yellow text for world coordinates

For depth information:

-   RGB images/videos without depth: Z coordinate will be NaN
-   RealSense camera input: actual depth values in meters
-   Transformed coordinates: distances relative to the reference keypoint

## License

This project is built on top of MMPose and MMDetection, which are released under the Apache 2.0 license.
