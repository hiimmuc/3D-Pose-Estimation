# Pose Estimation Project

This project provides a clean and structured implementation of pose estimation for both images and videos, based on MMPose and MMDetection frameworks. It supports regular RGB cameras, webcams, and RealSense D435i depth cameras.

## Features

-   Image, video, webcam, and RealSense D435i camera support
-   3D coordinate display for each joint (with depth information when available)
-   Both CPU and GPU support
-   FPS and timestamp display
-   Customizable visualization

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
git clone https://github.com/your_username/pose-estimation-project.git
cd pose-estimation-project
```

2. Download the model checkpoints:

```bash
./src/scripts/download_checkpoints.sh
```

### Usage

#### Using the helper script:

```bash
# Process an image:
./src/scripts/run_pose_estimation.sh --input /path/to/image.jpg

# Process a video:
./src/scripts/run_pose_estimation.sh --input /path/to/video.mp4

# Use webcam:
./src/scripts/run_pose_estimation.sh --webcam

# Use RealSense camera:
./src/scripts/run_pose_estimation.sh --realsense
```

#### Using the Python script directly:

```bash
python src/scripts/pose_estimation.py --config src/config.yaml --input /path/to/image_or_video.jpg
```

### Configuration

You can customize the behavior by editing the `src/config.yaml` file. The configuration includes:

-   Detection model settings
-   Pose estimation model settings
-   Visualization options
-   Input/Output settings
-   Camera settings

## Example Output

When running the script, you will see the pose estimation results with 3D coordinates for each keypoint.

-   For RGB images/videos without depth information, the Z coordinate will be 0.
-   For RealSense camera input, actual depth values will be shown for each keypoint.

## License

This project is built on top of MMPose and MMDetection, which are released under the Apache 2.0 license.
