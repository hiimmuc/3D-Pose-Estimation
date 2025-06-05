# Pose Estimation Project

This project provides a clean architecture implementation for 2D/3D pose estimation. It supports processing images, videos, webcam feeds, and Intel RealSense D435i depth cameras using MMPose and MMDetection frameworks.

## Features

-   **CLEAN Architecture** for maintainable, testable, and flexible codebase
-   **Multiple input sources** - images, videos, webcams, and RealSense cameras
-   **3D visualization capabilities**:
    -   2D pixel coordinates (white)
    -   3D camera coordinates (green)
    -   3D world coordinates (cyan)
-   **Hardware flexibility** - CPU and GPU support
-   **Real-time performance** with FPS display
-   **Camera calibration** for accurate 3D pose reconstruction
-   **Customizable visualization** with color-coded information
-   **Robust depth processing** with fallback for missing depth data

## Directory Structure

```
├── run_pose_estimation.py      # Main entry point (simplified launcher)
├── setup.sh                    # Setup script to install dependencies
├── src/                        # Source code with CLEAN architecture
│   ├── application/            # Application layer (use cases)
│   │   └── use_cases/          # Business logic use cases
│   ├── domain/                 # Domain layer (entities, interfaces)
│   │   ├── entities/           # Core business entities
│   │   └── interfaces/         # Interfaces/ports for adapters
│   ├── interfaces/             # Interfaces layer (adapters, presenters)
│   │   ├── adapters/           # Adapters for external systems
│   │   └── presenters/         # Presenters for output formatting
│   ├── frameworks/             # Frameworks layer (utilities, drivers)
│   │   └── utils/              # Utility functions
│   ├── config.yaml             # Configuration file
│   └── main.py                 # Original main entry point
├── checkpoints/                # Directory for model checkpoints
│   ├── detection/              # Object detection models
│   └── pose-estimation/        # Pose estimation models
├── examples/                   # Example input files
│   ├── sample_image.jpg        # Sample test image
│   └── sample_video.mp4        # Sample test video
├── calibration_data/           # Camera calibration files
│   └── calib_hd_pro_webcam_c920__046d_082d__1920.json # Sample calibration
└── output/                     # Output directory for results
```

## Quick Start

### Prerequisites

-   Python 3.6+
-   CUDA-capable GPU (recommended) or CPU
-   Intel RealSense SDK (optional, for RealSense cameras)

### Installation

1. Clone the repository:

    ```bash
    git clone [repository-url]
    cd [repository-directory]
    ```

2. Run the setup script to install dependencies:
    ```bash
    ./setup.sh
    ```

### Usage

The system can be run with the simplified launcher script:

```bash
# Process an image
./run_pose_estimation.py --image examples/sample_image.jpg --show

# Process a video
./run_pose_estimation.py --video examples/sample_video.mp4 --output-root output --show

# Use webcam
./run_pose_estimation.py --webcam --show

# Use RealSense camera
./run_pose_estimation.py --realsense --show

# With camera calibration
./run_pose_estimation.py --webcam --calibration calibration_data/calib_hd_pro_webcam_c920__046d_082d__1920.json --show
```

Use the `--help` option to see all available parameters:

```bash
./run_pose_estimation.py --help
```

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
