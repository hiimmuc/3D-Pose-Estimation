# 3D Pose Estimation System 🧍‍♂️🔍

![Badge-License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Badge-Python](https://img.shields.io/badge/Python-3.6+-blue)
![Badge-MMPose](https://img.shields.io/badge/Framework-MMPose-green)

A state-of-the-art pose estimation system capable of advanced 3D coordinate visualization with support for multiple camera inputs including RealSense depth cameras.

<p align="center">
  <img src="https://img.shields.io/badge/RealTime-Enabled-success" alt="RealTime">
  <img src="https://img.shields.io/badge/3D_Coordinates-Supported-success" alt="3D">
  <img src="https://img.shields.io/badge/MMDetection-Integrated-informational" alt="MMDetection">
  <img src="https://img.shields.io/badge/RealSense-Compatible-blue" alt="RealSense">
</p>

## ✨ Key Features

-   **📷 Multi-source Support** - Process images, videos, webcams, and RealSense cameras
-   **🔍 Advanced 3D Visualization**:
    -   2D pixel coordinates (white text)
    -   3D camera coordinates (green text)
    -   3D world coordinates (cyan text)
    -   Body-relative coordinate systems
-   **⚡ Hardware Flexibility** - Optimized for both GPU and CPU execution
-   **📊 Performance Metrics** - Real-time FPS counter and processing statistics
-   **📏 Camera Calibration** - Support for accurate 3D reconstruction
-   **🎨 Customizable UI** - Color-coded information and display options
-   **💪 Robust Processing** - Intelligent fallbacks for missing depth data

## 🏗️ Architecture

This project is structured with modularity in mind and is being refactored to follow CLEAN architecture principles:

<p align="center">
  <table>
    <tr>
      <th>Layer</th>
      <th>Responsibility</th>
      <th>Current Implementation</th>
      <th>Future Implementation</th>
    </tr>
    <tr>
      <td>📊 Domain</td>
      <td>Business entities and rules</td>
      <td>Embedded in scripts</td>
      <td>Keypoint, Pose, Frame entities</td>
    </tr>
    <tr>
      <td>⚙️ Application</td>
      <td>Use cases and business logic</td>
      <td>main.py, image_processor.py</td>
      <td>Process frames, Calculate 3D keypoints</td>
    </tr>
    <tr>
      <td>🔌 Interfaces</td>
      <td>Adapters and presenters</td>
      <td>camera_utils.py</td>
      <td>Camera adapters, Model adapters, Visualizers</td>
    </tr>
    <tr>
      <td>🧰 Frameworks</td>
      <td>External tools and utilities</td>
      <td>config_utils.py, save_utils.py</td>
      <td>Camera utils, File utils, Logging</td>
    </tr>
  </table>
</p>

## 📁 Current Directory Structure

````
├── src/                        # Source code
│   ├── scripts/                # Current implementation
│   │   ├── main.py             # Main entry point
│   │   ├── image_processor.py  # Core image processing logic
│   │   ├── camera_utils.py     # Camera handling utilities
│   │   ├── config_utils.py     # Configuration utilities
│   │   ├── save_utils.py       # Output saving utilities
│   │   ├── argument_parser.py  # Command line argument parsing
│   │   └── download_checkpoints.sh # Download script for models
│   ├── config.yaml             # Configuration file
│   ├── detection/              # Detection model configs
│   └── pose-estimation/        # Pose estimation model configs
├── mmdetection/                # MMDetection framework (submodule)
├── mmpose/                     # MMPose framework (submodule)
├── checkpoints/                # Directory for model checkpoints
│   ├── detection/              # Object detection models
│   └── pose-estimation/        # Pose estimation models
├── examples/                   # Example input files
│   ├── sample_image.jpg        # Sample image for testing
│   └── sample_video.mp4        # Sample video for testing
├── calibration_data/           # Camera calibration files
│   └── calib_hd_pro_webcam_c920__046d_082d__1920.json # Webcam calibration
└── output/                     # Output directory for results


## 🚀 Quick Start

### Prerequisites

-   Python 3.6+
-   CUDA-capable GPU (recommended) or CPU
-   Intel RealSense SDK (optional, for RealSense cameras)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/3D-Pose-Estimation.git
cd 3D-Pose-Estimation
````

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install mmdet
mim install mmpose
```

3. Download the model checkpoints:

```bash
bash src/scripts/download_checkpoints.sh
```

### Usage

Run the script with different input options:

```bash
# Process an image:
python src/scripts/main.py --input examples/sample_image.jpg --output-root output --show

# Process a video:
python src/scripts/main.py --input examples/sample_video.mp4 --output-root output --show

# Use webcam:
python src/scripts/main.py --webcam --output-root output --show

# Use RealSense camera:
python src/scripts/main.py --realsense --output-root output --show

# With camera calibration:
python src/scripts/main.py --webcam --calibration calibration_data/calib_hd_pro_webcam_c920__046d_082d__1920.json --output-root output --show
```

Use the `--help` option to see all available parameters:

```bash
python src/scripts/main.py --help
```

## ⚙️ Configuration

You can customize the behavior by editing the `src/config.yaml` file. The configuration includes:

-   Detection model settings
-   Pose estimation model settings
-   Visualization options (including color schemes)
-   Input/Output settings
-   Camera settings and calibration parameters

## Advanced Usage

```bash
# Transform coordinates relative to a specific keypoint:
python src/scripts/main.py --input examples/sample_video.mp4 --reference-keypoint 0

# Using a specific config file:
python src/scripts/main.py --config src/config.yaml --input examples/sample_image.jpg
```

## 🖼️ Example Output

When running the script, you will see the pose estimation results with multiple coordinate representations for each keypoint:

-   2D pixel coordinates (u, v) showing the position in the image
-   3D camera coordinates (X, Y, Z) showing the position relative to the camera
-   3D world coordinates showing the position in the global reference frame
-   Optional transformed coordinates relative to a specified keypoint

The visualization uses color coding to distinguish different types of information:

-   White text for 2D pixel coordinates
-   Green text for camera coordinates
-   Yellow text for world coordinates (when available)

For depth information:

-   RGB images/videos without depth: Z coordinate will be NaN
-   RealSense camera input: actual depth values in meters
-   Transformed coordinates: distances relative to the reference keypoint

## 📄 License

This project is built on top of MMPose and MMDetection, which are released under the Apache 2.0 license.
