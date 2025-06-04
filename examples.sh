#!/bin/bash

# Example usage of the pose estimation project
# This script demonstrates various ways to use the pose estimation system

WORKSPACE_DIR="/home/phgnam/Workspace/VINMOTION/workspace"
CONFIG_FILE="$WORKSPACE_DIR/src/config.yaml"

# First ensure the checkpoints are downloaded
echo "Ensuring model checkpoints are downloaded..."
bash "$WORKSPACE_DIR/src/scripts/download_checkpoints.sh"

# Create a directory for our examples
mkdir -p "$WORKSPACE_DIR/examples"

# Create a symlink to the scripts directory for convenience
ln -sf "$WORKSPACE_DIR/src/scripts" "$WORKSPACE_DIR/examples/scripts"

# Function to download a sample image if not exists
download_sample() {
    local file="$WORKSPACE_DIR/examples/sample_image.jpg"
    if [ ! -f "$file" ]; then
        echo "Downloading sample image..."
        wget -O "$file" "https://raw.githubusercontent.com/open-mmlab/mmpose/main/demo/resources/demo.jpg"
    fi
    
    local video="$WORKSPACE_DIR/examples/sample_video.mp4"
    if [ ! -f "$video" ]; then
        echo "Downloading sample video..."
        wget -O "$video" "https://raw.githubusercontent.com/open-mmlab/mmpose/main/demo/resources/demo.mp4"
    fi
}

# Download sample files
download_sample

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Display available examples with colorful formatting
echo -e "${BOLD}${CYAN}"
echo -e "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo -e "┃                   POSE ESTIMATION EXAMPLES                  ┃"
echo -e "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo -e "${NC}"
echo -e "${YELLOW}The following examples demonstrate the pose estimation capabilities:${NC}"
echo
echo -e "${BOLD}${GREEN}1. Process an image:${NC}"
echo -e "   ${BLUE}./examples/scripts/run_pose_estimation.sh --input examples/sample_image.jpg${NC}"
echo
echo -e "${BOLD}${GREEN}2. Process a video:${NC}"
echo -e "   ${BLUE}./examples/scripts/run_pose_estimation.sh --input examples/sample_video.mp4${NC}"
echo
echo -e "${BOLD}${GREEN}3. Use webcam:${NC}"
echo -e "   ${BLUE}./examples/scripts/run_pose_estimation.sh --webcam${NC}"
echo
echo -e "${BOLD}${GREEN}4. Use RealSense camera (if available):${NC}"
echo -e "   ${BLUE}./examples/scripts/run_pose_estimation.sh --realsense${NC}"
echo
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo -e "${BOLD}${MAGENTA}Which example would you like to run? ${NC}${YELLOW}(1-4, or q to quit)${NC}"
read -r choice

case $choice in
    1)
        echo "Running image example..."
        bash "$WORKSPACE_DIR/src/scripts/run_pose_estimation.sh" --input "$WORKSPACE_DIR/examples/sample_image.jpg"
        ;;
    2)
        echo "Running video example..."
        bash "$WORKSPACE_DIR/src/scripts/run_pose_estimation.sh" --input "$WORKSPACE_DIR/examples/sample_video.mp4"
        ;;
    3)
        echo "Running webcam example..."
        bash "$WORKSPACE_DIR/src/scripts/run_pose_estimation.sh" --webcam
        ;;
    4)
        echo "Running RealSense camera example..."
        bash "$WORKSPACE_DIR/src/scripts/run_pose_estimation.sh" --realsense
        ;;
    q|Q)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac
