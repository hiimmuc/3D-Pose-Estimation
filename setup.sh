#!/bin/bash
# setup.sh - Setup script for 3D Pose Estimation System
# This script installs all required dependencies and downloads model checkpoints

# ANSI color codes for nice output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${GREEN}${BOLD}Setting up 3D Pose Estimation System${NC}"
echo -e "${BLUE}Installing Python dependencies...${NC}"

# Install basic dependencies
pip install -r requirements.txt

# Install MMEngine and related packages
echo -e "\n${BLUE}Installing MMEngine and related packages...${NC}"
pip install -U openmim
mim install mmengine
mim install mmdet
mim install mmpose

# Download model checkpoints
echo -e "\n${BLUE}Downloading model checkpoints...${NC}"
bash src/scripts/download_checkpoints.sh

# Optional: Install PyTorch with CUDA support if not already installed
if ! python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA not available')" | grep -q "CUDA available"; then
    echo -e "\n${YELLOW}${BOLD}CUDA not detected in your PyTorch installation.${NC}"
    echo -e "${YELLOW}For better performance, consider installing PyTorch with CUDA support:${NC}"
    echo -e "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
fi

# Optional: Check for RealSense SDK
if ! python -c "import pyrealsense2" &> /dev/null; then
    echo -e "\n${YELLOW}Intel RealSense SDK not detected.${NC}"
    echo -e "${YELLOW}If you want to use RealSense cameras, install the SDK:${NC}"
    echo -e "  pip install pyrealsense2"
fi

echo -e "\n${GREEN}${BOLD}Setup complete!${NC}"
echo -e "${GREEN}You can now run the pose estimation system.${NC}"
echo -e "Example: ${BOLD}python src/scripts/main.py --webcam --show${NC}"
