#!/bin/bash

# Script to run pose estimation

WORKSPACE_DIR="/home/phgnam/Workspace/VINMOTION/workspace"
CONFIG_FILE="$WORKSPACE_DIR/src/config.yaml"

# ANSI color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Display help with colors
function display_help() {
    echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║                   3D POSE ESTIMATION                       ║${NC}"
    echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${BOLD}Usage:${NC} $0 [OPTIONS]"
    echo -e "${YELLOW}Run pose estimation on images, videos, webcam or RealSense camera${NC}"
    echo
    echo -e "${BOLD}${GREEN}Options:${NC}"
    echo -e "  ${BOLD}-h, --help${NC}                 Display this help message"
    echo -e "  ${BOLD}-i, --input PATH${NC}           Path to input image or video file"
    echo -e "  ${BOLD}-w, --webcam${NC}               Use webcam as input"
    echo -e "  ${BOLD}-r, --realsense${NC}            Use RealSense camera as input"
    echo -e "  ${BOLD}-c, --config PATH${NC}          Path to config file (default: $CONFIG_FILE)"
    echo
    echo -e "${BOLD}${BLUE}Examples:${NC}"
    echo -e "  ${MAGENTA}$0 --input path/to/image.jpg${NC}"
    echo -e "  ${MAGENTA}$0 --input path/to/video.mp4${NC}"
    echo -e "  ${MAGENTA}$0 --webcam${NC}"
    echo -e "  ${MAGENTA}$0 --realsense${NC}"
    echo
}

# Check if any arguments are provided
if [ $# -eq 0 ]; then
    display_help
    exit 1
fi

# Parse arguments
INPUT=""
CONFIG="$CONFIG_FILE"

while [[ $# -gt 0 ]]; do
    key="$1"
    
    case $key in
        -h|--help)
            display_help
            exit 0
            ;;
        -i|--input)
            INPUT="$2"
            shift
            shift
            ;;
        -w|--webcam)
            INPUT="webcam"
            shift
            ;;
        -r|--realsense)
            INPUT="realsense"
            shift
            ;;
        -c|--config)
            CONFIG="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $key"
            display_help
            exit 1
            ;;
    esac
done

# Check if input is provided
if [ -z "$INPUT" ]; then
    echo "Error: No input specified"
    display_help
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Run the pose estimation script
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                STARTING POSE ESTIMATION                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "${YELLOW}Input:${NC} ${BOLD}$INPUT${NC}"
echo -e "${YELLOW}Config:${NC} ${BOLD}$CONFIG${NC}"
echo

python "$WORKSPACE_DIR/src/scripts/pose_estimation.py" --config "$CONFIG" --input "$INPUT"

# Display completion message
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                COMPLETED SUCCESSFULLY                      ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
else
    echo
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║               COMPLETED WITH ERRORS                       ║${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════╝${NC}"
fi
