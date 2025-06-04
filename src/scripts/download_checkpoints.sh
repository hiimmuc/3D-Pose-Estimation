#!/bin/bash

# Script to download model checkpoints for pose estimation and detection using mim

WORKSPACE_DIR="/home/phgnam/Workspace/VINMOTION/workspace"
CONFIG_FILE="$WORKSPACE_DIR/src/config.yaml"

# Print usage
function print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --config PATH   Path to config.yaml file (default: $CONFIG_FILE)"
    echo "  --verbose       Enable verbose output for debugging"
    echo "  --help          Show this help message"
}

# Parse command line arguments
VERBOSE=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift # past argument
      shift # past value
      ;;
    --verbose)
      VERBOSE=true
      shift # past argument
      ;;
    --help)
      print_usage
      exit 0
      ;;
    *)
      # Unknown option
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Function for debug logging
debug_log() {
  if [[ "$VERBOSE" == "true" ]]; then
    echo -e "${BLUE}[DEBUG] $1${NC}"
  fi
}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error:${NC} Config file not found: $CONFIG_FILE"
    exit 1
fi

echo -e "${GREEN}Using config file:${NC} $CONFIG_FILE"
POSE_CHECKPOINT_DIR="$WORKSPACE_DIR/checkpoints/pose-estimation"
DETECTION_CHECKPOINT_DIR="$WORKSPACE_DIR/checkpoints/detection"

# ANSI color codes for nice output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Function to parse YAML files
function parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\([^#]*\)[#].*$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         gsub(/[ \t]+$/, "", $3); # Remove trailing whitespace
         printf("%s%s=\"%s\"\n", "'$prefix'",vn$2, $3);
      }
   }'
}

# Check if mim is installed
if ! command -v mim &> /dev/null; then
    echo "Error: 'mim' command not found. Please install openmim first:"
    echo "pip install openmim"
    exit 1
fi

echo "Parsing config file or using defaults..."

# Try to parse config file
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${GREEN}Parsing config file: ${CONFIG_FILE}${NC}"
    # Extract the detection and pose config values
    detection_config=$(grep -A5 "detection:" "$CONFIG_FILE" | grep "config:" | awk -F ': ' '{print $2}' | tr -d ' ')
    detection_checkpoint=$(grep -A5 "detection:" "$CONFIG_FILE" | grep "checkpoint:" | awk -F ': ' '{print $2}' | tr -d ' ')
    pose_config=$(grep -A5 "pose:" "$CONFIG_FILE" | grep "config:" | awk -F ': ' '{print $2}' | tr -d ' ')
    pose_checkpoint=$(grep -A5 "pose:" "$CONFIG_FILE" | grep "checkpoint:" | awk -F ': ' '{print $2}' | tr -d ' ')
    
    # Validate that we got all the required values
    if [ -z "$detection_config" ] || [ -z "$detection_checkpoint" ] || [ -z "$pose_config" ] || [ -z "$pose_checkpoint" ]; then
        echo -e "${YELLOW}Warning: Could not extract all required values from config file.${NC}"
        # Use the parse_yaml function as backup
        eval $(parse_yaml $CONFIG_FILE) 2>/dev/null || true
    fi
else
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Ensure we have the required values (no defaults)
if [ -z "$detection_config" ] || [ -z "$detection_checkpoint" ] || [ -z "$pose_config" ] || [ -z "$pose_checkpoint" ]; then
    echo -e "${RED}Error: Missing required configuration values in $CONFIG_FILE${NC}"
    echo -e "Required fields: detection.config, detection.checkpoint, pose.config, pose.checkpoint"
    exit 1
fi

# Print the config values being used
echo -e "${BLUE}Using the following configuration:${NC}"
echo -e "Detection config: ${YELLOW}$detection_config${NC}"
echo -e "Detection checkpoint: ${YELLOW}$detection_checkpoint${NC}"
echo -e "Pose config: ${YELLOW}$pose_config${NC}"
echo -e "Pose checkpoint: ${YELLOW}$pose_checkpoint${NC}"

# Create directories if they don't exist
mkdir -p $POSE_CHECKPOINT_DIR
mkdir -p $DETECTION_CHECKPOINT_DIR

# Check for each file separately and determine what to download
detection_config_path="$DETECTION_CHECKPOINT_DIR/$detection_config/$detection_config.py"
detection_checkpoint_path="$DETECTION_CHECKPOINT_DIR/$detection_checkpoint"
pose_config_path="$POSE_CHECKPOINT_DIR/$pose_config/$pose_config.py"
pose_checkpoint_path="$POSE_CHECKPOINT_DIR/$pose_checkpoint"

# Check old path structure for backward compatibility
old_detection_config="$WORKSPACE_DIR/src/detection/configs/${detection_config}.py"
old_pose_config="$WORKSPACE_DIR/src/pose-estimation/configs/${pose_config}.py"

# Initialize download flags to false
download_detection_config=false
download_detection_checkpoint=false
download_pose_config=false
download_pose_checkpoint=false

# Check each file
if [[ ! -f "$detection_config_path" && ! -f "$old_detection_config" ]]; then
    echo -e "${YELLOW}Missing detection config: ${detection_config}.py${NC}"
    debug_log "Detection config paths checked: $detection_config_path and $old_detection_config"
    download_detection_config=true
else
    echo -e "${GREEN}Detection config exists: ${detection_config}.py${NC}"
    if [[ -f "$detection_config_path" ]]; then
        debug_log "Found at new location: $detection_config_path"
    fi
    if [[ -f "$old_detection_config" ]]; then
        debug_log "Found at old location: $old_detection_config"
    fi
fi

if [[ ! -f "$detection_checkpoint_path" ]]; then
    echo -e "${YELLOW}Missing detection checkpoint: ${detection_checkpoint}${NC}"
    debug_log "Detection checkpoint path checked: $detection_checkpoint_path"
    download_detection_checkpoint=true
else
    echo -e "${GREEN}Detection checkpoint exists: ${detection_checkpoint}${NC}"
    debug_log "Found at: $detection_checkpoint_path"
fi

if [[ ! -f "$pose_config_path" && ! -f "$old_pose_config" ]]; then
    echo -e "${YELLOW}Missing pose config: ${pose_config}.py${NC}"
    debug_log "Pose config paths checked: $pose_config_path and $old_pose_config"
    download_pose_config=true
else
    echo -e "${GREEN}Pose config exists: ${pose_config}.py${NC}"
    if [[ -f "$pose_config_path" ]]; then
        debug_log "Found at new location: $pose_config_path"
    fi
    if [[ -f "$old_pose_config" ]]; then
        debug_log "Found at old location: $old_pose_config"
    fi
fi

if [[ ! -f "$pose_checkpoint_path" ]]; then
    echo -e "${YELLOW}Missing pose checkpoint: ${pose_checkpoint}${NC}"
    debug_log "Pose checkpoint path checked: $pose_checkpoint_path"
    download_pose_checkpoint=true
else
    echo -e "${GREEN}Pose checkpoint exists: ${pose_checkpoint}${NC}"
    debug_log "Found at: $pose_checkpoint_path"
fi

# Check if any downloads are needed
if [[ "$download_detection_config" == "true" || "$download_detection_checkpoint" == "true" || 
      "$download_pose_config" == "true" || "$download_pose_checkpoint" == "true" ]]; then
    echo -e "\n${YELLOW}${BOLD}Some model files are missing. Starting download process...${NC}"
else
    echo -e "\n${GREEN}${BOLD}All model files already exist. No downloads needed.${NC}"
    exit 0
fi

# Only download detection files if needed
if [[ "$download_detection_config" == "true" || "$download_detection_checkpoint" == "true" ]]; then
    echo -e "\n${BLUE}${BOLD}Downloading detection files for: $detection_config${NC}"
    DETECTION_DEST_DIR="$DETECTION_CHECKPOINT_DIR/$detection_config"
    mkdir -p "$DETECTION_DEST_DIR"
    
    echo -e "${BLUE}${BOLD}Running:${NC} mim download mmdet --config $detection_config --dest $DETECTION_DEST_DIR"
    mim download mmdet --config "$detection_config" --dest "$DETECTION_DEST_DIR" || {
        echo -e "${YELLOW}${BOLD}Warning:${NC} mim download failed for detection config. Trying alternative method..."
    }
else
    echo -e "\n${GREEN}${BOLD}Skipping detection files download (already exist)${NC}"
fi

# Copy or link the config and checkpoint files to expected locations
DOWNLOADED_CONFIG=$(find "$DETECTION_DEST_DIR" -name "*.py" | head -n 1)
DOWNLOADED_CHECKPOINT=$(find "$DETECTION_DEST_DIR" -name "*.pth" | head -n 1)

if [ -n "$DOWNLOADED_CONFIG" ] && [ -n "$DOWNLOADED_CHECKPOINT" ]; then
    echo -e "${GREEN}${BOLD}✓ Successfully downloaded detection files:${NC}"
    echo -e "  Config: ${BLUE}$DOWNLOADED_CONFIG${NC}"
    echo -e "  Checkpoint: ${BLUE}$DOWNLOADED_CHECKPOINT${NC}"
    
    # Ensure the detection config directory exists
    mkdir -p "$WORKSPACE_DIR/src/detection/configs"
    
    # Create a copy of the config file in the expected location
    cp "$DOWNLOADED_CONFIG" "$WORKSPACE_DIR/src/detection/configs/${detection_config}.py"
    
    # Create a symlink to the checkpoint with the expected name
    ln -sf "$DOWNLOADED_CHECKPOINT" "$DETECTION_CHECKPOINT_DIR/${detection_checkpoint}"
else
    echo -e "${YELLOW}${BOLD}❌ Failed to download detection files using mim. Trying alternative download method...${NC}"
    
    # Try to download directly from mmdetection github repo
    echo "Downloading detection config from GitHub..."
    MMDET_CONFIG_URL="https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/${detection_config}.py"
    curl -s -o "$WORKSPACE_DIR/src/detection/configs/${detection_config}.py" "$MMDET_CONFIG_URL" || {
        echo "❌ Failed to download config file from GitHub."
    }
    
    # Try to get checkpoint URL from config file
    if [ -f "$WORKSPACE_DIR/src/detection/configs/${detection_config}.py" ]; then
        CHECKPOINT_URL=$(grep -A 10 "Weights:" "$WORKSPACE_DIR/src/detection/configs/${detection_config}.py" | grep "http" | head -n 1 | awk -F'"' '{print $2}')
    fi
    
    if [ -z "$CHECKPOINT_URL" ]; then
        echo -e "${RED}✗ Error: Could not determine checkpoint URL from config file.${NC}"
        echo -e "${RED}  This is likely because the checkpoint is not available in the standard location.${NC}"
        echo -e "${RED}  Please manually download the checkpoint for ${detection_config}.${NC}"
    fi
    
    echo "Downloading detection checkpoint from: $CHECKPOINT_URL"
    wget -c -O "$DETECTION_CHECKPOINT_DIR/${detection_checkpoint}" "$CHECKPOINT_URL" || {
        echo "❌ Failed to download checkpoint file directly."
    }
fi

# Only download pose files if needed
if [[ "$download_pose_config" == "true" || "$download_pose_checkpoint" == "true" ]]; then
    echo -e "\n${BLUE}${BOLD}Downloading pose files for: $pose_config${NC}"
    POSE_DEST_DIR="$POSE_CHECKPOINT_DIR/$pose_config"
    mkdir -p "$POSE_DEST_DIR"
    
    echo -e "${BLUE}${BOLD}Running:${NC} mim download mmpose --config $pose_config --dest $POSE_DEST_DIR"
    mim download mmpose --config "$pose_config" --dest "$POSE_DEST_DIR" || {
        echo -e "${YELLOW}${BOLD}Warning:${NC} mim download failed for pose config. Trying alternative method..."
    }
else
    echo -e "\n${GREEN}${BOLD}Skipping pose files download (already exist)${NC}"
fi

# Copy or link the config and checkpoint files to expected locations
DOWNLOADED_CONFIG=$(find "$POSE_DEST_DIR" -name "*.py" | head -n 1)
DOWNLOADED_CHECKPOINT=$(find "$POSE_DEST_DIR" -name "*.pth" | head -n 1)

if [ -n "$DOWNLOADED_CONFIG" ] && [ -n "$DOWNLOADED_CHECKPOINT" ]; then
    echo -e "${GREEN}${BOLD}✓ Successfully downloaded pose files:${NC}"
    echo -e "  Config: ${BLUE}$DOWNLOADED_CONFIG${NC}"
    echo -e "  Checkpoint: ${BLUE}$DOWNLOADED_CHECKPOINT${NC}"
    
    # Ensure the pose config directory exists
    mkdir -p "$WORKSPACE_DIR/src/pose-estimation/configs"
    
    # Create a copy of the config file in the expected location
    cp "$DOWNLOADED_CONFIG" "$WORKSPACE_DIR/src/pose-estimation/configs/${pose_config}.py"
    
    # Create a symlink to the checkpoint with the expected name
    ln -sf "$DOWNLOADED_CHECKPOINT" "$POSE_CHECKPOINT_DIR/${pose_checkpoint}"
else
    echo -e "${YELLOW}${BOLD}❌ Failed to download pose files using mim. Trying alternative download method...${NC}"
    
    # Try to download directly from mmpose github repo
    echo "Downloading pose config from GitHub..."
    # Determine likely path structure in the repo
    if [[ "$pose_config" == *"rtmpose"* ]]; then
        CONFIG_PATH="body_2d_keypoint/rtmpose/coco/${pose_config}.py"
    else
        CONFIG_PATH="body_2d_keypoint/${pose_config}.py"
    fi
    
    mkdir -p "$WORKSPACE_DIR/src/pose-estimation/configs/$(dirname "$CONFIG_PATH")"
    
    MMPOSE_CONFIG_URL="https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/$CONFIG_PATH"
    curl -s -o "$WORKSPACE_DIR/src/pose-estimation/configs/${CONFIG_PATH}" "$MMPOSE_CONFIG_URL" || {
        echo "❌ Failed to download config file from GitHub."
    }
    
    # Try to get checkpoint URL from config file only
    if [ -f "$WORKSPACE_DIR/src/pose-estimation/configs/${CONFIG_PATH}" ]; then
        CHECKPOINT_URL=$(grep -A 10 "Weights:" "$WORKSPACE_DIR/src/pose-estimation/configs/${CONFIG_PATH}" | grep "http" | head -n 1 | awk -F'"' '{print $2}')
    fi
    
    if [ -z "$CHECKPOINT_URL" ]; then
        echo -e "${RED}✗ Error: Could not determine pose checkpoint URL from config file.${NC}"
        echo -e "${RED}  This is likely because the checkpoint is not available in the standard location.${NC}"
        echo -e "${RED}  Please manually download the checkpoint for ${pose_config}.${NC}"
    fi
    
    echo "Downloading pose checkpoint from: $CHECKPOINT_URL"
    wget -c -O "$POSE_CHECKPOINT_DIR/${pose_checkpoint}" "$CHECKPOINT_URL" || {
        echo "❌ Failed to download checkpoint file directly."
    }
fi

echo -e "${GREEN}${BOLD}✓ Done downloading checkpoints.${NC}"

# Define a function to check if all files are present
function check_all_files_exist() {
    local missing=false
    
    # Check detection files
    if [[ ! -f "$DETECTION_CHECKPOINT_DIR/$detection_config/$detection_config.py" && 
          ! -f "$WORKSPACE_DIR/src/detection/configs/${detection_config}.py" ]]; then
        echo -e "${YELLOW}Still missing detection config: ${detection_config}.py${NC}"
        missing=true
    fi
    
    if [[ ! -f "$DETECTION_CHECKPOINT_DIR/${detection_checkpoint}" ]]; then
        echo -e "${YELLOW}Still missing detection checkpoint: ${detection_checkpoint}${NC}"
        missing=true
    fi
    
    # Check pose files
    if [[ ! -f "$POSE_CHECKPOINT_DIR/$pose_config/$pose_config.py" && 
          ! -f "$WORKSPACE_DIR/src/pose-estimation/configs/${pose_config}.py" ]]; then
        echo -e "${YELLOW}Still missing pose config: ${pose_config}.py${NC}"
        missing=true
    fi
    
    if [[ ! -f "$POSE_CHECKPOINT_DIR/${pose_checkpoint}" ]]; then
        echo -e "${YELLOW}Still missing pose checkpoint: ${pose_checkpoint}${NC}"
        missing=true
    fi
    
    # Return result
    if $missing; then
        return 1  # Files are missing
    else
        return 0  # All files exist
    fi
}

# Verify that all required files are now present
if ! check_all_files_exist; then
    echo -e "${RED}${BOLD}❌ Some model files are still missing after download attempts.${NC}"
    echo -e "${YELLOW}Please check your internet connection and try again manually.${NC}"
    exit 1
else
    echo -e "${GREEN}${BOLD}✓ All model files have been successfully downloaded.${NC}"
    exit 0
fi
