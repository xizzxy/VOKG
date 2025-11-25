#!/bin/bash
# Download SAM 2 models

set -e

echo "Creating models directory..."
mkdir -p models

echo "Downloading SAM 2 Hiera Large checkpoint..."
wget -O models/sam2_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt

echo "Downloading SAM 2 config..."
wget -O models/sam2_hiera_l.yaml \
  https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2_configs/sam2_hiera_l.yaml

echo "SAM 2 models downloaded successfully!"
echo ""
echo "Available models:"
ls -lh models/

echo ""
echo "Update your .env file with:"
echo "SAM_CHECKPOINT_PATH=/models/sam2_hiera_large.pt"
echo "SAM_CONFIG_PATH=/models/sam2_hiera_l.yaml"
