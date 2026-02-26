#!/usr/bin/env bash
#
# Upload dataset to Vast.ai instance and convert to NVIDIA ZIP format.
#
# Usage:
#   ./upload_dataset.sh <instance_id> [local_data_dir] [resolution]
#
# Example:
#   ./upload_dataset.sh 12345 ../datagen/data/processed/train 512

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTANCE_ID="${1:?Usage: upload_dataset.sh <instance_id> [local_data_dir] [resolution]}"
LOCAL_DATA="${2:-${PROJECT_DIR}/datagen/data/processed/train}"
RESOLUTION="${3:-512}"

# Verify local data exists
if [ ! -d "${LOCAL_DATA}" ]; then
    echo "Error: Dataset directory not found: ${LOCAL_DATA}"
    exit 1
fi

IMAGE_COUNT=$(find "${LOCAL_DATA}" -name "*.png" | wc -l | tr -d ' ')
echo "Uploading ${IMAGE_COUNT} images from ${LOCAL_DATA}..."

# Upload dataset
vastai copy "${LOCAL_DATA}/" "${INSTANCE_ID}:/workspace/data/pngs/"

echo "Upload complete. Converting to NVIDIA ZIP format..."

# Convert to NVIDIA format on the instance
vastai execute "${INSTANCE_ID}" "bash -c '
    set -ex
    cd /workspace/stylegan2-ada-pytorch

    python dataset_tool.py \
        --source=/workspace/data/pngs \
        --dest=/workspace/data/fractals${RESOLUTION}.zip \
        --resolution=${RESOLUTION}x${RESOLUTION}

    echo \"\"
    echo \"Dataset ready: /workspace/data/fractals${RESOLUTION}.zip\"
    python -c \"
import zipfile
z = zipfile.ZipFile(\\\"/workspace/data/fractals${RESOLUTION}.zip\\\")
pngs = [n for n in z.namelist() if n.endswith(\\\".png\\\")]
print(f\\\"Images in ZIP: {len(pngs)}\\\")
\"
'" 2>&1

echo ""
echo "Dataset uploaded and converted."
echo "Next: ./start_training.sh ${INSTANCE_ID}"
