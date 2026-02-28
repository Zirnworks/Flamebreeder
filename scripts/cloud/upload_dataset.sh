#!/usr/bin/env bash
#
# Upload dataset to Vast.ai instance and convert to NVIDIA ZIP format.
#
# Usage:
#   ./upload_dataset.sh <ssh_host> <ssh_port> [local_data_dir] [resolution]
#
# Example:
#   ./upload_dataset.sh 69.162.73.55 1073 ~/Data/Praeceptor/data/consolidated 512

set -euo pipefail

SSH_HOST="${1:?Usage: upload_dataset.sh <ssh_host> <ssh_port> [local_data_dir] [resolution]}"
SSH_PORT="${2:?Usage: upload_dataset.sh <ssh_host> <ssh_port> [local_data_dir] [resolution]}"
LOCAL_DATA="${3:?Usage: upload_dataset.sh <ssh_host> <ssh_port> <local_data_dir> [resolution]}"
RESOLUTION="${4:-512}"

SSH_CMD="ssh -p ${SSH_PORT} root@${SSH_HOST}"

# Verify local data exists
if [ ! -d "${LOCAL_DATA}" ]; then
    echo "Error: Dataset directory not found: ${LOCAL_DATA}"
    exit 1
fi

IMAGE_COUNT=$(find "${LOCAL_DATA}" -name "*.png" | wc -l | tr -d ' ')
echo "Uploading ${IMAGE_COUNT} images from ${LOCAL_DATA}..."
echo "Destination: ${SSH_HOST}:/workspace/data/consolidated/"

# Upload PNGs and dataset.json via rsync
rsync -avP --include='*.png' --include='dataset.json' --exclude='*' \
    -e "ssh -p ${SSH_PORT}" \
    "${LOCAL_DATA}/" \
    "root@${SSH_HOST}:/workspace/data/consolidated/"

echo ""
echo "Upload complete. Converting to NVIDIA ZIP format..."

# Convert to NVIDIA format on the instance
${SSH_CMD} "bash -s" <<REMOTE
set -ex
cd /workspace/stylegan2-ada-pytorch

python dataset_tool.py \
    --source=/workspace/data/consolidated \
    --dest=/workspace/data/fractals${RESOLUTION}.zip \
    --resolution=${RESOLUTION}x${RESOLUTION}

echo ""
echo "Dataset ready: /workspace/data/fractals${RESOLUTION}.zip"
python -c "
import zipfile
z = zipfile.ZipFile('/workspace/data/fractals${RESOLUTION}.zip')
pngs = [n for n in z.namelist() if n.endswith('.png')]
jsons = [n for n in z.namelist() if n.endswith('.json')]
print(f'Images in ZIP: {len(pngs)}')
print(f'JSON files: {len(jsons)}')
"
REMOTE

echo ""
echo "Dataset uploaded and converted."
echo "Next: ./start_training.sh ${SSH_HOST} ${SSH_PORT}"
