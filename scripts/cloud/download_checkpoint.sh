#!/usr/bin/env bash
#
# Download trained checkpoint from Vast.ai instance and convert to portable format.
#
# Usage:
#   ./download_checkpoint.sh <instance_id> [local_dest] [specific_pkl]
#
# Examples:
#   ./download_checkpoint.sh 12345
#   ./download_checkpoint.sh 12345 checkpoints/run2
#   ./download_checkpoint.sh 12345 checkpoints/run1 /workspace/training-runs/00000/network-snapshot-005000.pkl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INSTANCE_ID="${1:?Usage: download_checkpoint.sh <instance_id> [local_dest] [specific_pkl]}"
LOCAL_DEST="${2:-${PROJECT_DIR}/checkpoints/stylegan2}"
SPECIFIC_PKL="${3:-}"

mkdir -p "${LOCAL_DEST}"

# Find the checkpoint to download
if [ -z "${SPECIFIC_PKL}" ]; then
    echo "Finding latest checkpoint..."
    LATEST=$(vastai execute "${INSTANCE_ID}" "bash -c 'ls -t /workspace/training-runs/*/network-snapshot-*.pkl 2>/dev/null | head -1'" 2>/dev/null | tr -d '[:space:]')
    if [ -z "${LATEST}" ]; then
        echo "No checkpoints found on instance."
        exit 1
    fi
    echo "Latest: ${LATEST}"
else
    LATEST="${SPECIFIC_PKL}"
fi

PKL_NAME=$(basename "${LATEST}")
PT_NAME="${PKL_NAME%.pkl}_portable.pt"

# Upload conversion script and run it on the instance
echo "Converting checkpoint to portable format..."
vastai copy "${SCRIPT_DIR}/convert_checkpoint.py" "${INSTANCE_ID}:/workspace/"

vastai execute "${INSTANCE_ID}" "bash -c '
    cd /workspace/stylegan2-ada-pytorch
    python /workspace/convert_checkpoint.py \"${LATEST}\" \"/workspace/${PT_NAME}\"
'" 2>&1

# Download both files
echo ""
echo "Downloading checkpoint files..."
vastai copy "${INSTANCE_ID}:${LATEST}" "${LOCAL_DEST}/"
vastai copy "${INSTANCE_ID}:/workspace/${PT_NAME}" "${LOCAL_DEST}/"

# Also download some sample images if available
TRAINING_DIR=$(dirname "${LATEST}")
echo "Downloading sample images..."
vastai copy "${INSTANCE_ID}:${TRAINING_DIR}/fakes*.png" "${LOCAL_DEST}/" 2>/dev/null || true
vastai copy "${INSTANCE_ID}:${TRAINING_DIR}/reals.png" "${LOCAL_DEST}/" 2>/dev/null || true

echo ""
echo "Downloaded to ${LOCAL_DEST}/:"
ls -lh "${LOCAL_DEST}/"
echo ""
echo "Portable checkpoint: ${LOCAL_DEST}/${PT_NAME}"
echo "Use this path with the breeding server."
