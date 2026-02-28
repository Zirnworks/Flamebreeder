#!/usr/bin/env bash
#
# Download trained checkpoint from Vast.ai instance and convert to portable format.
#
# Usage:
#   ./download_checkpoint.sh <ssh_host> <ssh_port> [local_dest] [specific_pkl]
#
# Examples:
#   ./download_checkpoint.sh 69.162.73.55 1073
#   ./download_checkpoint.sh 69.162.73.55 1073 checkpoints/run2
#   ./download_checkpoint.sh 69.162.73.55 1073 checkpoints/run1 /workspace/training-runs/00000/network-snapshot-005000.pkl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SSH_HOST="${1:?Usage: download_checkpoint.sh <ssh_host> <ssh_port> [local_dest] [specific_pkl]}"
SSH_PORT="${2:?Usage: download_checkpoint.sh <ssh_host> <ssh_port> [local_dest] [specific_pkl]}"
LOCAL_DEST="${3:-${PROJECT_DIR}/checkpoints/stylegan2}"
SPECIFIC_PKL="${4:-}"

SSH_CMD="ssh -p ${SSH_PORT} root@${SSH_HOST}"

mkdir -p "${LOCAL_DEST}"

# Find the checkpoint to download
if [ -z "${SPECIFIC_PKL}" ]; then
    echo "Finding latest checkpoint..."
    LATEST=$(${SSH_CMD} 'ls -t /workspace/training-runs/*/network-snapshot-*.pkl 2>/dev/null | head -1' | tr -d '[:space:]')
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
echo "Uploading conversion script..."
rsync -avP -e "ssh -p ${SSH_PORT}" \
    "${SCRIPT_DIR}/convert_checkpoint.py" \
    "root@${SSH_HOST}:/workspace/"

echo "Converting checkpoint to portable format..."
${SSH_CMD} "bash -s" <<REMOTE
cd /workspace/stylegan2-ada-pytorch
python /workspace/convert_checkpoint.py "${LATEST}" "/workspace/${PT_NAME}"
REMOTE

# Download both files
echo ""
echo "Downloading checkpoint files..."
rsync -avP -e "ssh -p ${SSH_PORT}" \
    "root@${SSH_HOST}:${LATEST}" \
    "${LOCAL_DEST}/"

rsync -avP -e "ssh -p ${SSH_PORT}" \
    "root@${SSH_HOST}:/workspace/${PT_NAME}" \
    "${LOCAL_DEST}/"

# Also download some sample images if available
TRAINING_DIR=$(dirname "${LATEST}")
echo "Downloading sample images..."
rsync -avP -e "ssh -p ${SSH_PORT}" \
    "root@${SSH_HOST}:${TRAINING_DIR}/fakes*.png" \
    "${LOCAL_DEST}/" 2>/dev/null || true
rsync -avP -e "ssh -p ${SSH_PORT}" \
    "root@${SSH_HOST}:${TRAINING_DIR}/reals.png" \
    "${LOCAL_DEST}/" 2>/dev/null || true

echo ""
echo "Downloaded to ${LOCAL_DEST}/:"
ls -lh "${LOCAL_DEST}/"
echo ""
echo "Portable checkpoint: ${LOCAL_DEST}/${PT_NAME}"
echo "Use this path with the breeding server."
