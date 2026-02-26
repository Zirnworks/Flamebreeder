#!/usr/bin/env bash
#
# Create and provision a Vast.ai instance for StyleGAN2-ADA training.
#
# Usage:
#   ./setup_instance.sh <offer_id>
#
# This will:
#   1. Create the instance from the offer
#   2. Wait for SSH availability
#   3. Clone NVIDIA's stylegan2-ada-pytorch repo
#   4. Install all dependencies
#   5. Print SSH connection info

set -euo pipefail

OFFER_ID="${1:?Usage: setup_instance.sh <offer_id>}"
IMAGE="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel"
DISK_GB=80

echo "Creating instance from offer ${OFFER_ID}..."
INSTANCE_ID=$(vastai create instance "${OFFER_ID}" \
    --image "${IMAGE}" \
    --disk "${DISK_GB}" \
    --onstart-cmd "touch /workspace/.ready" \
    2>&1 | grep -oP 'new instance \K\d+' || true)

if [ -z "${INSTANCE_ID}" ]; then
    echo "Failed to create instance. Check offer ID and try again."
    exit 1
fi

echo "Instance created: ${INSTANCE_ID}"
echo "Waiting for instance to start..."

# Wait for ready (up to 5 minutes)
for i in $(seq 1 60); do
    STATUS=$(vastai show instance "${INSTANCE_ID}" --raw 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['actual_status'])" 2>/dev/null || echo "unknown")
    if [ "${STATUS}" = "running" ]; then
        echo "Instance is running."
        break
    fi
    printf "."
    sleep 5
done
echo ""

# Get SSH info
echo "Getting SSH connection info..."
SSH_URL=$(vastai ssh-url "${INSTANCE_ID}" 2>/dev/null || echo "")
echo "SSH: ${SSH_URL}"

# Provision the instance
echo ""
echo "Provisioning instance..."
vastai execute "${INSTANCE_ID}" "bash -c '
    set -ex

    # System packages
    apt-get update -qq
    apt-get install -y -qq tmux htop rsync > /dev/null 2>&1

    # Clone NVIDIA StyleGAN2-ADA
    cd /workspace
    if [ ! -d stylegan2-ada-pytorch ]; then
        git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
    fi

    # Python dependencies
    pip install -q click requests tqdm pyspng ninja scipy Pillow psutil imageio-ffmpeg==0.4.3

    # Verify CUDA
    python -c \"import torch; print(f\\\"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}\\\")\"

    # Verify StyleGAN2 import
    cd stylegan2-ada-pytorch
    python -c \"import dnnlib; print(\\\"dnnlib OK\\\")\"

    echo \"\"
    echo \"=== Provisioning complete ===\"
    echo \"NVIDIA repo: /workspace/stylegan2-ada-pytorch\"
'" 2>&1

echo ""
echo "============================================="
echo "Instance ${INSTANCE_ID} is ready."
echo "SSH: ${SSH_URL}"
echo ""
echo "Next steps:"
echo "  1. Upload dataset:  ./upload_dataset.sh ${INSTANCE_ID}"
echo "  2. Start training:  ./start_training.sh ${INSTANCE_ID}"
echo ""
echo "Save this instance ID: ${INSTANCE_ID}"
