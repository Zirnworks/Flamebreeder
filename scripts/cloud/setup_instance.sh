#!/usr/bin/env bash
#
# Provision a running Vast.ai instance for StyleGAN2-ADA training.
#
# Usage:
#   ./setup_instance.sh <ssh_host> <ssh_port>
#
# Example:
#   ./setup_instance.sh 69.162.73.55 1073
#
# This will:
#   1. Clone NVIDIA's stylegan2-ada-pytorch repo
#   2. Install all dependencies (ninja, pyspng, etc.)
#   3. Verify CUDA and StyleGAN2 imports

set -euo pipefail

SSH_HOST="${1:?Usage: setup_instance.sh <ssh_host> <ssh_port>}"
SSH_PORT="${2:?Usage: setup_instance.sh <ssh_host> <ssh_port>}"
SSH_CMD="ssh -p ${SSH_PORT} root@${SSH_HOST}"

echo "Provisioning instance at ${SSH_HOST}:${SSH_PORT}..."

${SSH_CMD} 'bash -s' <<'REMOTE'
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
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
for i in $(seq 0 $(($(nvidia-smi -L | wc -l) - 1))); do
    python -c "import torch; print(f'  GPU {'"$i"'}: {torch.cuda.get_device_name('"$i"')}')"
done

# Verify StyleGAN2 import
cd stylegan2-ada-pytorch
python -c "import dnnlib; print('dnnlib OK')"

echo ""
echo "=== Provisioning complete ==="
echo "NVIDIA repo: /workspace/stylegan2-ada-pytorch"
REMOTE

echo ""
echo "Instance provisioned."
echo "Next steps:"
echo "  1. Upload dataset:  ./upload_dataset.sh ${SSH_HOST} ${SSH_PORT} <data_dir>"
echo "  2. Start training:  ./start_training.sh ${SSH_HOST} ${SSH_PORT}"
