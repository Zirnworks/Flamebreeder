#!/usr/bin/env bash
#
# Launch StyleGAN2-ADA training on a Vast.ai instance.
# Training runs in a tmux session so it persists after disconnect.
#
# Usage:
#   ./start_training.sh <ssh_host> <ssh_port> [kimg] [snap] [extra_args]
#
# Examples:
#   ./start_training.sh 69.162.73.55 1073                     # Default: 10000 kimg
#   ./start_training.sh 69.162.73.55 1073 2000                # Short run: 2000 kimg
#   ./start_training.sh 69.162.73.55 1073 10000 10 "--gamma=10"

set -euo pipefail

SSH_HOST="${1:?Usage: start_training.sh <ssh_host> <ssh_port> [kimg] [snap] [extra_args]}"
SSH_PORT="${2:?Usage: start_training.sh <ssh_host> <ssh_port> [kimg] [snap] [extra_args]}"
KIMG="${3:-10000}"
SNAP="${4:-10}"
EXTRA="${5:-}"

SSH_CMD="ssh -p ${SSH_PORT} root@${SSH_HOST}"

echo "Starting StyleGAN2-ADA training"
echo "  Host:      ${SSH_HOST}:${SSH_PORT}"
echo "  kimg:      ${KIMG}"
echo "  Snapshot:  every ${SNAP} ticks"
echo "  Extra:     ${EXTRA:-none}"
echo ""

${SSH_CMD} "bash -s" <<REMOTE
# Kill any existing training session
tmux kill-session -t train 2>/dev/null || true

# Launch training in tmux
tmux new-session -d -s train "bash -c '
    cd /workspace/stylegan2-ada-pytorch && \
    python train.py \
        --outdir=/workspace/training-runs \
        --data=/workspace/data/fractals512.zip \
        --gpus=2 \
        --cond=1 \
        --cfg=auto \
        --mirror=0 \
        --snap=${SNAP} \
        --snap-max=3 \
        --metrics=fid50k_full \
        --kimg=${KIMG} \
        ${EXTRA} \
    2>&1 | tee /workspace/training.log; \
    echo \"Training complete. Press enter to close.\"; \
    read
'"

echo "Training launched in tmux session 'train'."
REMOTE

echo ""
echo "Training is running."
echo ""
echo "Monitor:"
echo "  ./monitor_training.sh ${SSH_HOST} ${SSH_PORT}"
echo ""
echo "Attach to training session:"
echo "  ssh -p ${SSH_PORT} root@${SSH_HOST} -t 'tmux attach -t train'"
echo ""
echo "Download checkpoint when done:"
echo "  ./download_checkpoint.sh ${SSH_HOST} ${SSH_PORT}"
