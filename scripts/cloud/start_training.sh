#!/usr/bin/env bash
#
# Launch StyleGAN2-ADA training on a Vast.ai instance.
# Training runs in a tmux session so it persists after disconnect.
#
# Usage:
#   ./start_training.sh <instance_id> [kimg] [snap] [extra_args]
#
# Examples:
#   ./start_training.sh 12345                    # Default: 5000 kimg
#   ./start_training.sh 12345 2000               # Short run: 2000 kimg
#   ./start_training.sh 12345 5000 10 "--gamma=10"  # Custom R1 weight

set -euo pipefail

INSTANCE_ID="${1:?Usage: start_training.sh <instance_id> [kimg] [snap] [extra_args]}"
KIMG="${2:-5000}"
SNAP="${3:-10}"
EXTRA="${4:-}"

echo "Starting StyleGAN2-ADA training"
echo "  Instance:  ${INSTANCE_ID}"
echo "  kimg:      ${KIMG}"
echo "  Snapshot:  every ${SNAP} ticks"
echo ""

vastai execute "${INSTANCE_ID}" "bash -c '
    # Kill any existing training session
    tmux kill-session -t train 2>/dev/null || true

    tmux new-session -d -s train \"bash -c \\\"
        cd /workspace/stylegan2-ada-pytorch && \\
        python train.py \\
            --outdir=/workspace/training-runs \\
            --data=/workspace/data/fractals512.zip \\
            --gpus=1 \\
            --cfg=auto \\
            --mirror=1 \\
            --snap=${SNAP} \\
            --metrics=fid50k_full \\
            --kimg=${KIMG} \\
            ${EXTRA} \\
        2>&1 | tee /workspace/training.log; \\
        echo \\\\\\\"Training complete. Press enter to close.\\\\\\\"; \\
        read
    \\\"\"

    echo \"Training launched in tmux session.\"
'" 2>&1

echo ""
echo "Training is running."
echo ""
echo "Monitor:"
echo "  ./monitor_training.sh ${INSTANCE_ID}"
echo ""
echo "Attach to training session:"
echo "  vastai ssh ${INSTANCE_ID} -c 'tmux attach -t train'"
echo ""
echo "Download checkpoint when done:"
echo "  ./download_checkpoint.sh ${INSTANCE_ID}"
