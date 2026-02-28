#!/usr/bin/env bash
#
# Monitor StyleGAN2-ADA training progress on a Vast.ai instance.
#
# Usage:
#   ./monitor_training.sh <ssh_host> <ssh_port>

set -euo pipefail

SSH_HOST="${1:?Usage: monitor_training.sh <ssh_host> <ssh_port>}"
SSH_PORT="${2:?Usage: monitor_training.sh <ssh_host> <ssh_port>}"

SSH_CMD="ssh -p ${SSH_PORT} root@${SSH_HOST}"

echo "=== Training Progress (${SSH_HOST}:${SSH_PORT}) ==="
echo ""

${SSH_CMD} "bash -s" <<'REMOTE'
echo "--- GPU Status ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null || echo "GPU info unavailable"
echo ""

echo "--- Latest Log (last 25 lines) ---"
tail -25 /workspace/training.log 2>/dev/null || echo "No training log yet"
echo ""

echo "--- Checkpoints ---"
ls -lt /workspace/training-runs/*/network-snapshot-*.pkl 2>/dev/null | head -5 || echo "No checkpoints yet"
echo ""

echo "--- FID Scores ---"
for f in /workspace/training-runs/*/metric-fid50k_full.jsonl; do
    if [ -f "$f" ]; then
        echo "Latest FID entries:"
        tail -5 "$f"
    fi
done 2>/dev/null || echo "No FID data yet"
echo ""

echo "--- Disk Usage ---"
du -sh /workspace/training-runs/ 2>/dev/null || echo "No training runs yet"
du -sh /workspace/data/ 2>/dev/null || echo "No data directory"
REMOTE
