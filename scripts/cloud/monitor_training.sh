#!/usr/bin/env bash
#
# Monitor StyleGAN2-ADA training progress on a Vast.ai instance.
#
# Usage:
#   ./monitor_training.sh <instance_id>

set -euo pipefail

INSTANCE_ID="${1:?Usage: monitor_training.sh <instance_id>}"

echo "=== Training Progress (Instance ${INSTANCE_ID}) ==="
echo ""

vastai execute "${INSTANCE_ID}" "bash -c '
    echo \"--- GPU Status ---\"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo \"GPU info unavailable\"
    echo \"\"

    echo \"--- Latest Log (last 25 lines) ---\"
    tail -25 /workspace/training.log 2>/dev/null || echo \"No training log yet\"
    echo \"\"

    echo \"--- Checkpoints ---\"
    ls -lt /workspace/training-runs/*/network-snapshot-*.pkl 2>/dev/null | head -5 || echo \"No checkpoints yet\"
    echo \"\"

    echo \"--- FID Scores ---\"
    for f in /workspace/training-runs/*/metric-fid50k_full.jsonl; do
        if [ -f \"\$f\" ]; then
            echo \"Latest FID entries:\"
            tail -5 \"\$f\"
        fi
    done 2>/dev/null || echo \"No FID data yet\"
    echo \"\"

    echo \"--- Disk Usage ---\"
    du -sh /workspace/training-runs/ 2>/dev/null || echo \"No training runs yet\"
'" 2>&1
