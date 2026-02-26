#!/usr/bin/env bash
#
# Search for cheapest RTX 4090 instances on Vast.ai suitable for StyleGAN2-ADA training.
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key <YOUR_API_KEY>
#
# Usage:
#   ./search_gpu.sh [gpu_name]

set -euo pipefail

GPU="${1:-RTX 4090}"

echo "Searching for ${GPU} instances on Vast.ai..."
echo "============================================="
echo ""

vastai search offers \
    "gpu_name == \"${GPU}\"" \
    "num_gpus == 1" \
    "cuda_vers >= 11.8" \
    "disk_space >= 50" \
    "inet_down >= 200" \
    "reliability > 0.95" \
    -o 'dph_total' \
    --limit 15 \
    --type on-demand

echo ""
echo "To create an instance:"
echo "  vastai create instance <ID> --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel --disk 80"
echo ""
echo "Or use setup_instance.sh <ID> to automate provisioning."
