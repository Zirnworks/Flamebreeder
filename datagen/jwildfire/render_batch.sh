#!/usr/bin/env bash
#
# Batch render fractal flames using JWildfire CLI.
#
# Usage:
#   ./render_batch.sh [total_images] [output_dir] [quality] [workers]
#
# Example:
#   ./render_batch.sh 15000 ../data/raw_jwildfire 300 8

set -euo pipefail

TOTAL=${1:-15000}
OUTPUT_DIR=${2:-../data/raw_jwildfire}
QUALITY=${3:-300}
WORKERS=${4:-8}

JAVA_HOME="/opt/homebrew/opt/openjdk@21"
JAVA="$JAVA_HOME/bin/java"
JWILDFIRE_LIB="$(cd "$(dirname "$0")/lib/lib" && pwd)"
WIDTH=512
HEIGHT=512

# Flame generators to cycle through (selected for visual quality and variety)
GENERATORS=(
    "(All)"
    "Bubbles"
    "Flowers3D (stunning)"
    "Galaxies"
    "Gnarl"
    "Julians"
    "Orchids"
    "Phoenix"
    "Sierpinsky"
    "Simple (stunning)"
    "Spirals"
    "Spherical"
    "Tentacle"
    "Bokeh"
    "Brokat"
    "Cross"
    "Duality"
    "JulianDisc"
    "Layers"
    "Machine"
    "Mandelbrot"
    "Raster"
    "Rays"
    "Splits"
    "Synth"
)

NUM_GENERATORS=${#GENERATORS[@]}

mkdir -p "$OUTPUT_DIR"

echo "JWildfire Batch Renderer"
echo "========================"
echo "  Total images:  $TOTAL"
echo "  Output:        $OUTPUT_DIR"
echo "  Resolution:    ${WIDTH}x${HEIGHT}"
echo "  Quality:       $QUALITY"
echo "  Workers:       $WORKERS"
echo "  Generators:    $NUM_GENERATORS types"
echo ""

# Calculate images per batch (each batch uses one generator)
IMAGES_PER_GENERATOR=$((TOTAL / NUM_GENERATORS + 1))
BATCH_SIZE=50  # JWildfire renders this many per invocation

render_batch() {
    local gen="$1"
    local count="$2"
    local batch_dir="$3"

    mkdir -p "$batch_dir"
    cd "$batch_dir"

    # Run in batches of BATCH_SIZE
    local remaining=$count
    while [ $remaining -gt 0 ]; do
        local this_batch=$((remaining < BATCH_SIZE ? remaining : BATCH_SIZE))
        "$JAVA" -Xmx2g \
            -cp "$JWILDFIRE_LIB/*" \
            org.jwildfire.cli.RandomFlameGenerator \
            -w "$WIDTH" -h "$HEIGHT" \
            -q "$QUALITY" \
            -bc "$this_batch" \
            -rgflame "$gen" \
            -rgsymm "None" \
            2>/dev/null || true
        remaining=$((remaining - this_batch))
    done
}

# Track progress
STARTED=0

for gen in "${GENERATORS[@]}"; do
    STARTED=$((STARTED + 1))
    BATCH_DIR="$OUTPUT_DIR"

    echo "[$STARTED/$NUM_GENERATORS] Rendering $IMAGES_PER_GENERATOR images with generator: $gen"

    render_batch "$gen" "$IMAGES_PER_GENERATOR" "$BATCH_DIR" &

    # Limit concurrent workers
    while [ "$(jobs -rp | wc -l)" -ge "$WORKERS" ]; do
        sleep 1
    done
done

# Wait for all workers to finish
echo ""
echo "Waiting for all render jobs to complete..."
wait

# Count results
COUNT=$(find "$OUTPUT_DIR" -name "*.png" | wc -l | tr -d ' ')
echo ""
echo "Done! Rendered $COUNT images to $OUTPUT_DIR"
