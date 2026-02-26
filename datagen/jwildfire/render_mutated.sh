#!/usr/bin/env bash
# Render mutated .flame files to PNG using JWildfire CLI.
#
# Usage:
#   ./render_mutated.sh [FLAME_DIR] [OUTPUT_DIR] [WORKERS]
#
# JWildfire saves PNGs next to the .flame files, so after rendering
# this script moves PNGs to the output directory.

set -euo pipefail

FLAME_DIR="$(cd "${1:-data/mutated_flames}" && pwd)"
OUTPUT_DIR="$(mkdir -p "${2:-data/raw_jwildfire}" && cd "${2:-data/raw_jwildfire}" && pwd)"
WORKERS="${3:-8}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JWF_JAR="$SCRIPT_DIR/lib/lib/j-wildfire.jar"

# Find Java
if [[ -x "/opt/homebrew/opt/openjdk/bin/java" ]]; then
    JAVA_CMD="/opt/homebrew/opt/openjdk/bin/java"
elif command -v java &>/dev/null; then
    JAVA_CMD="$(command -v java)"
else
    echo "ERROR: Java is not installed. Run: brew install openjdk"
    exit 1
fi

if [[ ! -f "$JWF_JAR" ]]; then
    echo "ERROR: JWildfire jar not found at $JWF_JAR"
    exit 1
fi

TOTAL=$(ls "$FLAME_DIR"/*.flame 2>/dev/null | wc -l | tr -d ' ')
echo "Rendering $TOTAL flame files with $WORKERS workers..."
echo "  Input:  $FLAME_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Java:   $JAVA_CMD"

DONE=0
FAILED=0

# Render using xargs with inline command (avoids exported function issues)
ls "$FLAME_DIR"/*.flame | xargs -P "$WORKERS" -I {} sh -c '
    flame_file="$1"
    basename="$(basename "$flame_file" .flame)"
    outfile="'"$OUTPUT_DIR"'/${basename}.png"
    rendered_png="${flame_file%.flame}.png"

    if [ -f "$outfile" ]; then
        exit 0
    fi

    "'"$JAVA_CMD"'" -cp "'"$JWF_JAR"'" org.jwildfire.cli.FlameRenderer \
        -f "$flame_file" -w 512 -h 512 -q 200 2>/dev/null

    if [ -f "$rendered_png" ]; then
        mv "$rendered_png" "$outfile"
    fi
' _ {}

# Count results
RENDERED=$(ls "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Done. $RENDERED images rendered to $OUTPUT_DIR/"
