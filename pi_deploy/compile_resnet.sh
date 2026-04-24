#!/bin/bash
# compile_resnet.sh — Compile resnet50_trunc.onnx to Memryx DFP
# Run this on the Raspberry Pi after installing the Memryx SDK
#
# Usage: bash compile_resnet.sh

set -e

echo "=== Compiling ResNet50 ONNX → DFP for Memryx MXA ==="

if ! command -v mx_nc &> /dev/null; then
    echo "[ERROR] Memryx Neural Compiler (mx_nc) not found."
    echo "  Install the Memryx SDK first: bash setup_pi.sh"
    exit 1
fi

mx_nc -v \
    --models resnet50_trunc.onnx \
    --autocrop \
    --dfp_fname resnet50_trunc.dfp

echo ""
echo "=== Done! Generated: resnet50_trunc.dfp ==="
echo "You can now run the pipeline:"
echo "  python pipeline_mxa.py --input input/ --output output/ --batch"
