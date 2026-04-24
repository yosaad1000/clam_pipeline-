#!/bin/bash
# setup_pi.sh — Install all dependencies on Raspberry Pi
# Run once after copying pi_deploy/ to the Pi
#
# Usage: bash setup_pi.sh

set -e

echo "=== Setting up CLAM pipeline on Raspberry Pi ==="

# 1. System deps
echo "[1/4] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip python3-venv \
    openslide-tools libopenslide-dev \
    libvips-dev libopenjp2-7 \
    libtiff-dev libpng-dev libjpeg-dev \
    git wget

# 2. Python venv
echo "[2/4] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Python packages
echo "[3/4] Installing Python packages..."
pip install --upgrade pip
pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    openslide-python \
    h5py \
    numpy \
    scipy \
    Pillow \
    tifffile \
    timm==0.9.8 \
    pandas \
    matplotlib \
    scikit-learn \
    tqdm \
    PyYAML \
    opencv-python-headless

# 4. Memryx SDK
echo "[4/4] Installing Memryx SDK..."
echo ""
echo "  --> Go to https://developer.memryx.com and download the Raspberry Pi SDK"
echo "  --> Then run: pip install memryx-<version>-linux_aarch64.whl"
echo ""
echo "  Or if you have the .whl file already:"
echo "  pip install memryx*.whl"
echo ""

echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Install Memryx SDK (see above)"
echo "  2. Run: bash compile_resnet.sh"
echo "  3. Run: python pipeline_mxa.py --input input/ --output output/ --batch"
