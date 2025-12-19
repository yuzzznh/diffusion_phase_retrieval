#!/bin/bash
# ============================================================================
# Environment Setup Script for ReSample + DAPS
# Tested on: Ubuntu 22.04, NVIDIA A100 80GB, CUDA 12.1
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "1. Downloading and Installing Anaconda"
echo "=========================================="

cd /home/ubuntu

# Download Anaconda (skip if already exists)
if [ ! -d "/home/ubuntu/anaconda3" ]; then
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O anaconda.sh
    bash anaconda.sh -b -p /home/ubuntu/anaconda3
    rm anaconda.sh
    echo "Anaconda installed successfully!"
else
    echo "Anaconda already installed, skipping..."
fi

# Initialize conda
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh

echo "=========================================="
echo "2. Creating Conda Environment"
echo "=========================================="

cd /home/ubuntu/project_resample_daps

# Create environment from unified_env.yaml
conda env create -f unified_env.yaml || echo "Environment may already exist, trying to update..."
conda activate Integrated_LDM

echo "=========================================="
echo "3. Installing Additional Packages"
echo "=========================================="

# Install missing packages from DAPS requirements.txt
pip install -r DAPS/requirements.txt

# Install additional tools
pip install hydra-core piq gdown imageio-ffmpeg

echo "=========================================="
echo "4. Downloading Pretrained Models & Datasets"
echo "=========================================="

cd /home/ubuntu/project_resample_daps/DAPS

# Download all checkpoints and datasets
pip install gdown
bash download.sh

echo "=========================================="
echo "5. Verification"
echo "=========================================="

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import kornia; print(f'Kornia: {kornia.__version__}')"

# Check datasets
echo ""
echo "Checking datasets..."
ls -la /home/ubuntu/project_resample_daps/DAPS/dataset/

# Check checkpoints
echo ""
echo "Checking checkpoints..."
ls -la /home/ubuntu/project_resample_daps/DAPS/checkpoints/

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source ~/anaconda3/etc/profile.d/conda.sh"
echo "  conda activate Integrated_LDM"
echo ""
echo "To run DAPS experiments:"
echo "  cd ~/project_resample_daps/DAPS"
echo "  bash commands/ldm_ffhq.sh"
echo "  bash commands/ldm_imagenet.sh"
echo ""
