#!/bin/bash
# Setup script for Brain Age Project
# Python 3.12 | RTX 4060 Ti | CUDA 12.4

set -e

echo " Brain Age Project Setup"
echo "=========================="
echo ""

# Check Python 3.12
PYTHON_CMD=$(command -v python3.12 || command -v python3)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)

echo " Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    echo "  Python 3.12 recommended. Found: $PYTHON_VERSION"
fi

# Check if already in venv
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "  Already in virtual environment: $VIRTUAL_ENV"
    echo "Run: deactivate"
    exit 1
fi

# Check GPU
echo ""
echo " GPU Information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader
else
    echo "  nvidia-smi not found"
fi

# Create virtual environment
echo ""
echo " Step 1: Creating virtual environment (Python 3.12)..."
$PYTHON_CMD -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "⬆ Step 2: Upgrading pip..."
pip install --upgrade pip setuptools wheel pip-tools

# Stage 1: PyTorch
echo ""
echo " Step 3: Installing PyTorch 2.5.1 (CUDA 12.4)..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Stage 2: Production deps (hashed)
echo ""
echo " Step 4: Installing production dependencies..."
pip install -r requirements/requirements.txt

# Stage 3: Dev deps (separate to avoid hash conflict)
echo ""
echo " Step 5: Installing development tools..."
pip install -r requirements/requirements-dev.txt

# Stage 4: Project
echo ""
echo " Step 6: Installing project..."
pip install -e .

# Verify
echo ""
echo "Step 7: Verification..."
python -c "
import torch
import lightning
import monai
import nibabel

print('')
print('Installation Successful!')
print('=' * 40)
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
if torch.cuda.is_available():
    print(f'Memory:       {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')
print(f'Lightning:    {lightning.__version__}')
print(f'MONAI:        {monai.__version__}')
print(f'NiBabel:      {nibabel.__version__}')
print('=' * 40)
"

echo ""
echo " Next steps:"
echo "  source .venv/bin/activate"
echo "  make train     # Start training"
echo "  make verify    # Check GPU"