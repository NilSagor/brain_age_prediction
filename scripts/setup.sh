#!/bin/bash
# scripts/setup.sh – Quick setup for Brain Age Project
# Python 3.12 | RTX 4060 Ti | CUDA 12.4

set -e

echo "🧠 Brain Age Project Setup"
echo "==========================="
echo ""

# Check Python version
PYTHON_CMD=$(command -v python3 || command -v python)
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10+."
    exit 1
fi

echo "✅ Python found: $($PYTHON_CMD --version)"

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found – GPU may not be available."
fi

# Run make commands
echo ""
echo "📦 Installing dependencies (this may take a few minutes)..."
make install
make install-dev

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  make test    # Run tests"
echo "  make train   # Start training"