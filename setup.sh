#!/bin/bash

# Quick setup script for Reptile Scaling Law experiments

echo "========================================"
echo "Reptile Scaling Law Setup"
echo "========================================"

# Check Python version
python3 --version

# Install requirements
echo ""
echo "Installing Python packages..."
pip install -r requirements.txt

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To run experiments:"
echo "  1. Jupyter: jupyter notebook reptile_scaling_law_experiments.ipynb"
echo "  2. Python: python3 run_experiments.py"
echo ""
