#!/bin/bash
# File: shell/train_vit_binary.sh

echo "Sourcing Conda initialization..."
# Source Conda initialization
source ~/anaconda3/etc/profile.d/conda.sh

echo "Activating the Conda environment..."
# Activate the Conda environment
conda activate pulmo-sense

echo "Generating a timestamp for the log file..."
# Generate a timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")

echo "Ensuring logs directory exists..."
# Ensure logs directory exists
mkdir -p logs

echo "Setting PYTHONPATH and running the Python script..."
# Set PYTHONPATH and run the Python script
export PYTHONPATH=$(pwd)

python -m scripts.train.train_vit_binary \
    --dataset ccccii \
    --num_epochs 1 \
    --k 5 \
    --i 0 \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --max_samples 500 \
    2>&1 | tee logs/train_vit_binary_$timestamp.log &

echo "Script execution started, check logs/train_vit_binary_$timestamp.log for details."
