#!/bin/bash
# File: shell/train_vgg_binary.sh

# Source Conda initialization
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate pulmo-sense

# Set CUDA_LAUNCH_BLOCKING to help debug kernel errors
export CUDA_LAUNCH_BLOCKING=1

# Generate a timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")

# Ensure logs directory exists
mkdir -p logs

# Set PYTHONPATH and run the Python script
export PYTHONPATH=$(pwd)

python -m scripts.train.train_vgg_binary \
  --dataset ccccii \
  --num_epochs 1 \
  --k 5 \
  --i 0 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --max_samples 100 \
  2>&1 | tee logs/train_vgg_binary_$timestamp.log &
