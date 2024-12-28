#!/bin/bash

# File: shell/train_cnn2d_multiclass.sh

# Source Conda initialization
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate pulmo-sense

# Navigate to the project root
cd /home/paulo/workspace/doutorado/pulmo-sense

# (Optional) Install the package in editable mode
# pip install -e .

# Generate a timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")

# Ensure logs directory exists
mkdir -p logs

# Set PYTHONPATH to the project root
export PYTHONPATH=$(pwd)

# Run the Python script and log to both file and terminal
python -m scripts.train.train_cnn2d_multiclass \
  --dataset ccccii \
  --num_epochs 1 \
  --k 5 \
  --i 0 \
  --batch_size 32 \
  --max_samples 1000 \
  --learning_rate 0.0005 \
  2>&1 | tee logs/train_cnn2d_multiclass_$timestamp.log &
