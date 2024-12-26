#!/bin/bash

# Source Conda initialization
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate pulmo-sense

# Generate a timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")

# Ensure logs directory exists
mkdir -p logs

# Set PYTHONPATH and run the Python script
export PYTHONPATH=$(pwd)

# Run the Python script and log to both file and terminal
python -m scripts.train.train_cnn2d_multiclass_128  \
  --dataset ccccii \
  --num_epochs 20 \
  --k 5 \
  --i 0 \
  --batch_size 16 \
  --learning_rate 0.0005 2>&1 | tee logs/train_cnn2d_multiclass_$timestamp.log &