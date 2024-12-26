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

python -m scripts.train.train_vit_multiclass \
  --dataset ccccii \
  --num_epochs 20 \
  --k 5 \
  --i 0 \
  --batch_size 8 \
  --learning_rate 0.0001 > logs/train_vit_multiclass_$timestamp.log 2>&1 &
