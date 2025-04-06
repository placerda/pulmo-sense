#!/bin/bash
# File: shell/train_lstm_vgg_binary.sh

# Source Conda initialization
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the pulmo-sense Conda environment
conda activate pulmo-sense

# Generate a timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")

# Ensure the logs directory exists
mkdir -p logs

# Set PYTHONPATH to the current working directory
export PYTHONPATH=$(pwd)

# Run the training script with binary settings and pretrained VGG weights
python -m scripts.train.train_lstm_vgg_binary \
  --dataset ccccii \
  --num_epochs 1 \
  --k 5 \
  --i 0 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --max_samples 500 \
  --vgg_model_path models/vgg_binary_1epoch_0.00050lr_0.981rec.pth \
  --sequence_length 30 \
  2>&1 | tee logs/train_lstm_vgg_binary_$timestamp.log &
