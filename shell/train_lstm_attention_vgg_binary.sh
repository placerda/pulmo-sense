#!/bin/bash
# File: shell/train_lstm_attention_vgg_binary.sh

# Source Conda initialization
source ~/anaconda3/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate pulmo-sense

# Generate a timestamp for the log file
timestamp=$(date +"%Y%m%d_%H%M%S")

# Ensure the logs directory exists
mkdir -p logs

# Set PYTHONPATH to current working directory
export PYTHONPATH=$(pwd)

# Run the training script with desired parameters
python -m scripts.train.train_lstm_attention_vgg_binary \
  --dataset ccccii \
  --num_epochs 1 \
  --k 5 \
  --i 0 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --max_samples 500 \
  --cnn_model_path models/vgg_binary_1epoch_0.00050lr_0.981rec.pth \
  2>&1 | tee logs/train_lstm_attention_vgg_binary_$timestamp.log &
