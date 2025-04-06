#!/bin/bash

# File: shell/train_lstm_vgg_multiclass.sh

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

python -m scripts.train.train_distill_lstm_attention_vgg_multiclass \
  --dataset ccccii \
  --num_epochs 1 \
  --k 5 \
  --i 0 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --max_samples 500 \
  --sequence_length 30 \
  --teacher_model_path "models/vgg_multiclass_19epoch_0.00050lr_0.991rec.pth" \
  --student_vgg_weights_path "models/vgg_multiclass_19epoch_0.00050lr_0.991rec.pth" \
  2>&1 | tee logs/distill_lstm_attention_vgg_multiclass_$timestamp.log &
