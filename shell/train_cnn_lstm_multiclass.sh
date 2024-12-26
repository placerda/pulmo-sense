#!/bin/bash

# Initialize Conda (optional; adapt if you use a different environment manager)
conda init

# Activate your environment
conda activate pulmo-sense

# Generate a timestamp for your log file
timestamp=$(date +"%Y%m%d_%H%M%S")

# Run the CNN-LSTM training script locally
python -m scripts.train.train_cnn_lstm_multiclass \
  --dataset ccccii \
  --num_epochs 20 \
  --k 5 \
  --i 0 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  > logs/train_cnn_lstm_multiclass_${timestamp}.log 2>&1 &
