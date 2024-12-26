#!/bin/bash
conda init
conda activate pulmo-sense

timestamp=$(date +"%Y%m%d_%H%M%S")
python -m scripts.train.train_vit_multiclass \
  --dataset ccccii \
  --num_epochs 20 \
  --k 5 \
  --i 0 \
  --batch_size 8 \
  --learning_rate 0.0001 > logs/train_vit_multiclass_$timestamp.log 2>&1 &
