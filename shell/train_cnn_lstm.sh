#!/bin/bash
conda acivate pulmo-sense
python -m scripts.train.train_cnn2d_multiclass --dataset ccccii --num_epochs 20 --batch_size 16 --learning_rate 0.0005