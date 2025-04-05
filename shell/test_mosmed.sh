#!/bin/bash
# run_test.sh: Script to test the best model on the Mosmed dataset.
#
# Usage:
#   ./run_test.sh --model_type <vgg|vit|lstm_attn> --model_path <path_to_best_model.pth> [--vgg_model_path <path_to_pretrained_vgg_weights.pth>] [--mosmed_dataset <path_to_mosmed_dataset>] [--batch_size <batch_size>]
#
# Example:
#   ./run_test.sh --model_type vgg --model_path outputs/best_vgg_model.pth --mosmed_dataset mosmed --batch_size 16
#
# For lstm_attn model, the --vgg_model_path parameter is required.

# Default values
MODEL_TYPE="vit"
MODEL_PATH="models/vit_binary_4epoch_0.00050lr_0.954rec.pth"
MOSMED_DATASET="data/mosmed_png"
BATCH_SIZE=16
VGG_MODEL_PATH=""

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift ;;
        --model_path)
            MODEL_PATH="$2"
            shift ;;
        --mosmed_dataset)
            MOSMED_DATASET="$2"
            shift ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift ;;
        --vgg_model_path)
            VGG_MODEL_PATH="$2"
            shift ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1 ;;
    esac
    shift
done

# Check if the model type is lstm_attn and if so, ensure VGG_MODEL_PATH is provided.
if [[ "$MODEL_TYPE" == "lstm_attn" && -z "$VGG_MODEL_PATH" ]]; then
    echo "Error: For lstm_attn model, you must provide --vgg_model_path."
    exit 1
fi

echo "Testing model type: $MODEL_TYPE"
echo "Mosmed dataset path: $MOSMED_DATASET"
echo "Batch size: $BATCH_SIZE"
echo "Model weights path: $MODEL_PATH"
if [[ "$MODEL_TYPE" == "lstm_attn" ]]; then
    echo "Pretrained VGG weights path: $VGG_MODEL_PATH"
fi

# Construct the command

CMD="python -m scripts.test.test_mosmed --mosmed_dataset $MOSMED_DATASET --batch_size $BATCH_SIZE --model_type $MODEL_TYPE --model_path $MODEL_PATH"
if [[ "$MODEL_TYPE" == "lstm_attn" ]]; then
    CMD="$CMD --vgg_model_path $VGG_MODEL_PATH"
fi

echo "Running command:"
echo "$CMD"
eval "$CMD"
