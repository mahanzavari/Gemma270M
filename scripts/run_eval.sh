#!/bin/bash

# Script to run evaluation on a saved checkpoint
# IMPORTANT: Update CHECKPOINT_PATH to the directory of the adapters you want to evaluate.
# This could be a specific checkpoint like './outputs/checkpoint-1000' or the final one.
CHECKPOINT_PATH="./outputs/final_checkpoint"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory not found at $CHECKPOINT_PATH"
    echo "Please update the CHECKPOINT_PATH variable in this script."
    exit 1
fi

echo "Starting evaluation on checkpoint: $CHECKPOINT_PATH"

python src/eval.py \
    --config configs/default_config.yaml \
    --adapter_path $CHECKPOINT_PATH