#!/bin/bash

# Simple LLaVA Anomaly Inference Script
echo "Running Simple LLaVA Anomaly Detection..."

# Set GPU device (change as needed)
export CUDA_VISIBLE_DEVICES=0

# Run inference
python simple_llava_anomaly_inference.py \
    --image-file /mnt/HDD/yoonji/anomalyCBM/data/DTD \
\

echo "Inference completed!"
echo "Results saved to: ./simple_llava_results.json"
