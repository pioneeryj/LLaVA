#!/bin/bash

# Text-to-Anomaly Prediction Script
python text_to_anomaly_inference.py \
  --input-json results/anomaly_results.json \
  --output-json results/text_anomaly_results.json \
  --model-path liuhaotian/llava-v1.5-7b \
  --device cuda \
  --temperature 0.1 \
  --top-p 0.9 \
  --max-new-tokens 512
