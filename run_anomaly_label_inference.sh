#!/bin/bash

# MVTec Anomaly Label Generation with LLaVA
python mvtec_anomaly_label_inference.py \
  --mvtec-path /mnt/HDD/yoonji/anomalyCBM/data/mvtec \
  --output-path results/mvtec_anomaly_labels_1024.json \
  --model-path liuhaotian/llava-v1.5-7b \
  --device cuda \
  --temperature 0.1 \
  --top-p 0.9 \
  --max-new-tokens 256