#!/bin/bash

# Generate Anomaly Category Texts - Simple Version
python make_category_text.py \
  --output-json results/anomaly_category_texts_simple.json \
  --model-path liuhaotian/llava-v1.5-7b \
  --device cuda \
  --texts-per-category 30 \
  --temperature 0.5 \
  --top-p 0.9 \
  --max-new-tokens 512
