#!/bin/bash

python make_category_text.py \
  --output-json results/anomaly_category_texts.json \
  --model-path liuhaotian/llava-v1.5-7b \
  --device cuda \
  --texts-per-category 50 \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 1024