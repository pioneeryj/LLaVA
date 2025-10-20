#!/bin/bash
python batch_llava_inference.py \
    --model-path "liuhaotian/llava-v1.5-7b" \
    --json-file "/mnt/HDD/yoonji/anomalyCBM/data/DTD/meta.json" \
    --base-path "/mnt/HDD/yoonji/anomalyCBM/data/DTD" \
    --prompt "default" \
    --max-new-tokens 256 \
    --temperature 0.0 \
    --load-in-8bit

