#!/bin/bash

# Clean the malformed anomaly category JSON file
python clean_json.py \
  --input-json results/anomaly_category_texts.json \
  --output-json results/anomaly_category_texts_cleaned.json \
  --target-count 30
