#!/bin/bash

# Example script to run LLaVA concept inference
# Make sure to adjust the paths according to your setup
# Step 1: Run LLaVA concept inference
echo "Step 1: Running LLaVA concept inference..."
CUDA_VISIBLE_DEVICES=3 python llava_concept_inference.py \
    --data_path "/mnt/HDD/yoonji/anomalyCBM/data/DTD" \
    --dataset "DTD" \
    --concepts_file "./concepts_new.json" \
    --output_file "./llava_concept_results_DTD.json" \
    --model-path "liuhaotian/llava-v1.5-7b" \
    --device "cuda" \
    --conv-mode "llava_v1" \
    --max-samples 0 \
    --image_size 336 \
    --temperature 0.0

echo "LLaVA concept inference completed!"
echo "Results saved in: ./llava_concept_results_DTD.json"
echo ""

# Step 2: Evaluate concept prediction accuracy
echo "Step 2: Evaluating concept prediction accuracy..."
echo "Make sure you have your CBM predictions JSON file ready!"
echo ""
echo "Example command to evaluate accuracy:"
echo "python evaluate_concept_accuracy.py \\"
echo "    --cbm_predictions /path/to/your/cbm_predictions.json \\"
echo "    --llava_predictions ./llava_concept_results_DTD.json \\"
echo "    --output ./concept_accuracy_results.json"
echo ""
echo "This will calculate the accuracy by checking if CBM predicted concepts"
echo "match any of the 3 LLaVA-inferred concepts for each image."
