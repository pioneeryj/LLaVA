#!/bin/bash

# Quick evaluation script for DTD dataset concept accuracy
# This assumes you have both CBM predictions and LLaVA concept inference results

echo "Evaluating Concept Prediction Accuracy for DTD Dataset"
echo "======================================================"

# Check if required files exist
CBM_PREDICTIONS="/path/to/your/cbm_dtd_predictions.json"  # Update this path
LLAVA_PREDICTIONS="./llava_concept_results_DTD.json"

if [ ! -f "$LLAVA_PREDICTIONS" ]; then
    echo "Error: LLaVA predictions not found at $LLAVA_PREDICTIONS"
    echo "Please run the LLaVA concept inference first:"
    echo "bash run_llava_concept_inference.sh"
    exit 1
fi

if [ ! -f "$CBM_PREDICTIONS" ]; then
    echo "Error: CBM predictions not found at $CBM_PREDICTIONS"
    echo "Please update the CBM_PREDICTIONS path in this script"
    echo "or provide the correct path to your CBM predictions JSON file"
    exit 1
fi

# Run the evaluation
echo "Running concept accuracy evaluation..."
python evaluate_concept_accuracy.py \
    --cbm_predictions "$CBM_PREDICTIONS" \
    --llava_predictions "$LLAVA_PREDICTIONS" \
    --output "./dtd_concept_accuracy_results.json"

echo ""
echo "Evaluation completed!"
echo "Detailed results saved in: ./dtd_concept_accuracy_results.json"
echo ""
echo "Key metrics:"
echo "- Overall accuracy: How many CBM predictions match LLaVA concepts"
echo "- Per-concept accuracy: Performance for each predicted concept"
echo "- Per-class accuracy: Performance for each object class (Woven_001, etc.)"
