#!/usr/bin/env python3
"""
Quick concept accuracy evaluation for your specific data format
"""

import json

def evaluate_cbm_vs_llava_concepts():
    """
    Example evaluation using the data format you provided
    """
    
    # Example CBM predictions (your format)
    cbm_predictions = [
        {
            "image_path": "/mnt/HDD/yoonji/anomalyCBM/data/DTD/Woven_001/test/bad/000.png",
            "predicted_concept": "weak joint",
            "concept_score": 0.058072783052921295,
            "concept_index": 88,
            "anomaly_score": 0.99845290184021,
            "ground_truth": 1,
            "class_name": "Woven_001"
        },
        {
            "image_path": "/mnt/HDD/yoonji/anomalyCBM/data/DTD/Woven_001/test/bad/001.png",
            "predicted_concept": "weak joint",
            "concept_score": 0.04485013335943222,
            "concept_index": 88,
            "anomaly_score": 0.9931166172027588,
            "ground_truth": 1,
            "class_name": "Woven_001"
        }
        # ... more predictions
    ]
    
    # Example LLaVA concept inference results (what our script will generate)
    llava_predictions = [
        {
            "image_path": "/mnt/HDD/yoonji/anomalyCBM/data/DTD/Woven_001/test/bad/000.png",
            "image_name": "000.png",
            "selected_concepts": ["weak joint", "torn edge", "fragmented"],
            "llava_response": "Looking at this woven fabric, I can see structural damage...",
            "class_name": "Woven_001"
        },
        {
            "image_path": "/mnt/HDD/yoonji/anomalyCBM/data/DTD/Woven_001/test/bad/001.png",
            "image_name": "001.png", 
            "selected_concepts": ["irregular support", "damaged support", "weak joint"],
            "llava_response": "This fabric shows signs of structural weakness...",
            "class_name": "Woven_001"
        }
        # ... more predictions
    ]
    
    # Calculate accuracy
    correct = 0
    total = 0
    
    # Create mapping by image path
    llava_map = {pred['image_path']: pred for pred in llava_predictions}
    
    for cbm_pred in cbm_predictions:
        image_path = cbm_pred['image_path']
        cbm_concept = cbm_pred['predicted_concept'].lower().strip()
        
        if image_path in llava_map:
            llava_concepts = [c.lower().strip() for c in llava_map[image_path]['selected_concepts']]
            
            # Check if CBM concept matches any of the 3 LLaVA concepts
            is_match = cbm_concept in llava_concepts
            
            total += 1
            if is_match:
                correct += 1
                
            print(f"Image: {image_path.split('/')[-1]}")
            print(f"  CBM: '{cbm_concept}'")
            print(f"  LLaVA: {llava_concepts}")
            print(f"  Match: {'✓' if is_match else '✗'}")
            print()
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    print("Example Concept Accuracy Evaluation")
    print("===================================")
    evaluate_cbm_vs_llava_concepts()
    
    print("\nTo run with your actual data:")
    print("1. First run LLaVA concept inference:")
    print("   bash run_llava_concept_inference.sh")
    print("")
    print("2. Then evaluate accuracy:")
    print("   python evaluate_concept_accuracy.py \\")
    print("     --cbm_predictions your_cbm_predictions.json \\")
    print("     --llava_predictions llava_concept_results_DTD.json \\")
    print("     --output concept_accuracy_results.json")
