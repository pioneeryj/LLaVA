#!/usr/bin/env python3
"""
Concept Prediction Accuracy Evaluation Script

This script compares CBM model predictions with LLaVA-inferred concepts to calculate accuracy.
- CBM predictions: Single predicted concept per image
- LLaVA predictions: 3 concepts per image
- Accuracy: % of CBM predictions that match any of the 3 LLaVA concepts
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict, Counter


def load_cbm_predictions(cbm_file):
    """Load CBM model predictions from JSON file"""
    try:
        with open(cbm_file, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict) and 'predictions' in data:
            predictions = data['predictions']
        elif isinstance(data, list):
            predictions = data
        else:
            predictions = data
            
        print(f"Loaded {len(predictions)} CBM predictions from {cbm_file}")
        return predictions
    
    except Exception as e:
        print(f"Error loading CBM predictions: {e}")
        return None


def load_llava_predictions(llava_file):
    """Load LLaVA concept predictions from JSON file"""
    try:
        with open(llava_file, 'r') as f:
            data = json.load(f)
        
        if 'results' in data:
            predictions = data['results']
        else:
            predictions = data
            
        print(f"Loaded {len(predictions)} LLaVA predictions from {llava_file}")
        return predictions
    
    except Exception as e:
        print(f"Error loading LLaVA predictions: {e}")
        return None


def normalize_concept_name(concept):
    """Normalize concept names for comparison"""
    if concept is None:
        return ""
    return str(concept).lower().strip()


def create_path_mapping(predictions, path_key='image_path'):
    """Create a mapping from image path to prediction data"""
    path_map = {}
    for pred in predictions:
        if path_key in pred:
            # Use basename for matching since paths might differ
            path = pred[path_key]
            basename = os.path.basename(path)
            path_map[basename] = pred
    return path_map


def calculate_accuracy(cbm_predictions, llava_predictions):
    """Calculate concept prediction accuracy"""
    
    # Create path mappings
    print("Creating path mappings...")
    cbm_map = create_path_mapping(cbm_predictions, 'image_path')
    llava_map = create_path_mapping(llava_predictions, 'image_path')
    
    print(f"CBM path map size: {len(cbm_map)}")
    print(f"LLaVA path map size: {len(llava_map)}")
    
    # Find common images
    cbm_paths = set(cbm_map.keys())
    llava_paths = set(llava_map.keys())
    common_paths = cbm_paths.intersection(llava_paths)
    
    print(f"Common images: {len(common_paths)}")
    print(f"CBM only: {len(cbm_paths - llava_paths)}")
    print(f"LLaVA only: {len(llava_paths - cbm_paths)}")
    
    if len(common_paths) == 0:
        print("No common images found! Check path formats.")
        return None
    
    # Calculate accuracy
    correct_predictions = 0
    total_predictions = 0
    detailed_results = []
    concept_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    print("\nEvaluating predictions...")
    print("=" * 60)
    
    for i, image_name in enumerate(sorted(common_paths)):
        cbm_pred = cbm_map[image_name]
        llava_pred = llava_map[image_name]
        
        # Extract CBM predicted concept
        cbm_concept = normalize_concept_name(cbm_pred.get('predicted_concept', ''))
        
        # Extract LLaVA predicted concepts (should be 3)
        llava_concepts = llava_pred.get('selected_concepts', [])
        if isinstance(llava_concepts, str):
            llava_concepts = [llava_concepts]
        
        llava_concepts_normalized = [normalize_concept_name(c) for c in llava_concepts]
        
        # Check if CBM concept matches any LLaVA concept
        is_correct = cbm_concept in llava_concepts_normalized
        
        # Update statistics
        total_predictions += 1
        if is_correct:
            correct_predictions += 1
        
        # Track by concept and class
        concept_stats[cbm_concept]['total'] += 1
        if is_correct:
            concept_stats[cbm_concept]['correct'] += 1
            
        class_name = cbm_pred.get('class_name', 'unknown')
        class_stats[class_name]['total'] += 1
        if is_correct:
            class_stats[class_name]['correct'] += 1
        
        # Store detailed result
        detailed_results.append({
            'image_name': image_name,
            'image_path': cbm_pred['image_path'],
            'cbm_concept': cbm_concept,
            'llava_concepts': llava_concepts,
            'is_correct': is_correct,
            'class_name': class_name,
            'anomaly_score': cbm_pred.get('anomaly_score', 0),
            'concept_score': cbm_pred.get('concept_score', 0)
        })
        
        # Print progress for first few examples
        if i < 10:
            print(f"Image: {image_name}")
            print(f"  CBM concept: '{cbm_concept}'")
            print(f"  LLaVA concepts: {llava_concepts}")
            print(f"  Match: {'✓' if is_correct else '✗'}")
            print()
    
    # Calculate overall accuracy
    overall_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'concept_stats': dict(concept_stats),
        'class_stats': dict(class_stats),
        'detailed_results': detailed_results
    }


def print_results(results):
    """Print detailed results"""
    if results is None:
        print("No results to display.")
        return
    
    print("=" * 80)
    print("CONCEPT PREDICTION ACCURACY RESULTS")
    print("=" * 80)
    
    # Overall accuracy
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Correct Predictions: {results['correct_predictions']} / {results['total_predictions']}")
    print()
    
    # Per-concept accuracy
    print("Per-Concept Accuracy:")
    print("-" * 40)
    concept_stats = results['concept_stats']
    for concept, stats in sorted(concept_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"{concept:<25} {stats['correct']:>3}/{stats['total']:<3} ({accuracy:>5.1f}%)")
    print()
    
    # Per-class accuracy
    print("Per-Class Accuracy:")
    print("-" * 40)
    class_stats = results['class_stats']
    for class_name, stats in sorted(class_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"{class_name:<25} {stats['correct']:>3}/{stats['total']:<3} ({accuracy:>5.1f}%)")
    print()
    
    # Top predicted concepts
    concept_counts = Counter([concept for concept in concept_stats.keys() if concept])
    print("Most Frequent CBM Predictions:")
    print("-" * 40)
    for concept, count in concept_counts.most_common(10):
        accuracy = (concept_stats[concept]['correct'] / concept_stats[concept]['total']) * 100
        print(f"{concept:<25} {count:>3} times ({accuracy:>5.1f}% acc)")
    print()
    
    # Worst performing concepts
    print("Concepts with Low Accuracy (>5 predictions):")
    print("-" * 40)
    low_acc_concepts = [(concept, stats) for concept, stats in concept_stats.items() 
                       if stats['total'] >= 5]
    low_acc_concepts.sort(key=lambda x: x[1]['correct'] / x[1]['total'])
    
    for concept, stats in low_acc_concepts[:10]:
        accuracy = (stats['correct'] / stats['total']) * 100
        print(f"{concept:<25} {stats['correct']:>3}/{stats['total']:<3} ({accuracy:>5.1f}%)")


def save_detailed_results(results, output_file):
    """Save detailed results to JSON file"""
    if results is None:
        return
        
    # Prepare data for JSON serialization
    output_data = {
        'summary': {
            'overall_accuracy': results['overall_accuracy'],
            'correct_predictions': results['correct_predictions'],
            'total_predictions': results['total_predictions'],
            'evaluation_timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S')
        },
        'per_concept_accuracy': {},
        'per_class_accuracy': {},
        'detailed_results': results['detailed_results']
    }
    
    # Add per-concept accuracy
    for concept, stats in results['concept_stats'].items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            output_data['per_concept_accuracy'][concept] = {
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy_percent': round(accuracy, 2)
            }
    
    # Add per-class accuracy
    for class_name, stats in results['class_stats'].items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            output_data['per_class_accuracy'][class_name] = {
                'correct': stats['correct'],
                'total': stats['total'],
                'accuracy_percent': round(accuracy, 2)
            }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate concept prediction accuracy by comparing CBM predictions with LLaVA concepts"
    )
    parser.add_argument("--cbm_predictions", type=str, required=True,
                      help="Path to CBM predictions JSON file")
    parser.add_argument("--llava_predictions", type=str, required=True,
                      help="Path to LLaVA concept predictions JSON file")
    parser.add_argument("--output", type=str, default="concept_accuracy_results.json",
                      help="Output file for detailed results")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.cbm_predictions):
        print(f"Error: CBM predictions file not found: {args.cbm_predictions}")
        return
        
    if not os.path.exists(args.llava_predictions):
        print(f"Error: LLaVA predictions file not found: {args.llava_predictions}")
        return
    
    print("Concept Prediction Accuracy Evaluation")
    print("=" * 50)
    print(f"CBM predictions: {args.cbm_predictions}")
    print(f"LLaVA predictions: {args.llava_predictions}")
    print()
    
    # Load predictions
    cbm_predictions = load_cbm_predictions(args.cbm_predictions)
    if cbm_predictions is None:
        return
        
    llava_predictions = load_llava_predictions(args.llava_predictions)
    if llava_predictions is None:
        return
    
    # Calculate accuracy
    results = calculate_accuracy(cbm_predictions, llava_predictions)
    
    # Print and save results
    print_results(results)
    save_detailed_results(results, args.output)
    
    print("=" * 80)
    print(f"FINAL ACCURACY: {results['overall_accuracy']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
