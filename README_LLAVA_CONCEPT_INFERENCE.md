# LLaVA Concept Inference for Anomaly Detection

This script uses LLaVA (Large Language and Vision Assistant) to analyze images and identify anomaly concepts from a predefined set of concepts.

## Overview

The script:
1. Loads a test dataset using the same pattern as `test_cbm_finetune_v4.py`
2. For each image, asks LLaVA to identify 3 relevant anomaly concepts from `concepts_new.json`
3. Saves results in a JSON file containing concept predictions, image paths, and LLaVA responses

## Files Created

- `llava_concept_inference.py`: Main inference script
- `utils.py`: Utility functions for image preprocessing
- `run_llava_concept_inference.sh`: Example run script
- `README_LLAVA_CONCEPT_INFERENCE.md`: This documentation

## Usage

### Basic Usage

```bash
python llava_concept_inference.py \
    --data_path "/path/to/your/dataset" \
    --dataset "mvtec" \
    --concepts_file "./concepts_new.json" \
    --output_file "./llava_concept_results.json" \
    --model-path "liuhaotian/llava-v1.5-7b"
```

### Arguments

#### Required Arguments
- `--data_path`: Path to the test dataset directory
- `--concepts_file`: Path to concepts JSON file (default: "./concepts_new.json")
- `--output_file`: Output JSON file path (default: "./llava_concept_results.json")

#### Dataset Arguments
- `--dataset`: Dataset name (default: "mvtec")
  - Supported: mvtec, visa, mpdd, btad, DAGM, SDD, DTD, colon, etc.
- `--image_size`: Image size for preprocessing (default: 518)

#### LLaVA Model Arguments
- `--model-path`: LLaVA model path (default: "liuhaotian/llava-v1.5-7b")
- `--model-base`: LLaVA model base (optional)
- `--device`: Device to use (default: auto-detect CUDA/CPU)
- `--conv-mode`: Conversation mode (default: "llava_v1")

#### Generation Arguments
- `--temperature`: Temperature for generation (default: 0.0 for deterministic)
- `--top_p`: Top-p filtering (optional)
- `--num_beams`: Number of beams for beam search (default: 1)
- `--max-new-tokens`: Max new tokens to generate (default: 512)

#### Processing Arguments
- `--max-samples`: Maximum number of samples to process (default: 0 for all)

## Concept Categories

The script uses concepts from `concepts_new.json` with the following categories:
- **shape**: Geometric anomalies (irregular shape, distorted contour, twisted, etc.)
- **color**: Color-related anomalies (stain, discoloration, wrong color, etc.)  
- **structure**: Structural defects (broken joint, missing part, cracked, etc.)
- **texture**: Surface texture issues (bumpy, wrinkled, rough, etc.)
- **normal**: Normal/defect-free concepts (normal, defect-free, intact)

## Output Format

The output JSON file contains:
```json
{
  "summary": {
    "total_processed": 150,
    "total_errors": 2,
    "dataset_name": "mvtec",
    "timestamp": "2025-09-19 10:30:00",
    "object_types": ["carpet", "bottle", "hazelnut", ...]
  },
  "results": [
    {
      "image_index": 0,
      "image_path": "/path/to/image.jpg",
      "image_name": "image.jpg",
      "class_name": "carpet",
      "anomaly_label": 1,
      "selected_concepts": ["hole", "irregular shape", "damaged support"],
      "llava_response": "Looking at this image, I can see...",
      "timestamp": "2025-09-19 10:30:15"
    },
    ...
  ]
}
```

## Features

### Robust Concept Extraction
- Parses LLaVA responses to extract exactly 3 concepts
- Handles various response formats
- Falls back to concept name matching if structured parsing fails
- Provides default concepts if extraction fails

### Error Handling
- Graceful handling of dataset loading errors
- Fallback image loading from directory structure
- Intermediate result saving every 50 images
- Comprehensive error logging

### Flexible Dataset Support
- Works with multiple dataset formats (MVTec, ViSA, etc.)
- Compatible with existing CBM dataset structure
- Automatic object type detection

## Example Results

The script will generate detailed analysis showing which concepts are most commonly identified for different types of anomalies across your dataset.

## Troubleshooting

### Dataset Loading Issues
- Make sure the dataset path is correct
- Ensure `meta.json` files exist for structured datasets
- Check that image files have supported extensions (.jpg, .jpeg, .png, .bmp, .tiff)

### Memory Issues
- Use `--max-samples` to limit processing
- Ensure sufficient GPU memory for LLaVA model
- Consider using smaller batch sizes

### Model Loading Issues
- Verify internet connection for downloading LLaVA models
- Check CUDA availability for GPU inference
- Ensure sufficient disk space for model cache

## Dependencies

- PyTorch
- PIL (Pillow)
- tqdm
- numpy
- LLaVA package (from this repository)

## Notes

- The script saves intermediate results every 50 processed images
- Processing time depends on model size and number of images
- Results include both successful concept identification and any errors encountered
- The script is compatible with the existing CBM (Concept Bottleneck Model) workflow
