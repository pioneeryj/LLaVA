#!/usr/bin/env python3
"""
Batch LLaVA Inference for Anomaly Detection Dataset

This script processes a test dataset JSON file, runs LLaVA inference on each image,
and saves the results to a JSON file as {image_path: llava_response}.
"""

import argparse
import json
import os
import torch
from PIL import Image
import logging
from tqdm import tqdm
from datetime import datetime
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LLaVA components
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
except ImportError as e:
    logger.error(f"Failed to import LLaVA modules: {e}")
    logger.error("Please ensure LLaVA is properly installed.")
    raise


def load_llava_model(model_path, model_base=None, device="cuda"):
    """Load LLaVA model and tokenizer."""
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Using device: {device}")
    
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device=device
    )
    
    logger.info(f"Model loaded successfully. Context length: {context_len}")
    return tokenizer, model, image_processor, context_len


def prepare_conversation(query, conv_mode="llava_v1"):
    """Prepare conversation with the query."""
    if "llama-2" in conv_mode:
        conv_mode = "llava_llama_2"
    elif "mistral" in conv_mode or "mixtral" in conv_mode:
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in conv_mode:
        conv_mode = "chatml_direct"
    elif "v1.6" in conv_mode:
        conv_mode = "vicuna_v1"
    
    conv = conv_templates[conv_mode].copy()
    
    # Add image token to the query
    if DEFAULT_IMAGE_TOKEN not in query:
        query = DEFAULT_IMAGE_TOKEN + '\n' + query
    
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    
    return conv


def run_inference(image, query, tokenizer, model, image_processor, conv_mode="llava_v1", 
                 temperature=0.2, top_p=None, num_beams=1, max_new_tokens=512, device="cuda"):
    """Run inference on a single image."""
    try:
        # Prepare conversation
        conv = prepare_conversation(query, conv_mode)
        prompt = conv.get_prompt()
        
        # Process image
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=torch.float16)
        
        # Tokenize input
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
        
        # Generate response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        
        # Decode response
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return outputs
    
    except Exception as e:
        logger.error(f"Error in inference: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}"


def process_dataset(dataset_path, base_image_path, model_path, output_path, 
                   query, model_base=None, conv_mode="llava_v1", 
                   temperature=0.2, top_p=None, num_beams=1, max_new_tokens=512, 
                   device="cuda"):
    """Process entire dataset and save results."""
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Extract all test images
    all_images = []
    for class_name, class_data in dataset['test'].items():
        all_images.extend(class_data)
    
    logger.info(f"Found {len(all_images)} images to process")
    
    # Load model
    tokenizer, model, image_processor, context_len = load_llava_model(
        model_path, model_base, device
    )
    
    # Process images
    results = {}
    failed_images = []
    
    for item in tqdm(all_images, desc="Processing images"):
        img_path = item['img_path']
        full_img_path = os.path.join(base_image_path, img_path)
        
        logger.info(f"Processing: {img_path}")
        
        try:
            # Check if image file exists
            if not os.path.exists(full_img_path):
                logger.warning(f"Image not found: {full_img_path}")
                results[img_path] = f"Error: Image not found at {full_img_path}"
                failed_images.append(img_path)
                continue
            
            # Load and process image
            image = Image.open(full_img_path).convert('RGB')
            
            # Run inference
            response = run_inference(
                image, query, tokenizer, model, image_processor, 
                conv_mode, temperature, top_p, num_beams, max_new_tokens, device
            )
            
            results[img_path] = response
            
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {str(e)}")
            results[img_path] = f"Error: {str(e)}"
            failed_images.append(img_path)
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    total_images = len(all_images)
    successful = total_images - len(failed_images)
    
    logger.info(f"Processing complete!")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {len(failed_images)}")
    
    if failed_images:
        logger.info("Failed images:")
        for img in failed_images:
            logger.info(f"  - {img}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch LLaVA Inference')
    
    # Required arguments
    parser.add_argument('--dataset-path', required=True, 
                       help='Path to test dataset JSON file')
    parser.add_argument('--base-image-path', required=True,
                       help='Base path where images are located')
    parser.add_argument('--model-path', required=True,
                       help='Path to LLaVA model')
    parser.add_argument('--output-path', required=True,
                       help='Path to save results JSON')
    parser.add_argument('--query', required=True,
                       help='Query prompt for LLaVA')
    
    # Optional arguments
    parser.add_argument('--model-base', default=None,
                       help='Base model path (if using LoRA)')
    parser.add_argument('--conv-mode', default="llava_v1",
                       help='Conversation mode')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Temperature for generation')
    parser.add_argument('--top-p', type=float, default=None,
                       help='Top-p for generation')
    parser.add_argument('--num-beams', type=int, default=1,
                       help='Number of beams for generation')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                       help='Maximum new tokens to generate')
    parser.add_argument('--device', default="cuda",
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("=== Batch LLaVA Inference Configuration ===")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Base image path: {args.base_image_path}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Query: {args.query}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")
    logger.info("=" * 45)
    
    # Process dataset
    results = process_dataset(
        dataset_path=args.dataset_path,
        base_image_path=args.base_image_path,
        model_path=args.model_path,
        output_path=args.output_path,
        query=args.query,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )
    
    logger.info(f"Batch inference completed. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
