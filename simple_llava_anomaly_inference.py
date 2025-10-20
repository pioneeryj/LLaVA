#!/usr/bin/env python3
"""
Simple LLaVA anomaly inference script
Load test dataloader, run LLaVA inference for anomaly detection, save results as JSON
"""

import argparse
import torch
import json
import os
import time
from PIL import Image
from tqdm import tqdm
import numpy as np

# LLaVA imports
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

# Dataset imports
from dataset import Dataset
from utils import get_transform


def create_anomaly_prompt():
    """Create a simple prompt for anomaly detection"""
    prompt = """Look at this image and analyze it for any anomalies or defects. 

Please describe what you see and determine if this image shows:
1. A normal, defect-free object/surface
2. An object/surface with anomalies or defects

If you detect any anomalies, please describe what type of defect or anomaly you observe (e.g., cracks, stains, holes, deformation, color changes, etc.).

Analysis:"""
    return prompt


def run_llava_inference(image, prompt, model, tokenizer, image_processor, args):
    """Run LLaVA inference on a single image"""
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare conversation
        qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv = conv_templates[getattr(args, 'conv_mode', getattr(args, 'conv-mode', 'llava_v1'))].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_for_model = conv.get_prompt()
        
        # Process image
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=model.dtype)
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_for_model, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=getattr(args, 'max_new_tokens', 150),
                use_cache=False,  # Save memory
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Clean up memory
        del image_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decode response
        input_token_len = input_ids.shape[1]
        generated_tokens = output_ids[0, input_token_len:]
        generated_text = tokenizer.batch_decode([generated_tokens], skip_special_tokens=True)[0].strip()
        
        return generated_text if generated_text else "No response generated"
    
    except Exception as e:
        print(f"Error in inference: {e}")
        return f"Error: {str(e)}"


def tensor_to_pil(img_tensor):
    """Convert normalized tensor back to PIL Image"""
    if isinstance(img_tensor, torch.Tensor):
        if len(img_tensor.shape) == 3:  # C, H, W format
            # Denormalize
            img_denorm = img_tensor.clone()
            
            # Check if normalized (ImageNet normalization)
            if img_denorm.min() < -1.5 or img_denorm.max() > 1.5:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = img_denorm * std + mean
            
            img_denorm = torch.clamp(img_denorm, 0, 1)
            img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype('uint8')
            return Image.fromarray(img_np).convert('RGB')
    
    return img_tensor


def main(args):
    print("=" * 50)
    print("Simple LLaVA Anomaly Detection Inference")
    print("=" * 50)
    
    # Initialize LLaVA
    disable_torch_init()
    
    print(f"Loading model: {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=args.device
    )
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")
    
    # Create prompt
    anomaly_prompt = create_anomaly_prompt()
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    
    try:
        preprocess, target_transform = get_transform(args)
    except Exception as e:
        print(f"Using simple preprocessing due to error: {e}")
        def simple_preprocess(image):
            img_array = np.array(image.resize((224, 224))) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (img_tensor - mean) / std
        preprocess = simple_preprocess
        target_transform = None
    
    # Load test data
    test_data = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Total samples: {len(test_data)}")
    
    # Process images
    results = {}
    processed_count = 0
    
    for idx, items in enumerate(tqdm(test_dataloader, desc="Processing images")):
        try:
            # Extract data following test_cbm_finetune_v4.py pattern
            image = items['img']
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            gt_mask = items['img_mask']
            
            # Get single batch item
            img_tensor = image[0]
            img_path = items['img_path'][0]
            cls_name_str = cls_name[0]
            anomaly_label = items['anomaly'].detach().cpu()[0].item()
            
            # Convert tensor to PIL image for LLaVA
            pil_image = tensor_to_pil(img_tensor)
            
            # Run LLaVA inference
            response = run_llava_inference(
                pil_image, anomaly_prompt, model, tokenizer, image_processor, args
            )
            
            # Store result
            results[img_path] = {
                'image_path': img_path,
                'class_name': cls_name_str,
                'anomaly_label': int(anomaly_label),
                'llava_response': response,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            processed_count += 1
            
            # Print progress
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} images...")
            
            # Limit processing if specified
            if args.max_samples > 0 and processed_count >= args.max_samples:
                print(f"Reached maximum sample limit: {args.max_samples}")
                break
                
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            continue
    
    # Save results
    output_data = {
        'metadata': {
            'total_processed': processed_count,
            'dataset': args.dataset,
            'data_path': args.data_path,
            'model_path': args.model_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': results
    }
    
    print(f"Saving results to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print("=" * 50)
    print("COMPLETED!")
    print(f"Processed {processed_count} images")
    print(f"Results saved to: {args.output_file}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple LLaVA Anomaly Detection")
    
    # Required arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--dataset", type=str, default="mvtec", help="Dataset name")
    parser.add_argument("--output_file", type=str, default="./llava_anomaly_results.json", help="Output JSON file")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b", help="LLaVA model path")
    parser.add_argument("--model-base", type=str, default=None, help="Model base")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="Conversation mode")
    
    # Generation arguments (matching your format)
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p filtering")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens to generate")
    
    # Processing arguments
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to process (0 for all)")
    parser.add_argument("--image-size", type=int, default=518, help="Image size")
    parser.add_argument("--features-list", type=int, nargs="+", default=[6, 12, 18, 24], help="Feature layers")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    main(args)
