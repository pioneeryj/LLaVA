import argparse
import torch
import json
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import glob

# LLaVA imports
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

# Dataset imports (following test_cbm_finetune_v4.py pattern)
from dataset import Dataset
from utils import get_transform


def load_concepts(concepts_file):
    """Load concepts from JSON file"""
    with open(concepts_file, 'r') as f:
        concepts = json.load(f)
    
    # Flatten all concepts into a single list with categories
    all_concepts = []
    for category, concept_list in concepts.items():
        for concept in concept_list:
            all_concepts.append({
                'concept': concept,
                'category': category
            })
    
    return all_concepts, concepts


def create_concept_prompt(all_concepts):
    """Create a prompt for LLaVA to identify anomaly concepts"""
    
    # Create concept list string
    concept_list = []
    for i, concept_info in enumerate(all_concepts):
        concept_list.append(f"{i+1}. {concept_info['concept']} ({concept_info['category']})")
    
    concept_string = "\n".join(concept_list)
    
    prompt = f"""Look at this image carefully and identify any anomalies or defects present. From the following list of concepts, select exactly 3 that best describe what you observe in the image. You must include 'normal' if the image appears defect-free, or select 3 anomaly concepts if defects are present.

Available concepts:
{concept_string}

Please respond with exactly 3 concept numbers and their names in this format:
Selected concepts: [number] [concept_name], [number] [concept_name], [number] [concept_name]

For example: "Selected concepts: 1 irregular shape, 15 stain, 145 normal"

Image analysis:"""

    return prompt


def extract_concepts_from_response(response, all_concepts):
    """Extract selected concepts from LLaVA response"""
    try:
        selected = []
        response_text = response.lower().strip()
        
        # Look for the "Selected concepts:" pattern first
        if "selected concepts:" in response_text:
            concepts_line = response_text.split("selected concepts:")[1].strip()
            # Split by comma and extract concept names
            parts = concepts_line.split(',')
            
            for part in parts[:3]:  # Only take first 3
                part = part.strip()
                # Try to extract concept name after the number
                words = part.split()
                if len(words) >= 2:
                    # Remove the number and get the concept name
                    concept_name = ' '.join(words[1:])
                    # Clean up any punctuation
                    concept_name = concept_name.strip('.,;:()[]{}')
                    selected.append(concept_name)
        
        # If we didn't get enough concepts, try to find them directly in the response
        if len(selected) < 3:
            found_concepts = []
            # Create a mapping of concepts to search for
            concept_names = [concept_info['concept'].lower() for concept_info in all_concepts]
            
            # Look for exact matches first
            for concept_info in all_concepts:
                concept_lower = concept_info['concept'].lower()
                if concept_lower in response_text:
                    # Check if it's not already found
                    if concept_info['concept'] not in [c.lower() for c in found_concepts]:
                        found_concepts.append(concept_info['concept'])
                        if len(found_concepts) >= 3:
                            break
            
            # If still not enough, look for partial matches
            if len(found_concepts) < 3:
                for concept_info in all_concepts:
                    concept_words = concept_info['concept'].lower().split()
                    for word in concept_words:
                        if len(word) > 3 and word in response_text:  # Only consider words longer than 3 chars
                            if concept_info['concept'] not in [c.lower() for c in found_concepts]:
                                found_concepts.append(concept_info['concept'])
                                if len(found_concepts) >= 3:
                                    break
                    if len(found_concepts) >= 3:
                        break
            
            # Use found concepts if we got some
            if found_concepts:
                selected.extend(found_concepts[:3 - len(selected)])
        
        # If still not enough, add 'normal' as default
        while len(selected) < 3:
            if 'normal' not in [c.lower() for c in selected]:
                selected.append('normal')
            else:
                selected.append('defect-free')
        
        return selected[:3]
    
    except Exception as e:
        print(f"Error extracting concepts: {e}")
        # Return default concepts
        return ['normal', 'defect-free', 'intact']


def load_images_from_directory(args):
    """Fallback function to load images directly from directory structure"""
    print("Loading images from directory structure...")
    
    # Common dataset directory patterns
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    # Walk through directory to find images
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images in directory")
    
    if len(image_files) == 0:
        print("No images found in the specified directory")
        return
    
    # Limit processing if specified
    if args.max_samples > 0:
        image_files = image_files[:args.max_samples]
    
    # Process images directly
    results = []
    processed_count = 0
    error_count = 0
    
    # Load LLaVA components
    all_concepts, _ = load_concepts(args.concepts_file)
    concept_prompt = create_concept_prompt(all_concepts)
    
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # Load image
            pil_image = Image.open(img_path).convert('RGB')
            
            # Run LLaVA inference (assuming model is available in scope)
            # This would need to be called from main function scope
            
            result_entry = {
                'image_index': idx,
                'image_path': img_path,
                'image_name': os.path.basename(img_path),
                'class_name': 'unknown',  # Can't determine from directory structure alone
                'anomaly_label': -1,  # Unknown
                'selected_concepts': ['normal', 'defect-free', 'intact'],  # Default
                'llava_response': 'Fallback loading - no LLaVA inference performed',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results.append(result_entry)
            processed_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Fallback processing completed: {processed_count} images processed, {error_count} errors")
    return results


def run_llava_inference(image, prompt, model, tokenizer, image_processor, args):
    """Run LLaVA inference on a single image with GPU memory optimization"""
    try:
        # Clear GPU cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare conversation
        use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        if use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_for_model = conv.get_prompt()
        
        print(f"Debug: Prompt length: {len(prompt_for_model)}")
        
        # Process image with memory optimization
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=model.dtype)
        
        input_ids = tokenizer_image_token(
            prompt_for_model, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)
        
        print(f"Debug: Input shape: {input_ids.shape}, Image shape: {image_tensor.shape}")
        
        # Generate with GPU memory-efficient settings
        with torch.inference_mode():
            try:
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    max_new_tokens=100,  # Reduced to save memory
                    min_new_tokens=5,
                    use_cache=False,  # Disable cache to save memory
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                print(f"Debug: Generation successful, output shape: {output_ids.shape}")
            except torch.cuda.OutOfMemoryError as oom_error:
                print(f"CUDA OOM Error during generation: {oom_error}")
                # Clear cache and retry with even smaller settings
                torch.cuda.empty_cache()
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image.size],
                    do_sample=False,
                    max_new_tokens=50,  # Even smaller
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        # Clear memory immediately after generation
        del image_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decode response
        input_token_len = input_ids.shape[1]
        generated_tokens = output_ids[0, input_token_len:]
        print(f"Debug: Generated {len(generated_tokens)} tokens")
        
        if len(generated_tokens) == 0:
            return "No tokens generated - model may have hit early stopping"
        
        generated_text = tokenizer.batch_decode([generated_tokens], skip_special_tokens=True)[0].strip()
        print(f"Debug: Generated text: '{generated_text[:200]}...'")
        
        return generated_text if generated_text else "Empty response generated"
    
    except torch.cuda.OutOfMemoryError as oom_error:
        print(f"CUDA Out of Memory Error: {oom_error}")
        torch.cuda.empty_cache()
        return f"CUDA OOM Error: {str(oom_error)}"
    except Exception as e:
        print(f"Error in LLaVA inference: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def main(args):
    print("="*60)
    print("LLaVA Concept Inference for Anomaly Detection")
    print("="*60)
    
    # Initialize LLaVA
    disable_torch_init()
    
    print(f"Loading LLaVA model: {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)
    
    # Load model with GPU optimization
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        device=args.device, load_8bit=False, load_4bit=False
    )
    model.eval()
    
    print("LLaVA model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Load concepts
    print(f"Loading concepts from: {args.concepts_file}")
    all_concepts, concepts_dict = load_concepts(args.concepts_file)
    print(f"Loaded {len(all_concepts)} concepts from {len(concepts_dict)} categories")
    
    # Create prompt
    concept_prompt = create_concept_prompt(all_concepts)
    
    # Load dataset (following test_cbm_finetune_v4.py pattern)
    print(f"Loading test dataset from: {args.data_path}")
    
    # Get data transforms
    try:
        preprocess, target_transform = get_transform(args)
    except Exception as e:
        print(f"Warning: Could not load transforms from utils: {e}")
        # Use simple tensor conversion without torchvision
        def simple_preprocess(image):
            # Convert PIL to tensor and normalize
            import numpy as np
            img_array = np.array(image.resize((args.image_size, args.image_size))) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
            # Simple normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            return img_tensor
        
        preprocess = simple_preprocess
        target_transform = None
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return
    
    try:
        # Load test data
        test_data = Dataset(
            root=args.data_path, 
            transform=preprocess, 
            target_transform=target_transform, 
            dataset_name=args.dataset
        )
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
        obj_list = test_data.obj_list
        print(f"Found {len(obj_list)} object types: {obj_list}")
        print(f"Total test samples: {len(test_data)}")
        
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTrying alternative dataset loading approach...")
        
        # Alternative approach: load images directly from directory
        try:
            # This is a fallback approach if the Dataset class fails
            print("Using fallback image loading from directory structure...")
            return load_images_from_directory(args)
        except Exception as e2:
            print(f"Fallback also failed: {str(e2)}")
            return
    
    # Initialize results storage
    results = []
    processed_count = 0
    error_count = 0
    
    print("\nStarting inference on test dataset...")
    print("="*60)
    
    # Process each image - following exact pattern from test_cbm_finetune_v4.py
    for idx, items in enumerate(tqdm(test_dataloader, desc="Processing images")):
        try:
            # Extract data from batch - exactly as in test_cbm_finetune_v4.py
            image = items['img']  # Keep batch dimension for now
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            gt_mask = items['img_mask']
            
            # Extract single batch item (batch_size=1)
            img_tensor = image[0]  # Remove batch dimension for processing
            img_path = items['img_path'][0]  # items['img_path'] is a list
            cls_name_str = cls_name[0]
            anomaly = items['anomaly'].detach().cpu()[0].item()  # Following exact pattern
            
            # Convert tensor back to PIL Image for LLaVA
            if isinstance(img_tensor, torch.Tensor):
                # Handle different tensor formats
                if len(img_tensor.shape) == 3:  # C, H, W format
                    # Denormalize the image tensor
                    img_tensor_denorm = img_tensor.clone()
                    
                    # Check if tensor is normalized (values around [-2, 2] indicate normalization)
                    if img_tensor_denorm.min() < -1.5 or img_tensor_denorm.max() > 1.5:
                        # Standard ImageNet denormalization
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor_denorm = img_tensor_denorm * std + mean
                    
                    img_tensor_denorm = torch.clamp(img_tensor_denorm, 0, 1)
                    
                    # Convert to PIL Image
                    img_np = (img_tensor_denorm.permute(1, 2, 0).numpy() * 255).astype('uint8')
                    pil_image = Image.fromarray(img_np).convert('RGB')
                else:
                    print(f"Unexpected tensor shape: {img_tensor.shape}")
                    continue
            else:
                # If it's already a PIL image or numpy array
                if isinstance(img_tensor, np.ndarray):
                    pil_image = Image.fromarray(img_tensor).convert('RGB')
                else:
                    pil_image = img_tensor
            
            # Run LLaVA inference
            response = run_llava_inference(
                pil_image, concept_prompt, model, tokenizer, image_processor, args
            )
            
            # Extract selected concepts
            selected_concepts = extract_concepts_from_response(response, all_concepts)
            
            # Store results
            result_entry = {
                'image_index': idx,
                'image_path': str(img_path),
                'image_name': os.path.basename(str(img_path)),
                'class_name': str(cls_name_str),
                'anomaly_label': int(anomaly),
                'selected_concepts': selected_concepts,
                'llava_response': response,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results.append(result_entry)
            processed_count += 1
            
            # Save intermediate results every 50 images
            if processed_count % 50 == 0:
                temp_output_file = args.output_file.replace('.json', f'_temp_{processed_count}.json')
                with open(temp_output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nIntermediate results saved to: {temp_output_file}")
            
            # Limit processing if specified
            if args.max_samples > 0 and processed_count >= args.max_samples:
                print(f"\nReached maximum sample limit: {args.max_samples}")
                break
                
        except Exception as e:
            error_count += 1
            print(f"\nError processing image {idx}: {str(e)}")
            continue
    
    # Save final results
    print(f"\nProcessing completed!")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors encountered: {error_count} images")
    
    # Add summary to results
    summary = {
        'total_processed': processed_count,
        'total_errors': error_count,
        'total_samples': len(test_data),
        'dataset_name': args.dataset,
        'data_path': args.data_path,
        'concepts_file': args.concepts_file,
        'model_path': args.model_path,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'object_types': obj_list
    }
    
    final_output = {
        'summary': summary,
        'results': results
    }
    
    # Save results to JSON file
    print(f"Saving results to: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print("="*60)
    print("COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {args.output_file}")
    print(f"Total processed images: {processed_count}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA Concept Inference for Anomaly Detection")
    
    # Dataset arguments (following test_cbm_finetune_v4.py)
    parser.add_argument("--data_path", type=str, required=True, help="path to test dataset")
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset name")
    parser.add_argument("--concepts_file", type=str, default="./concepts_new.json", help="path to concepts JSON file")
    parser.add_argument("--output_file", type=str, default="./llava_concept_results.json", help="output JSON file path")
    
    # LLaVA model arguments
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-mistral-7b", help="LLaVA model path")
    parser.add_argument("--model-base", type=str, default=None, help="LLaVA model base")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    parser.add_argument("--conv-mode", type=str, default="llava_v1", help="conversation mode")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for generation")
    parser.add_argument("--top_p", type=float, default=None, help="top-p filtering")
    parser.add_argument("--num_beams", type=int, default=1, help="number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="max new tokens to generate")
    
    # Processing arguments
    parser.add_argument("--max-samples", type=int, default=0, help="maximum number of samples to process (0 for all)")
    parser.add_argument("--image_size", type=int, default=518, help="image size for dataset transform")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used (for compatibility)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    main(args)
