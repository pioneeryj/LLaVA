import torch
import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
import argparse

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MVTecAnomalyInference:
    def __init__(self, model_path="liuhaotian/llava-v1.5-13b", device="cuda"):
        self.device = device
        disable_torch_init()
        
        logger.info(f"Loading LLaVA model from {model_path}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="llava-v1.5-13b",
            load_8bit=False,
            load_4bit=False,
            device_map="auto"
        )
        logger.info("Model loaded successfully")
    
    def load_image(self, image_path):
        """Load and convert image to RGB"""
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def run_inference(self, image, prompt, temperature=0.2, top_p=1.0, max_new_tokens=512):
        """Run inference on a single image"""
        try:
            # Prepare conversation
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            
            # Process image
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(self.device)
            
            # Prepare input with image token
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt_formatted = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(
                prompt_formatted, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # Generate response
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    use_cache=True
                )
            
            # Decode response
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            return outputs
            
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return f"Error: {str(e)}"
    
    def find_mvtec_images(self, mvtec_root_path):
        """Find all PNG images in MVTec dataset structure"""
        mvtec_path = Path(mvtec_root_path)
        image_list = []
        
        # Find all class directories
        for class_dir in mvtec_path.iterdir():
            if class_dir.is_dir():
                test_dir = class_dir / "test"
                if test_dir.exists():
                    # Find all subdirectories in test (normal, anomaly types)
                    for subdir in test_dir.iterdir():
                        if subdir.is_dir():
                            # Find all PNG images
                            for img_file in subdir.glob("*.png"):
                                relative_path = img_file.relative_to(mvtec_path)
                                image_list.append({
                                    'class': class_dir.name,
                                    'subset': subdir.name,
                                    'filename': img_file.name,
                                    'full_path': str(img_file),
                                    'relative_path': str(relative_path)
                                })
        
        logger.info(f"Found {len(image_list)} images across all classes")
        return image_list
    
    def process_mvtec_dataset(self, mvtec_root_path, prompt, output_path, temperature=0.2, top_p=1.0, max_new_tokens=512):
        """Process entire MVTec dataset"""
        
        # Find all images
        image_list = self.find_mvtec_images(mvtec_root_path)
        
        if not image_list:
            logger.error("No images found in the dataset")
            return
        
        # Process each image
        results = {}
        failed_images = []
        
        for item in tqdm(image_list, desc="Processing MVTec images"):
            img_path = item['full_path']
            relative_path = item['relative_path']
            
            # Use tqdm.write for cleaner output
            tqdm.write(f"🔄 Processing: {relative_path}")
            
            try:
                # Load image
                image = self.load_image(img_path)
                if image is None:
                    failed_images.append(relative_path)
                    results[relative_path] = "Error: Failed to load image"
                    continue
                
                # Run inference
                response = self.run_inference(
                    image, prompt, temperature, top_p, max_new_tokens
                )
                
                # Display response in real-time
                print(f"\n{'='*80}")
                print(f"🖼️  Image: {relative_path}")
                print(f"📝 Response:")
                print(f"{response}")
                print(f"{'='*80}\n")
                
                # Store result with metadata
                results[relative_path] = {
                    'class': item['class'],
                    'subset': item['subset'],
                    'filename': item['filename'],
                    'response': response
                }
                
            except Exception as e:
                tqdm.write(f"❌ ERROR: Failed to process {relative_path}: {e}")
                failed_images.append(relative_path)
                results[relative_path] = {
                    'class': item['class'],
                    'subset': item['subset'],
                    'filename': item['filename'],
                    'response': f"Error: {str(e)}"
                }
        
        # Save results
        logger.info(f"Saving results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        total_images = len(image_list)
        successful = total_images - len(failed_images)
        
        logger.info(f"Processing complete!")
        logger.info(f"Total images: {total_images}")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Failed: {len(failed_images)}")
        
        # Print class statistics
        class_stats = {}
        for item in image_list:
            class_name = item['class']
            if class_name not in class_stats:
                class_stats[class_name] = 0
            class_stats[class_name] += 1
        
        logger.info("Images per class:")
        for class_name, count in sorted(class_stats.items()):
            logger.info(f"  {class_name}: {count} images")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='MVTec Anomaly Detection with LLaVA')
    
    # Required arguments
    parser.add_argument('--mvtec-path', required=True, 
                       help='Path to MVTec dataset root directory')
    parser.add_argument('--output-path', required=True,
                       help='Path to save results JSON file')
    
    # Optional arguments
    parser.add_argument('--model-path', default="liuhaotian/llava-v1.5-13b",
                       help='Path to LLaVA model (default: liuhaotian/llava-v1.5-13b)')
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"],
                       help='Device to use for inference (default: cuda)')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Temperature for text generation (default: 0.2)')
    parser.add_argument('--top-p', type=float, default=1.0,
                       help='Top-p for text generation (default: 1.0)')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                       help='Maximum new tokens to generate (default: 512)')
    parser.add_argument('--prompt', type=str,
                       help='Custom prompt for anomaly detection')
    
    args = parser.parse_args()
    
    # Default anomaly detection prompt if not provided
    if args.prompt is None:
        prompt = """You are a visual inspector. Using only the image, decide for each category if a visual anomaly exists (no external context/logic like missing/wrong part).
Categories: Geometry & Shape, Color & Photometry, Texture & Pattern, Surface Integrity (Damage), Material & Composition.
If present, list 1–3 short, concrete, visual-only anomaly texts (≤7 words).
Return JSON only in this exact schema:

{
  "geometry_shape": { "is_present": false, "specific_anomalies": [] },
  "color_photometry": { "is_present": false, "specific_anomalies": [] },
  "texture_pattern": { "is_present": false, "specific_anomalies": [] },
  "surface_integrity": { "is_present": false, "specific_anomalies": [] },
  "material_composition": { "is_present": false, "specific_anomalies": [] }
}"""
    else:
        prompt = args.prompt
    
    # Validate paths
    if not os.path.exists(args.mvtec_path):
        logger.error(f"MVTec dataset path does not exist: {args.mvtec_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference engine
    logger.info("Initializing LLaVA inference engine...")
    inference_engine = MVTecAnomalyInference(model_path=args.model_path, device=args.device)
    
    # Process dataset
    results = inference_engine.process_mvtec_dataset(
        mvtec_root_path=args.mvtec_path,
        prompt=prompt,
        output_path=args.output_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )
    
    logger.info(f"Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()