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

class MVTecAnomalyLabelInference:
    def __init__(self, model_path="liuhaotian/llava-v1.5-7b", device="cuda"):
        self.device = device
        disable_torch_init()
        
        logger.info(f"Loading LLaVA model from {model_path}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name="llava-v1.5-7b",
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
    
    def create_anomaly_label_prompt(self, image_path):
        """Create prompt for anomaly label prediction"""
        
        # Extract information from image path
        path_parts = image_path.split('/')
        object_type = "unknown"
        anomaly_type = "unknown"
        
        # Find object type and anomaly type from path
        for i, part in enumerate(path_parts):
            if part == "test" and i > 0:
                object_type = path_parts[i-1]
            if part == "test" and i < len(path_parts) - 1:
                anomaly_type = path_parts[i+1]
        
        # Check if it's a normal (good) image
        is_normal = "good" in image_path.lower() or "/good/" in image_path
        
        prompt = f"""Analyze this image with the given path information:

Image Path: {image_path}
Object Type: {object_type}
Anomaly Type from Path: {anomaly_type}
Is Normal Image: {is_normal}

IMPORTANT RULES:
1. If the path contains 'good' or '/good/', this is a NORMAL image → Return [0,0,0,0,0]
2. Otherwise, analyze the image AND path information together

Anomaly Categories (in order):
0. geometry_shape: Shape deformations, bending, warping, size changes
1. color_photometry: Color variations, discoloration, brightness changes  
2. texture_pattern: Surface texture changes, pattern disruptions, weave issues
3. surface_integrity: Physical damage, scratches, dents, cracks, holes, tears
4. material_composition: Material defects, foreign materials, inclusions

Based on BOTH the image content AND the path information, determine which anomaly categories are present.
Return ONLY a JSON with "label" key containing a list of 5 binary values [0 or 1]:

Example outputs:
- Normal image: {{"label": [0,0,0,0,0]}}
- Shape + Surface damage: {{"label": [1,0,0,1,0]}}
- Only texture issues: {{"label": [0,0,1,0,0]}}

Analyze the image and return the label:"""
        
        return prompt
    
    def run_inference(self, image, prompt, temperature=0.1, top_p=0.9, max_new_tokens=256):
        """Run inference on image with prompt"""
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
            
            # Extract only the assistant's response
            if conv.roles[1] in outputs:
                response = outputs.split(conv.roles[1])[-1].strip()
            else:
                response = outputs
            
            return response
            
        except Exception as e:
            logger.error(f"Error in inference: {e}")
            return f"Error: {str(e)}"
    
    def parse_label_from_response(self, response, image_path):
        """Parse binary label from model response"""
        try:
            # Check if it's a normal image first
            if "good" in image_path.lower() or "/good/" in image_path:
                return [0, 0, 0, 0, 0]
            
            # Try to extract JSON from response
            import re
            
            # Look for JSON pattern
            json_match = re.search(r'\{[^}]*"label"[^}]*\}', response)
            if json_match:
                json_str = json_match.group()
                try:
                    parsed = json.loads(json_str)
                    if "label" in parsed and isinstance(parsed["label"], list):
                        label = parsed["label"][:5]  # Take first 5 elements
                        # Ensure all elements are 0 or 1
                        label = [1 if x else 0 for x in label]
                        # Pad with zeros if needed
                        while len(label) < 5:
                            label.append(0)
                        return label
                except json.JSONDecodeError:
                    pass
            
            # Look for array pattern like [1,0,0,1,0]
            array_match = re.search(r'\[[\d\s,]+\]', response)
            if array_match:
                array_str = array_match.group()
                try:
                    # Extract numbers
                    numbers = re.findall(r'\d', array_str)
                    if len(numbers) >= 5:
                        label = [int(x) for x in numbers[:5]]
                        return label
                except:
                    pass
            
            # Fallback: analyze path for basic labeling
            return self.get_label_from_path(image_path)
            
        except Exception as e:
            logger.error(f"Error parsing label: {e}")
            return self.get_label_from_path(image_path)
    
    def get_label_from_path(self, image_path):
        """Fallback: get label based on path analysis"""
        if "good" in image_path.lower() or "/good/" in image_path:
            return [0, 0, 0, 0, 0]
        
        path_lower = image_path.lower()
        label = [0, 0, 0, 0, 0]
        
        # Basic path-based labeling
        if any(x in path_lower for x in ['bent', 'curve', 'warp', 'twist', 'deform']):
            label[0] = 1  # geometry_shape
        if any(x in path_lower for x in ['color', 'fade', 'bright', 'dark', 'tint']):
            label[1] = 1  # color_photometry  
        if any(x in path_lower for x in ['texture', 'pattern', 'weave', 'thread', 'grain']):
            label[2] = 1  # texture_pattern
        if any(x in path_lower for x in ['crack', 'hole', 'tear', 'cut', 'scratch', 'dent', 'break']):
            label[3] = 1  # surface_integrity
        if any(x in path_lower for x in ['material', 'foreign', 'inclusion', 'contaminat']):
            label[4] = 1  # material_composition
        
        return label
    
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
    
    def process_mvtec_dataset(self, mvtec_root_path, output_path, temperature=0.1, top_p=0.9, max_new_tokens=256):
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
            
            tqdm.write(f"🔄 Processing: {relative_path}")
            
            try:
                # Load image
                image = self.load_image(img_path)
                if image is None:
                    failed_images.append(relative_path)
                    results[relative_path] = {
                        'class': item['class'],
                        'subset': item['subset'], 
                        'filename': item['filename'],
                        'label': [0, 0, 0, 0, 0],
                        'response': "Error: Failed to load image"
                    }
                    continue
                
                # Create prompt
                prompt = self.create_anomaly_label_prompt(relative_path)
                
                # Run inference
                response = self.run_inference(
                    image, prompt, temperature, top_p, max_new_tokens
                )
                
                # Parse label from response
                label = self.parse_label_from_response(response, relative_path)
                
                # Display response in real-time
                print(f"\n{'='*80}")
                print(f"🖼️  Image: {relative_path}")
                print(f"📝 Label: {label}")
                print(f"📄 Response: {response[:200]}...")
                print(f"{'='*80}\n")
                
                # Store result with metadata and label
                results[relative_path] = {
                    'class': item['class'],
                    'subset': item['subset'],
                    'filename': item['filename'],
                    'label': label,
                    'response': response
                }
                
            except Exception as e:
                tqdm.write(f"❌ ERROR: Failed to process {relative_path}: {e}")
                failed_images.append(relative_path)
                results[relative_path] = {
                    'class': item['class'],
                    'subset': item['subset'],
                    'filename': item['filename'],
                    'label': [0, 0, 0, 0, 0],
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
        
        # Print label statistics
        label_stats = {}
        normal_count = 0
        for image_path, data in results.items():
            label = data.get('label', [0, 0, 0, 0, 0])
            label_key = str(label)
            if label_key not in label_stats:
                label_stats[label_key] = 0
            label_stats[label_key] += 1
            
            if label == [0, 0, 0, 0, 0]:
                normal_count += 1
        
        logger.info("Label distribution:")
        logger.info(f"  Normal [0,0,0,0,0]: {normal_count} images")
        for label_str, count in sorted(label_stats.items()):
            if label_str != "[0, 0, 0, 0, 0]":
                logger.info(f"  {label_str}: {count} images")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='MVTec Anomaly Label Generation with LLaVA')
    
    # Required arguments
    parser.add_argument('--mvtec-path', required=True, 
                       help='Path to MVTec dataset root directory')
    parser.add_argument('--output-path', required=True,
                       help='Path to save results JSON file')
    
    # Optional arguments
    parser.add_argument('--model-path', default="liuhaotian/llava-v1.5-7b",
                       help='Path to LLaVA model (default: liuhaotian/llava-v1.5-7b)')
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"],
                       help='Device to use for inference (default: cuda)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for text generation (default: 0.1)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p for text generation (default: 0.9)')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                       help='Maximum new tokens to generate (default: 256)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.mvtec_path):
        logger.error(f"MVTec dataset path does not exist: {args.mvtec_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference engine
    logger.info("Initializing LLaVA label inference engine...")
    inference_engine = MVTecAnomalyLabelInference(model_path=args.model_path, device=args.device)
    
    # Process dataset
    results = inference_engine.process_mvtec_dataset(
        mvtec_root_path=args.mvtec_path,
        output_path=args.output_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )
    
    logger.info(f"Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()