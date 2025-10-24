import torch
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyTextGenerator:
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
    
    def create_category_prompt(self, category_name, category_description, num_texts=30):
        """Create simple prompt for generating anomaly texts"""
        
        prompt = f"""Generate {num_texts} short visual anomaly descriptions for "{category_name}".

Category: {category_description}

Rules:
- Each description: 2-5 words only
- Visual defects only
- Return as comma-separated list
- NO numbers, NO bullets, NO newlines
- Format: "text1, text2, text3, ..."

Examples: "small hole visible, crack line present, bent corner shown"

Generate {num_texts} descriptions:"""
        
        return prompt
    
    def run_text_generation(self, text_prompt, temperature=0.7, top_p=0.9, max_new_tokens=1024):
        """Run text generation"""
        try:
            # Prepare conversation for text-only
            conv_mode = "llava_v1"
            conv = conv_templates[conv_mode].copy()
            
            # Add text prompt directly (no image token)
            conv.append_message(conv.roles[0], text_prompt)
            conv.append_message(conv.roles[1], None)
            prompt_formatted = conv.get_prompt()
            
            # Tokenize
            input_ids = self.tokenizer(
                prompt_formatted,
                return_tensors='pt'
            )['input_ids'].to(self.device)
            
            # Generate response (text-only, no images)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
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
            logger.error(f"Error in text generation: {e}")
            return f"Error: {str(e)}"
    
    def parse_generated_texts(self, generated_text, target_count=30):
        """Simple parsing with robust cleaning"""
        import re
        
        try:
            # Clean up the text thoroughly
            cleaned = generated_text.strip()
            
            # Remove newlines and extra spaces
            cleaned = re.sub(r'\n+', ' ', cleaned)
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # Remove numbered lists (1., 2., etc.)
            cleaned = re.sub(r'\d+\.\s*', '', cleaned)
            
            # Split by commas
            texts = [t.strip() for t in cleaned.split(',')]
            
            valid_texts = []
            for text in texts:
                if not text:
                    continue
                
                # Clean each text
                text = text.strip().strip('"').strip("'")
                text = re.sub(r'^[-•*]\s*', '', text)  # Remove bullets
                text = text.strip()
                
                # Check validity
                if (text and 
                    2 <= len(text.split()) <= 5 and  # 2-5 words
                    not any(skip in text.lower() for skip in ['error', 'sorry', 'cannot', 'example']) and
                    text not in valid_texts):
                    valid_texts.append(text)
                
                if len(valid_texts) >= target_count:
                    break
            
            # If not enough, add simple fallbacks
            if len(valid_texts) < target_count:
                fallbacks = self.get_fallback_texts(target_count - len(valid_texts))
                valid_texts.extend(fallbacks)
            
            return valid_texts[:target_count]
            
        except Exception as e:
            logger.error(f"Error parsing: {e}")
            return self.get_fallback_texts(target_count)
    
    def get_fallback_texts(self, count):
        """Simple fallback texts"""
        fallbacks = [
            "small crack visible", "surface damage present", "color change detected",
            "texture variation shown", "material defect present", "shape distortion visible",
            "hole damage present", "scratch mark visible", "dent impression shown",
            "tear edge present", "bend deformation visible", "wear pattern shown",
            "stain mark present", "chip damage visible", "rough surface area",
            "smooth patch present", "dark spot visible", "bright area present",
            "pattern break visible", "surface irregularity present"
        ]
        return fallbacks[:count]
    
    def generate_all_categories(self, output_path, texts_per_category=30, temperature=0.7, top_p=0.9, max_new_tokens=512):
        """Generate specific anomaly texts for all categories"""
        
        # Define anomaly categories and their descriptions
        categories = {
            "geometry_shape": "Shape deformations, bending, warping, size changes, dimensional anomalies, structural distortions",
            "color_photometry": "Color variations, discoloration, brightness changes, contrast issues, hue shifts, saturation problems",
            "texture_pattern": "Surface texture changes, pattern disruptions, weave issues, grain problems, surface roughness variations",
            "surface_integrity": "Physical damage to surface, scratches, dents, cracks, holes, tears, surface breaks",
            "material_composition": "Material defects, composition changes, foreign materials, inclusions, material degradation"
        }
        
        results = {}
        
        for category_name, description in categories.items():
            logger.info(f"Generating texts for category: {category_name}")
            tqdm.write(f"🔄 Processing category: {category_name}")
            
            try:
                # Create prompt for this category
                prompt = self.create_category_prompt(category_name, description, texts_per_category)
                
                # Generate texts
                generated_response = self.run_text_generation(
                    prompt, temperature, top_p, max_new_tokens
                )
                
                # Parse and clean the generated texts
                parsed_texts = self.parse_generated_texts(generated_response, texts_per_category)
                
                # Display some examples
                print(f"\n{'='*60}")
                print(f"📂 Category: {category_name}")
                print(f"📝 Generated {len(parsed_texts)} texts")
                print(f"Examples (first 5):")
                for i, text in enumerate(parsed_texts[:5]):
                    print(f"  {i+1}. {text}")
                print(f"{'='*60}\n")
                
                results[category_name] = parsed_texts
                
            except Exception as e:
                logger.error(f"Failed to process category {category_name}: {e}")
                results[category_name] = []
        
        # Save results
        logger.info(f"Saving results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info(f"Generation complete!")
        for category, texts in results.items():
            logger.info(f"  {category}: {len(texts)} texts generated")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Generate Anomaly Category Texts with LLaVA')
    
    # Required arguments
    parser.add_argument('--output-json', required=True,
                       help='Path to save generated texts JSON file')
    
    # Optional arguments
    parser.add_argument('--model-path', default="liuhaotian/llava-v1.5-7b",
                       help='Path to LLaVA model (default: liuhaotian/llava-v1.5-7b)')
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"],
                       help='Device to use for inference (default: cuda)')
    parser.add_argument('--texts-per-category', type=int, default=30,
                       help='Number of texts to generate per category (default: 30)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for text generation (default: 0.7)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p for text generation (default: 0.9)')
    parser.add_argument('--max-new-tokens', type=int, default=1024,
                       help='Maximum new tokens to generate (default: 1024)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_json).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize text generator
    logger.info("Initializing LLaVA text generator...")
    generator = AnomalyTextGenerator(model_path=args.model_path, device=args.device)
    
    # Generate texts for all categories
    results = generator.generate_all_categories(
        output_path=args.output_json,
        texts_per_category=args.texts_per_category,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )
    
    logger.info(f"Results saved to: {args.output_json}")

if __name__ == "__main__":
    main()