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
from llava.mm_utils import tokenizer_image_token

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToAnomalyInference:
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
    
    def create_anomaly_prediction_prompt(self, image_path):
        """Create prompt for anomaly type prediction based on image path"""
        
        prompt = f"""Based on the image path '{image_path}', predict what type of visual anomalies would be present.

Rules:
- If path contains '/test/good' → This is NORMAL, all anomalies should be FALSE
- If path contains '/test/cut' → Likely has cuts/holes (texture_pattern, surface_integrity, material_composition = TRUE)
- If path contains '/test/thread' → Likely has thread issues (texture_pattern = TRUE)
- If path contains '/test/hole' → Likely has holes (texture_pattern, surface_integrity, material_composition = TRUE)
- If path contains '/test/color' → Likely has color issues (color_photometry = TRUE)
- If path contains '/test/bent' → Likely has shape issues (geometry_shape = TRUE)

For each TRUE category, provide 1-3 short, concrete anomaly descriptions (≤7 words).

Return JSON only in this exact schema:

{{
  "geometry_shape": {{ "is_present": false, "specific_anomalies": [] }},
  "color_photometry": {{ "is_present": false, "specific_anomalies": [] }},
  "texture_pattern": {{ "is_present": false, "specific_anomalies": [] }},
  "surface_integrity": {{ "is_present": false, "specific_anomalies": [] }},
  "material_composition": {{ "is_present": false, "specific_anomalies": [] }}
}}"""
        
        return prompt
    
    def run_text_inference(self, text_prompt, temperature=0.1, top_p=0.9, max_new_tokens=512):
        """Run inference on text-only prompt (no image)"""
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
            logger.error(f"Error in text inference: {e}")
            return f"Error: {str(e)}"
    
    def process_json_file(self, input_json_path, output_json_path, temperature=0.1, top_p=0.9, max_new_tokens=512):
        """Process JSON file and generate anomaly predictions"""
        
        # Load input JSON
        logger.info(f"Loading input JSON from {input_json_path}")
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        logger.info(f"Found {len(input_data)} entries to process")
        
        # Process each entry
        results = {}
        failed_entries = []
        
        for image_path, metadata in tqdm(input_data.items(), desc="Processing entries"):
            tqdm.write(f"🔄 Processing: {image_path}")
            
            try:
                # Create prediction prompt based on image path
                prompt = self.create_anomaly_prediction_prompt(image_path)
                
                # Run text-based inference
                response = self.run_text_inference(
                    prompt, temperature, top_p, max_new_tokens
                )
                
                # Display response in real-time
                print(f"\n{'='*80}")
                print(f"📁 Path: {image_path}")
                print(f"📝 Predicted Response:")
                print(f"{response}")
                print(f"{'='*80}\n")
                
                # Store result with same format as input
                results[image_path] = {
                    'class': metadata.get('class', ''),
                    'subset': metadata.get('subset', ''),
                    'filename': metadata.get('filename', ''),
                    'response': response
                }
                
            except Exception as e:
                tqdm.write(f"❌ ERROR: Failed to process {image_path}: {e}")
                failed_entries.append(image_path)
                results[image_path] = {
                    'class': metadata.get('class', ''),
                    'subset': metadata.get('subset', ''),
                    'filename': metadata.get('filename', ''),
                    'response': f"Error: {str(e)}"
                }
        
        # Save results
        logger.info(f"Saving results to {output_json_path}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        total_entries = len(input_data)
        successful = total_entries - len(failed_entries)
        
        logger.info(f"Processing complete!")
        logger.info(f"Total entries: {total_entries}")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Failed: {len(failed_entries)}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Text-to-Anomaly Prediction with LLaVA')
    
    # Required arguments
    parser.add_argument('--input-json', required=True,
                       help='Path to input JSON file (like anomaly_results.json)')
    parser.add_argument('--output-json', required=True,
                       help='Path to save predicted results JSON file')
    
    # Optional arguments
    parser.add_argument('--model-path', default="liuhaotian/llava-v1.5-7b",
                       help='Path to LLaVA model (default: liuhaotian/llava-v1.5-7b)')
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"],
                       help='Device to use for inference (default: cuda)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for text generation (default: 0.1)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Top-p for text generation (default: 0.9)')
    parser.add_argument('--max-new-tokens', type=int, default=512,
                       help='Maximum new tokens to generate (default: 512)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.input_json).exists():
        logger.error(f"Input JSON file does not exist: {args.input_json}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_json).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    logger.info("Initializing LLaVA text inference engine...")
    inference_engine = TextToAnomalyInference(model_path=args.model_path, device=args.device)
    
    # Process JSON file
    results = inference_engine.process_json_file(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )
    
    logger.info(f"Results saved to: {args.output_json}")

if __name__ == "__main__":
    main()
