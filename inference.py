import argparse
import torch
from PIL import Image
import json
import os
import logging
from datetime import datetime
from tqdm import tqdm

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


def setup_logger(log_file):
    """Setup logger to write to both console and file"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_test_data(json_file, base_path):
    """Load test data from JSON file"""
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Extract all image paths from test data
    all_images = []
    for class_name, images in data["test"].items():
        for img_info in images:
            img_info["full_path"] = os.path.join(base_path, img_info["img_path"])
            all_images.append(img_info)
    
    return all_images


def process_single_image(image_info, model, tokenizer, image_processor, args, logger):
    """Process a single image and return result"""
    try:
        # Load and process image
        image_path = image_info["full_path"]
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            return None
            
        image = Image.open(image_path).convert("RGB")
        
        # Create prompt
        if args.prompt == "default":
            if image_info["anomaly"] == 1:
                prompt = f"This is a {image_info['cls_name']} image with {image_info['specie_name']} defects. Describe what you see in detail."
            else:
                prompt = f"This is a {image_info['cls_name']} image. Describe what you see and check if there are any defects."
        else:
            prompt = args.prompt
        
        # Prepare conversation
        use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        if use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_for_model = conv.get_prompt()
        
        # Process image and tokenize
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=model.dtype)
        
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
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=10,  # ensure at least some output
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode result
        input_token_len = input_ids.shape[1]
        generated_tokens = output_ids[0, input_token_len:]
        generated_text = tokenizer.batch_decode([generated_tokens], skip_special_tokens=True)[0].strip()
        
        if not generated_text:
            generated_text = "[EMPTY RESPONSE]"
        
        # Create result
        result = {
            "img_path": image_info["img_path"],
            "cls_name": image_info["cls_name"],
            "specie_name": image_info["specie_name"],
            "anomaly": image_info["anomaly"],
            "prompt": prompt,
            "generated_text": generated_text
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {image_info['img_path']}: {str(e)}")
        return None


def main(args):
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/batch_inference_{timestamp}.txt"
    logger = setup_logger(log_file)
    
    logger.info(f"Starting batch LLaVA inference")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Base path: {args.base_path}")
    
    # Load test data
    logger.info(f"Loading test data from: {args.json_file}")
    all_images = load_test_data(args.json_file, args.base_path)
    logger.info(f"Total images to process: {len(all_images)}")
    
    # Initialize model
    disable_torch_init()
    logger.info(f"Loading model: {args.model_path}")
    model_name = get_model_name_from_path(args.model_path)

    # BitsAndBytes quantization settings
    load_kwargs = {}
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    elif args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=args.device, **load_kwargs
    )
    model.eval()
    logger.info("Model loaded successfully")
    
    # Process all images
    results = []
    successful = 0
    failed = 0
    
    for i, image_info in enumerate(tqdm(all_images, desc="Processing images")):
        logger.info(f"Processing {i+1}/{len(all_images)}: {image_info['img_path']}")
        
        result = process_single_image(image_info, model, tokenizer, image_processor, args, logger)
        
        if result is not None:
            results.append(result)
            successful += 1
            
            # Log result
            logger.info("=" * 80)
            logger.info(f"Image: {result['img_path']}")
            logger.info(f"Class: {result['cls_name']}, Defect: {result['specie_name']}, Anomaly: {result['anomaly']}")
            logger.info(f"Prompt: {result['prompt']}")
            logger.info("Generated Text:")
            logger.info(result['generated_text'])
            logger.info("=" * 80)
        else:
            failed += 1
    
    # Save results
    results_file = f"results/batch_results_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Batch processing completed!")
    logger.info(f"Successful: {successful}, Failed: {failed}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch LLaVA inference for anomaly detection")
    parser.add_argument("--model-path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--json-file", type=str, default="/mnt/HDD/yoonji/anomalyCBM/data/DTD/meta.json", help="JSON file with test data")
    parser.add_argument("--base-path", type=str, default="/mnt/HDD/yoonji/anomalyCBM/data/DTD", help="Base path for images")
    parser.add_argument("--prompt", type=str, default="default", help="Prompt template (use 'default' for auto-generated prompts)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p filtering")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")

    # Quantization flags
    parser.add_argument("--load-in-8bit", action="store_true", default=True, help="Load model in 8-bit (reduce VRAM)")
    # parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (reduce VRAM further)")

    args = parser.parse_args()
    main(args)
