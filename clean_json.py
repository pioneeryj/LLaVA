import json
import re
import argparse
from pathlib import Path

def clean_anomaly_text(text):
    """Clean up individual anomaly text"""
    if not text:
        return None
    
    # Remove quotes and escape characters
    text = text.replace('\\"', '"').replace("\\", "")
    
    # Remove quotes at beginning and end
    text = text.strip().strip('"').strip("'")
    
    # Handle cases like "uneven surface.\" \"Deformed shape"
    # Split by \" \" pattern and take the first valid part
    if '\\" \\"' in text or '" "' in text:
        parts = re.split(r'["\'"]\s*["\'"]\s*', text)
        text = parts[0].strip() if parts else text
    
    # Remove trailing periods and quotes
    text = re.sub(r'[."\'"]+$', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check if valid (2-7 words, no special characters)
    if (text and 
        2 <= len(text.split()) <= 7 and 
        not any(char in text for char in ['\\', '{', '}', '[', ']']) and
        not text.lower().startswith(('error', 'sorry', 'cannot'))):
        return text
    
    return None

def clean_json_file(input_path, output_path, target_count_per_category=30):
    """Clean the malformed JSON file"""
    
    print(f"Loading JSON from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    cleaned_data = {}
    
    # Predefined clean texts for each category as fallback
    fallback_texts = {
        "geometry_shape": [
            "bent corner visible", "curved edge present", "warped surface area",
            "twisted section shown", "deformed shape detected", "irregular outline visible",
            "asymmetric structure present", "compressed area shown", "stretched region visible",
            "distorted form present", "skewed angle visible", "bowed edge present",
            "folded section shown", "creased line visible", "bulging area present",
            "irregular contour shown", "misaligned edge visible", "crooked line present",
            "uneven border shown", "deformed corner visible", "twisted edge present",
            "warped corner shown", "irregular shape visible", "distorted angle present",
            "bent section shown", "curved surface visible", "asymmetric edge present",
            "deformed border shown", "irregular line visible", "twisted corner present"
        ],
        "color_photometry": [
            "faded color patch", "dark spot visible", "bright area present",
            "color variation shown", "discolored region detected", "pale section visible",
            "saturated area present", "color bleeding edge", "tinted region shown",
            "color shift detected", "washed out area", "overexposed region visible",
            "underexposed area present", "color cast detected", "hue shift visible",
            "brightness variation shown", "contrast issue present", "desaturated area visible",
            "muted color region", "vibrant spot present", "color distortion shown",
            "tonal shift visible", "luminance change present", "chromatic aberration shown",
            "color balance issue", "saturation loss visible", "color temperature shift",
            "exposure variation present", "color gradient issue", "tint variation shown"
        ],
        "texture_pattern": [
            "rough surface area", "smooth patch present", "irregular pattern shown",
            "texture variation visible", "pattern disruption detected", "weave irregularity present",
            "surface roughness shown", "texture change visible", "pattern break detected",
            "surface inconsistency present", "grain variation shown", "texture distortion visible",
            "pattern misalignment present", "surface texture change", "weave pattern break",
            "texture gradient issue", "pattern repetition error", "surface finish variation",
            "texture density change", "pattern scale variation", "surface grain change",
            "texture orientation shift", "pattern contrast issue", "surface porosity change",
            "texture uniformity loss", "pattern clarity reduction", "surface texture loss",
            "texture detail missing", "pattern definition loss", "surface smoothness change"
        ],
        "surface_integrity": [
            "small hole present", "crack line visible", "scratch mark shown",
            "dent impression detected", "tear edge present", "chip damage visible",
            "surface break shown", "puncture hole detected", "abrasion mark present",
            "impact damage visible", "surface cut shown", "gouge mark present",
            "surface pit visible", "nick damage shown", "surface score present",
            "cut mark visible", "surface groove shown", "damage streak present",
            "surface wear visible", "erosion mark shown", "surface defect present",
            "damage spot visible", "surface flaw shown", "integrity loss present",
            "surface breach visible", "damage line shown", "surface fault present",
            "structural damage visible", "surface compromise shown", "integrity issue present"
        ],
        "material_composition": [
            "foreign material visible", "inclusion spot present", "material change shown",
            "composition variation detected", "different material present", "contamination visible",
            "material defect shown", "impurity detected", "composition anomaly present",
            "material inconsistency visible", "foreign particle present", "inclusion defect shown",
            "material mixture visible", "composition irregularity present", "material variation shown",
            "foreign substance visible", "material contamination present", "composition change shown",
            "material impurity visible", "inclusion anomaly present", "material degradation shown",
            "composition defect visible", "material alteration present", "foreign element shown",
            "material heterogeneity visible", "composition disturbance present", "material deviation shown",
            "composition modification visible", "material transformation present", "composition error shown"
        ]
    }
    
    for category, texts in data.items():
        print(f"\nProcessing category: {category}")
        print(f"Original count: {len(texts)}")
        
        cleaned_texts = []
        
        # Clean existing texts
        for text in texts:
            cleaned = clean_anomaly_text(text)
            if cleaned and cleaned not in cleaned_texts:
                cleaned_texts.append(cleaned)
        
        print(f"Cleaned count: {len(cleaned_texts)}")
        
        # If not enough clean texts, add fallbacks
        if len(cleaned_texts) < target_count_per_category:
            needed = target_count_per_category - len(cleaned_texts)
            if category in fallback_texts:
                fallbacks = fallback_texts[category][:needed]
                for fallback in fallbacks:
                    if fallback not in cleaned_texts:
                        cleaned_texts.append(fallback)
        
        # Ensure we have exactly the target count
        cleaned_texts = cleaned_texts[:target_count_per_category]
        
        cleaned_data[category] = cleaned_texts
        print(f"Final count: {len(cleaned_texts)}")
        
        # Show first 5 examples
        print("Examples:")
        for i, text in enumerate(cleaned_texts[:5]):
            print(f"  {i+1}. {text}")
    
    # Save cleaned data
    print(f"\nSaving cleaned JSON to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    for category, texts in cleaned_data.items():
        print(f"{category}: {len(texts)} texts")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Clean malformed anomaly category JSON')
    
    parser.add_argument('--input-json', required=True,
                       help='Path to input JSON file to clean')
    parser.add_argument('--output-json', required=True,
                       help='Path to save cleaned JSON file')
    parser.add_argument('--target-count', type=int, default=30,
                       help='Target number of texts per category (default: 30)')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_json).exists():
        print(f"Error: Input file does not exist: {args.input_json}")
        return
    
    # Create output directory if needed
    output_dir = Path(args.output_json).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean the JSON file
    clean_json_file(args.input_json, args.output_json, args.target_count)
    
    print(f"\nCleaning complete! Results saved to: {args.output_json}")

if __name__ == "__main__":
    main()
