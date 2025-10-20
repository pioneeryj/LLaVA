python annotate_anomaly_type_llava_1020.py \
  --mvtec-path /mnt/HDD/yoonji/anomalyCBM/data/mvtec \
  --output-path results/anomaly_results.json \
  --model-path liuhaotian/llava-v1.5-7b \
  --device cuda \
  --temperature 0.1 \
  --top-p 0.9 \
  --max-new-tokens 1024 \
  --prompt "You are a visual inspector. Using only the image, decide for each category if a visual anomaly exists (no external context/logic like missing/wrong part).
Categories: Geometry & Shape, Color & Photometry, Texture & Pattern, Surface Integrity (Damage), Material & Composition.
If present, list 1–3 short, concrete, visual-only anomaly texts (≤7 words).
Return JSON only in this exact schema:

{
  \"geometry_shape\": { \"is_present\": false, \"specific_anomalies\": [] },
  \"color_photometry\": { \"is_present\": false, \"specific_anomalies\": [] },
  \"texture_pattern\": { \"is_present\": false, \"specific_anomalies\": [] },
  \"surface_integrity\": { \"is_present\": false, \"specific_anomalies\": [] },
  \"material_composition\": { \"is_present\": false, \"specific_anomalies\": [] }
}"