
import AnomalyCLIP_lib
import torch
import torch.nn.functional as F
import json
import numpy as np
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import os
import random
import argparse
from utils import get_transform
from prompt_ensemble import tokenize
from cbm_model import ConceptBottleneckModel
from metrics import image_level_metrics, pixel_level_metrics
from tabulate import tabulate
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import time
import traceback
import glob
import sys
import torch.nn as nn
import math

def interpolate_pos_embed(model, checkpoint_state_dict):
    """
    Interpolate positional embeddings to match the current model size
    """
    if 'visual.positional_embedding' in checkpoint_state_dict:
        pos_embed_checkpoint = checkpoint_state_dict['visual.positional_embedding']
        pos_embed_model = model.visual.positional_embedding
        
        print(f"Checkpoint pos_embed shape: {pos_embed_checkpoint.shape}")
        print(f"Model pos_embed shape: {pos_embed_model.shape}")
        
        if pos_embed_checkpoint.shape != pos_embed_model.shape:
            print("Interpolating positional embeddings...")
            
            embedding_size = pos_embed_checkpoint.shape[-1]
            
            # For Vision Transformer, typically first token is CLS token
            num_extra_tokens = 1  # Assuming 1 CLS token
            
            # Extract CLS token and position tokens
            if pos_embed_checkpoint.shape[0] > num_extra_tokens:
                extra_tokens = pos_embed_checkpoint[:num_extra_tokens]  # [1, embedding_size]
                pos_tokens = pos_embed_checkpoint[num_extra_tokens:]     # [H*W, embedding_size]
            else:
                extra_tokens = None
                pos_tokens = pos_embed_checkpoint
            
            # Calculate grid sizes
            orig_size = int((pos_tokens.shape[0]) ** 0.5)
            
            # Target size from current model
            target_patches = pos_embed_model.shape[0] - num_extra_tokens
            new_size = int(target_patches ** 0.5)
            
            print(f"Interpolating from {orig_size}x{orig_size} to {new_size}x{new_size}")
            
            if orig_size != new_size:
                # Reshape to 2D grid: [H*W, embedding_size] -> [1, H, W, embedding_size] -> [1, embedding_size, H, W]
                pos_tokens_2d = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                
                # Interpolate
                pos_tokens_interp = F.interpolate(
                    pos_tokens_2d, 
                    size=(new_size, new_size), 
                    mode='bicubic', 
                    align_corners=False
                )
                
                # Reshape back: [1, embedding_size, H, W] -> [1, H, W, embedding_size] -> [H*W, embedding_size]
                pos_tokens = pos_tokens_interp.permute(0, 2, 3, 1).reshape(new_size * new_size, embedding_size)
            
            # Combine CLS token and position tokens
            if extra_tokens is not None:
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
            else:
                new_pos_embed = pos_tokens
            
            print(f"New pos_embed shape: {new_pos_embed.shape}")
            checkpoint_state_dict['visual.positional_embedding'] = new_pos_embed
        else:
            print("Positional embeddings already match, no interpolation needed.")


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 원래 weight는 freeze (pretrained weight)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False  # freeze original weight

        # LoRA low-rank 행렬
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        if self.r > 0:
            # frozen original weight
            result = x @ self.weight.T
            # low-rank adaptation
            lora_update = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
            return result + lora_update + self.bias
        else:
            return x @ self.weight.T + self.bias
        
def apply_lora_to_anomalyclip(model, r=8, alpha=16, target_modules=None):
    """
    model: AnomalyCLIP_lib.load(...)로 로딩된 모델
    r, alpha: LoRA 설정
    target_modules: LoRA로 바꾸고 싶은 모듈 이름 리스트 (예: ['text_projection','visual.transformer.resblocks.0.attn.in_proj'])
    """
    if target_modules is None:
        # 기본값: 텍스트 프로젝션만
        target_modules = ['text_projection']

    # named_modules()는 모듈 트리 전체를 반환
    for name, module in model.named_modules():
        for target_name in target_modules:
            if target_name in name and isinstance(module, nn.Linear):
                in_f = module.in_features
                out_f = module.out_features
                lora_layer = LoRALinear(in_f, out_f, r=r, alpha=alpha)
                # pretrained weight 복사
                with torch.no_grad():
                    lora_layer.weight.copy_(module.weight)
                    lora_layer.bias.copy_(module.bias)

                # 부모 모듈 찾아 교체
                parent_path = name.rsplit('.', 1)
                if len(parent_path) == 2:
                    parent_name, attr_name = parent_path
                    parent_module = dict(model.named_modules())[parent_name]
                    setattr(parent_module, attr_name, lora_layer)
                else:
                    # top-level attribute
                    setattr(model, name, lora_layer)

                print(f"Replaced {name} with LoRALinear(r={r},alpha={alpha})")
    return model



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    
def test(args):
    """Main testing function for CBM model (finetune version)"""
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get data transforms
    preprocess, target_transform = get_transform(args)
    
    logger.info(f"Testing with data path: {args.data_path}")
    
    # Check if data path exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        logger.info("Please check the path and run again.")
        return
    
    try:
        # Load test data
        test_data = Dataset(
            root=args.data_path, 
            transform=preprocess, 
            target_transform=target_transform, 
            dataset_name=args.dataset
        )
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        obj_list = test_data.obj_list
        logger.info(f"Found {len(obj_list)} object types: {obj_list}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        logger.info("Please check the dataset path and run again.")
        logger.error(traceback.format_exc())
        return
    
    # Load models
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device)
    model.eval()
    model_ano, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, 
                                       design_details={"Prompt_length": 9, 
                                                     "learnabel_text_embedding_depth": 12, 
                                                     "learnabel_text_embedding_length": 4})
    model_ano.eval()
    
    
    # Create concept bottleneck model
    cbm = ConceptBottleneckModel(args.concepts_file, model, device, input_size=args.image_size)
    cbm.to(device)
    cbm.eval()
    
    # checkpoint 불러오기
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Load CBM
        cbm.load_state_dict(checkpoint["cbm_state_dict"])
        logger.info(f"Loaded CBM weights from {args.checkpoint_path}")
        
        # Load text encoder
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            logger.info("Loaded text encoder weights")
        else:
            model.load_state_dict(checkpoint["text_encoder_state_dict"], strict=False)
            logger.info("Loaded text encoder weights")
        
        # Load image encoder (model_ano)
        if "model_ano_state_dict" in checkpoint:
            # model_ano.load_state_dict(checkpoint["model_ano_state_dict"], strict=False)
            interpolate_pos_embed(model_ano, checkpoint["model_ano_state_dict"])
            model_ano.load_state_dict(checkpoint["model_ano_state_dict"], strict=False)
            
            logger.info("Loaded image encoder weights")


    # Initialize results storage
    results = {}
    metrics = {}
    sample_data = []  # For MD file generation
    
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        results[obj]['img_paths'] = []  # Store image paths for visualization
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0
    
    # Apply DPAM
    model_ano.visual.DAPM_replace(DPAM_layer=20)

    # Add error handling around the dataloader iteration
    skipped_samples = 0
    total_samples = 0
    concept_embedding_list = []
    
    with torch.no_grad():
        torch.cuda.empty_cache()
        for concept in cbm.all_concepts:
            tokenized = tokenize([concept]).to(device)
            embedding = model.encode_text(tokenized)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            concept_embedding_list.append(embedding.squeeze(0))
        concept_embeddings = torch.stack(concept_embedding_list, dim=0)
    
    for idx, items in enumerate(tqdm(test_dataloader, desc="Testing samples")):
        try:
            total_samples += 1
            image = items['img'].to(device)
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            
            results[cls_name[0]]['imgs_masks'].append(gt_mask)
            results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())
            # Store image path for visualization
            if 'img_path' in items:
                results[cls_name[0]]['img_paths'].append(items['img_path'][0])  # items['img_path'] is a list
            else:
                results[cls_name[0]]['img_paths'].append("")  # Placeholder if path not available
        except Exception as e:
            skipped_samples += 1
            logger.warning(f"Error processing sample {idx}: {str(e)}")
            continue
        
        with torch.no_grad():
            # Get image features
            torch.cuda.empty_cache()
            image_features, patch_features = model_ano.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            print(f'len patch features: {len(patch_features)}')
            
            # Get CBM predictions (concept embeddings already updated)
            class_logits, concept_scores = cbm(image_features, concept_embeddings)
            class_probs = torch.softmax(class_logits / 0.07, dim=1)
            image_score = class_probs[:, 1].cpu().numpy()[0]  # Anomaly probability
            results[cls_name[0]]['pr_sp'].extend(class_probs[:, 1].detach().cpu())
            
            # Get concept scores for max selection (same as train_cbm_finetune.py)
            normal_concept_scores = concept_scores[:, cbm.normal_indices]  # [batch, num_normal]
            anomaly_concept_scores = concept_scores[:, cbm.anomaly_indices]  # [batch, num_anomaly]
            
            # Find the concept with highest similarity in each category (same as train_cbm_finetune.py)
            # For normal concepts
            _, max_normal_idx = torch.max(normal_concept_scores, dim=1)  # [batch]
            selected_normal_embeddings = concept_embeddings[cbm.normal_indices][max_normal_idx]  # [batch, 768]
            
            # For anomaly concepts  
            _, max_anomaly_idx = torch.max(anomaly_concept_scores, dim=1)  # [batch]
            selected_anomaly_embeddings = concept_embeddings[cbm.anomaly_indices][max_anomaly_idx]  # [batch, 768]
            
            # Stack selected embeddings: [batch, 2, 768] (normal, anomaly)
            selected_text_features = torch.stack([selected_normal_embeddings, selected_anomaly_embeddings], dim=1)
            selected_text_features = selected_text_features / selected_text_features.norm(dim=-1, keepdim=True)
            
            # Generate anomaly maps using max concept approach - use only highest resolution feature map
            # Get the highest resolution feature map (last layer)
            highest_res_patch_feature = patch_features[-1]
            patch_feature = highest_res_patch_feature / highest_res_patch_feature.norm(dim=-1, keepdim=True)
            
            # Compute similarity with selected embeddings
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, selected_text_features[0])  # Use first batch item
            
            # Normal similarity map (index 0)
            normal_similarity = similarity[:, 1:, 0:1]  # [batch_size, num_patches, 1] - skip CLS token
            
            # Anomaly similarity map (index 1)
            anomaly_similarity = similarity[:, 1:, 1:2]  # [batch_size, num_patches, 1] - skip CLS token
            
            # Combine normal and anomaly similarities (2-channel approach)
            combined_similarity = torch.cat([normal_similarity, anomaly_similarity], dim=2)
            
            similarity_map = AnomalyCLIP_lib.get_similarity_map(combined_similarity, args.image_size).permute(0, 3, 1, 2)
            
            # segmentation head 사용시
            refined_seg_map = cbm.segmentation_head(similarity_map)
            seg_logits = refined_seg_map
            seg_probs = F.softmax(seg_logits, dim=1)

            # 기본 finetune code
            # anomaly_map = torch.stack(anomaly_map_list)
            # print(f"anomaly_map shape: {anomaly_map.shape}")  # Should be [num_layers, batch, H, W]
            # anomaly_map = anomaly_map.sum(dim = 0)
            # print(f"anomaly_map after sum shape: {anomaly_map.shape}") 
            
            # seg_probs_list = []
            # for sim_map in similarity_map_list:
            #     seg_probs = F.softmax(sim_map, dim=1)
            #     seg_probs_list.append(seg_probs[:, 1, :, :])  # Anomaly channel
            
            # anomaly_map = torch.stack(seg_probs_list).mean(dim=0).squeeze(0)
            
            # segmentation head 사용시
            anomaly_map = seg_probs[:,1,:,:]
            
            # Apply Gaussian filter (optimized to avoid CPU transfers)
            if args.sigma > 0:
                anomaly_map_cpu = anomaly_map.detach().cpu()
                anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map_cpu], dim=0)
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            
    logger.info(f"Processed {total_samples - skipped_samples} samples successfully, skipped {skipped_samples} samples")
    
    # Calculate metrics for each object
    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    
    
    for obj in obj_list:
        try:
            table = []
            table.append(obj)
            
            # Check if we have results for this object
            if not results[obj]['imgs_masks'] or not results[obj]['anomaly_maps']:
                logger.warning(f"No results found for object {obj}")
                continue
            
            # Concatenate results using the same approach as test_cbm_seg.py
            try:
                results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
                results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
                # Convert gt_sp and pr_sp to numpy arrays (lists of scalars)
                results[obj]['gt_sp'] = [tensor.item() if torch.is_tensor(tensor) else tensor for tensor in results[obj]['gt_sp']]
                results[obj]['pr_sp'] = [tensor.item() if torch.is_tensor(tensor) else tensor for tensor in results[obj]['pr_sp']]
            except Exception as e:
                logger.error(f"Error concatenating results for {obj}: {str(e)}")
                continue
            
            # Calculate all metrics using the same approach as test_cbm_seg.py
            try:
                pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
                pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
                image_auroc = image_level_metrics(results, obj, "image-auroc")
                image_ap = image_level_metrics(results, obj, "image-ap")
                
                table.extend([str(np.round(pixel_auroc * 100, decimals=1)),
                             str(np.round(pixel_aupro * 100, decimals=1)),
                             str(np.round(image_auroc * 100, decimals=1)),
                             str(np.round(image_ap * 100, decimals=1))])
                table_ls.append(table)
                
                # Store metrics
                metrics[obj]['pixel-auroc'] = pixel_auroc
                metrics[obj]['pixel-aupro'] = pixel_aupro
                metrics[obj]['image-auroc'] = image_auroc
                metrics[obj]['image-ap'] = image_ap
                
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap)
                
                # Save visualization if enabled and pixel AUROC >= 80%
                if args.visualization and pixel_auroc >= 0.8:
                    logger.info(f"Saving visualization for {obj} (Pixel AUROC: {pixel_auroc*100:.1f}%)")
                    save_visualization_heatmaps(obj, results, args, logger)
                    save_overlay_images(results[obj]['anomaly_maps'], results[obj]['gt_sp'], 
                                      results[obj]['img_paths'], obj, pixel_auroc*100, args, logger)
                    
                    # Add sample data for markdown report
                    sample_data.append({
                        'object': obj,
                        'pixel_auroc': pixel_auroc,
                        'pixel_aupro': pixel_aupro,
                        'image_auroc': image_auroc,
                        'image_ap': image_ap,
                        'num_samples': len(results[obj]['gt_sp']),
                        'num_anomalies': sum(results[obj]['gt_sp']),
                        'num_normals': len(results[obj]['gt_sp']) - sum(results[obj]['gt_sp'])
                    })
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {obj}: {str(e)}")
                continue
        except Exception as e:
            logger.error(f"Error processing object {obj}: {str(e)}")
            continue
    
    # Add mean row
    table_ls.append([
        'mean',
        str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
        str(np.round(np.mean(image_ap_list) * 100, decimals=1))
    ])
    
    # Create and log the table
    results_table = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    logger.info("\n%s", results_table)
    
    # Save results to file
    with open(os.path.join(args.save_path, 'cbm_finetune_results.txt'), 'w') as f:
        f.write(results_table)
    
    # Generate MD file with sample visualizations for high-performance objects
    if sample_data and args.visualization:
        logger.info(f"Generating markdown report for {len(sample_data)} high-performance objects")
        generate_md_report(sample_data, metrics, args, logger)
    
    # Analyze concept importance
    analyze_concept_importance(cbm, logger)
    
    logger.info("Testing completed successfully!")


def generate_md_report(sample_data, metrics, args, logger):
    """Generate markdown report with sample visualizations for high-performance objects (Pixel AUROC >= 80%)"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(args.save_path, f'cbm_high_performance_analysis_{timestamp}.md')
    vis_dir = os.path.join(args.save_path, 'visualizations')
    overlay_dir = os.path.join(args.save_path, 'overlays')
    
    # Filter for high-performance objects only
    high_perf_objects = [obj for obj in sample_data if obj['pixel_auroc'] >= 0.8]
    
    with open(md_path, 'w') as f:
        f.write("# AnomalyCLIP-CBM High Performance Analysis\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Method: Max concept selection with visualization enabled\n\n")
        f.write(f"**Note**: This report includes only objects with Pixel AUROC ≥ 80%\n\n")
        
        # High performance summary
        f.write("## High Performance Objects Summary\n\n")
        f.write(f"**Total Objects with Pixel AUROC ≥ 80%**: {len(high_perf_objects)}\n\n")
        
        if high_perf_objects:
            f.write("| Object | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Samples | Anomalies | Normals |\n")
            f.write("|--------|-------------|-------------|-------------|----------|---------|-----------|----------|\n")
            
            for obj_data in high_perf_objects:
                f.write(f"| {obj_data['object']} | {obj_data['pixel_auroc']*100:.1f}% | {obj_data['pixel_aupro']*100:.1f}% | {obj_data['image_auroc']*100:.1f}% | {obj_data['image_ap']*100:.1f}% | {obj_data['num_samples']} | {obj_data['num_anomalies']} | {obj_data['num_normals']} |\n")
            
            f.write("\n")
            
            # Overall statistics for high-performance objects
            f.write("### Statistics for High-Performance Objects\n\n")
            avg_pixel_auroc = np.mean([obj['pixel_auroc'] for obj in high_perf_objects])
            avg_pixel_aupro = np.mean([obj['pixel_aupro'] for obj in high_perf_objects])
            avg_image_auroc = np.mean([obj['image_auroc'] for obj in high_perf_objects])
            avg_image_ap = np.mean([obj['image_ap'] for obj in high_perf_objects])
            
            f.write("| Metric | Average Value |\n")
            f.write("|--------|---------------|\n")
            f.write(f"| Average Pixel AUROC | {avg_pixel_auroc*100:.1f}% |\n")
            f.write(f"| Average Pixel AUPRO | {avg_pixel_aupro*100:.1f}% |\n")
            f.write(f"| Average Image AUROC | {avg_image_auroc*100:.1f}% |\n")
            f.write(f"| Average Image AP | {avg_image_ap*100:.1f}% |\n\n")
            
            # Detailed analysis for each high-performance object
            f.write("## Detailed Analysis\n\n")
            
            for i, obj_data in enumerate(high_perf_objects):
                obj_name = obj_data['object']
                f.write(f"### {i+1}. Object: {obj_name}\n\n")
                
                # Performance metrics
                f.write("#### Performance Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Pixel AUROC | {obj_data['pixel_auroc']*100:.1f}% |\n")
                f.write(f"| Pixel AUPRO | {obj_data['pixel_aupro']*100:.1f}% |\n")
                f.write(f"| Image AUROC | {obj_data['image_auroc']*100:.1f}% |\n")
                f.write(f"| Image AP | {obj_data['image_ap']*100:.1f}% |\n")
                f.write(f"| Total Samples | {obj_data['num_samples']} |\n")
                f.write(f"| Anomaly Samples | {obj_data['num_anomalies']} |\n")
                f.write(f"| Normal Samples | {obj_data['num_normals']} |\n\n")
                
                # Add sample visualizations if they exist
                obj_vis_dir = os.path.join(vis_dir, obj_name)
                obj_overlay_dir = os.path.join(overlay_dir, obj_name)
                
                if os.path.exists(obj_vis_dir):
                    f.write("#### Sample Visualizations\n\n")
                    
                    # Find visualization files
                    vis_files = glob.glob(os.path.join(obj_vis_dir, "sample_*_visualization.png"))
                    vis_files.sort()
                    
                    if vis_files:
                        f.write("**3-Panel Visualizations (Original | Heatmap | Overlay)**:\n\n")
                        for j, vis_file in enumerate(vis_files[:5]):  # Show up to 5 samples
                            filename = os.path.basename(vis_file)
                            sample_num = filename.split('_')[1]
                            rel_path = os.path.relpath(vis_file, args.save_path)
                            f.write(f"![Sample {sample_num} Visualization](./{rel_path})\n\n")
                
                if os.path.exists(obj_overlay_dir):
                    # Find overlay files
                    overlay_files = glob.glob(os.path.join(obj_overlay_dir, f"{obj_name}_sample_*_overlay.png"))
                    overlay_files.sort()
                    
                    if overlay_files:
                        f.write("**Clean Overlay Images**:\n\n")
                        f.write("| Sample 1 | Sample 2 | Sample 3 |\n")
                        f.write("|:--------:|:--------:|:--------:|\n")
                        
                        # Show first 3 overlay images in a table
                        overlay_row = []
                        for j, overlay_file in enumerate(overlay_files[:3]):
                            rel_path = os.path.relpath(overlay_file, args.save_path)
                            overlay_row.append(f"![Overlay {j+1}](./{rel_path})")
                        
                        # Fill remaining cells if less than 3 images
                        while len(overlay_row) < 3:
                            overlay_row.append("-")
                        
                        f.write(f"| {' | '.join(overlay_row)} |\n\n")
                        
                        # Show remaining overlay images if any
                        if len(overlay_files) > 3:
                            f.write("**Additional Overlay Images**:\n\n")
                            for j, overlay_file in enumerate(overlay_files[3:6]):  # Show up to 3 more
                                rel_path = os.path.relpath(overlay_file, args.save_path)
                                sample_num = os.path.basename(overlay_file).split('_')[2]
                                f.write(f"![Sample {sample_num} Overlay](./{rel_path})\n\n")
                
                f.write("---\n\n")
            
        else:
            f.write("**No objects achieved Pixel AUROC ≥ 80%**\n\n")
        
        # Method description
        f.write("## Method Description\n\n")
        f.write("This analysis focuses on high-performing objects using the **max concept selection** approach:\n\n")
        f.write("1. **Concept Selection**: For each image, select the highest scoring concept from normal category and highest from anomaly category\n")
        f.write("2. **Classification**: Use CBM association matrix with all concepts\n")
        f.write("3. **Segmentation**: Use only the 2 selected concepts (max normal + max anomaly) to compute similarity maps\n")
        f.write("4. **Visualization**: Generated for objects with Pixel AUROC ≥ 80% to analyze successful cases\n\n")
        
        f.write("### Visualization Types\n\n")
        f.write("- **3-Panel Visualizations**: Show original image, heatmap only, and overlay side by side\n")
        f.write("- **Clean Overlay Images**: Single overlay images for easy comparison and analysis\n")
        f.write("- **Color Mapping**: Hot colormap for overlays, Jet colormap for standalone heatmaps\n\n")
        
        # Overall dataset performance context
        f.write("## Dataset Performance Context\n\n")
        all_objects = list(metrics.keys())
        total_objects = len(all_objects)
        high_perf_count = len(high_perf_objects)
        
        f.write(f"- **Total Objects in Dataset**: {total_objects}\n")
        f.write(f"- **High-Performance Objects (≥80% Pixel AUROC)**: {high_perf_count}\n")
        f.write(f"- **Success Rate**: {(high_perf_count/total_objects)*100:.1f}%\n\n")
        
        if total_objects > 0:
            all_pixel_auroc = [metrics[obj]['pixel-auroc'] for obj in all_objects]
            all_pixel_aupro = [metrics[obj]['pixel-aupro'] for obj in all_objects]
            all_image_auroc = [metrics[obj]['image-auroc'] for obj in all_objects]
            all_image_ap = [metrics[obj]['image-ap'] for obj in all_objects]
            
            f.write("### Overall Dataset Performance\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Mean Pixel AUROC | {np.mean(all_pixel_auroc)*100:.1f}% |\n")
            f.write(f"| Mean Pixel AUPRO | {np.mean(all_pixel_aupro)*100:.1f}% |\n")
            f.write(f"| Mean Image AUROC | {np.mean(all_image_auroc)*100:.1f}% |\n")
            f.write(f"| Mean Image AP | {np.mean(all_image_ap)*100:.1f}% |\n\n")
    
    logger.info(f"High-performance analysis report generated: {md_path}")
    if high_perf_objects:
        logger.info(f"Report includes {len(high_perf_objects)} high-performance objects")
    else:
        logger.info("No objects achieved Pixel AUROC ≥ 80%")
        
        # Sample analysis
        f.write("## Sample Analysis\n\n")
        f.write("Analysis of 5 anomaly samples showing selected concepts and segmentation results.\n\n")
        
        for i, sample in enumerate(sample_data):
            f.write(f"### Sample {i+1}: {sample['object']}\n\n")
            
            # Performance metrics for this sample
            obj_metrics = metrics[sample['object']]
            f.write("#### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Pixel AUROC | {obj_metrics['pixel-auroc']*100:.1f}% |\n")
            f.write(f"| Pixel AUPRO | {obj_metrics['pixel-aupro']*100:.1f}% |\n")
            f.write(f"| Image AUROC | {obj_metrics['image-auroc']*100:.1f}% |\n")
            f.write(f"| Image AP | {obj_metrics['image-ap']*100:.1f}% |\n\n")
            
            # Sample prediction info
            f.write("#### Prediction Results\n\n")
            f.write(f"- **True Label**: {'Anomaly' if sample['label'] == 1 else 'Normal'}\n")
            f.write(f"- **Prediction**: {'Anomaly' if sample['prediction'] == 1 else 'Normal'}\n")
            f.write(f"- **Confidence Score**: {sample['image_score']:.4f}\n\n")
            
            # Selected concepts (key difference from finetune_2 approach)
            f.write("#### Selected Concepts (Max Selection)\n\n")
            f.write(f"- **Selected Normal Concept**: {sample['selected_normal_concept']}\n")
            f.write(f"- **Selected Anomaly Concept**: {sample['selected_anomaly_concept']}\n\n")
            
            # Save and display images
            save_sample_visualizations(sample, i, vis_dir)
            
            # Add visualization section
            f.write("#### Segmentation Visualization\n\n")
            f.write("| Original Image | Ground Truth | Anomaly Map |\n")
            f.write("|:--------------:|:------------:|:-----------:|\n")
            f.write(f"| ![Original](./visualizations/sample_{i}_original.png) | ![GT](./visualizations/sample_{i}_gt.png) | ![Anomaly](./visualizations/sample_{i}_anomaly.png) |\n\n")
            
            # Top concepts
            f.write("#### Top 10 Activated Concepts\n\n")
            f.write("| Rank | Concept | Activation Score |\n")
            f.write("|:----:|:--------|:----------------:|\n")
            
            for rank, (concept, score) in enumerate(sample['top_concepts']):
                f.write(f"| {rank+1} | {concept} | {score:.4f} |\n")
            
            f.write("\n---\n\n")
        
        f.write("## Method Description\n\n")
        f.write("This test uses the **max concept selection** approach from `train_cbm_finetune.py`:\n\n")
        f.write("1. **Concept Selection**: For each image, select the highest scoring concept from normal category and highest from anomaly category\n")
        f.write("2. **Classification**: Use CBM association matrix with all concepts\n")
        f.write("3. **Segmentation**: Use only the 2 selected concepts (max normal + max anomaly) to compute similarity maps\n")
        f.write("4. **Difference from finetune_2**: Uses max selection instead of top-k selection\n\n")
        
        f.write("### Comparison with Other Approaches\n\n")
        f.write("- **train_cbm.py**: Uses all concept type embeddings (5 channels: normal + 4 anomaly types)\n")
        f.write("- **train_cbm_finetune.py**: Uses max concept selection (2 concepts per image)\n")
        f.write("- **train_cbm_finetune_2.py**: Uses top-k concept selection (6 concepts per image: top-3 normal + top-3 anomaly)\n\n")
    
    logger.info(f"Markdown report generated: {md_path}")


def save_sample_visualizations(sample, idx, vis_dir):
    """Save visualization images for a sample"""
    
    # Original image
    original = sample['image'].numpy()
    if original.shape[0] == 3:  # CHW format
        original = np.transpose(original, (1, 2, 0))
    
    # Normalize to [0, 1] if needed
    if original.max() > 1.0:
        original = original / 255.0
    
    plt.figure(figsize=(6, 6))
    plt.imshow(original)
    plt.axis('off')
    plt.title('Original Image')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'sample_{idx}_original.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Ground truth mask
    gt_mask = sample['gt_mask'].numpy()
    if len(gt_mask.shape) == 3:
        gt_mask = gt_mask.squeeze()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')
    plt.title('Ground Truth Mask')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'sample_{idx}_gt.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Anomaly map
    anomaly_map = sample['anomaly_map'].numpy()
    if len(anomaly_map.shape) == 3:
        anomaly_map = anomaly_map.squeeze()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(anomaly_map, cmap='jet')
    plt.axis('off')
    plt.title('Anomaly Map')
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'sample_{idx}_anomaly.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_visualization_heatmaps(obj, results, args, logger):
    """Save visualization heatmaps with original image overlay for objects with high pixel AUROC"""
    
    # Create visualization directory for this object
    vis_dir = os.path.join(args.save_path, 'visualizations', obj)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get results for this object
    anomaly_maps = results[obj]['anomaly_maps']  # List of tensors
    gt_sp = results[obj]['gt_sp']  # Ground truth labels
    img_paths = results[obj]['img_paths']  # Image paths
    
    # Convert to numpy if needed
    if isinstance(anomaly_maps, torch.Tensor):
        anomaly_maps = anomaly_maps.detach().cpu().numpy()
    elif isinstance(anomaly_maps, list) and len(anomaly_maps) > 0:
        if torch.is_tensor(anomaly_maps[0]):
            anomaly_maps = torch.cat(anomaly_maps).detach().cpu().numpy()
        else:
            anomaly_maps = np.array(anomaly_maps)
    
    # Save sample heatmaps (save up to 10 samples)
    num_samples = min(10, len(anomaly_maps))
    saved_count = 0
    
    for i in range(num_samples):
        try:
            anomaly_map = anomaly_maps[i]
            
            # Handle different tensor dimensions
            if len(anomaly_map.shape) == 3:
                anomaly_map = anomaly_map.squeeze()
            elif len(anomaly_map.shape) == 1:
                # If it's flattened, try to reshape to square
                side_length = int(np.sqrt(len(anomaly_map)))
                if side_length * side_length == len(anomaly_map):
                    anomaly_map = anomaly_map.reshape(side_length, side_length)
                else:
                    logger.warning(f"Cannot reshape anomaly map for {obj} sample {i}")
                    continue
            
            # Normalize to [0, 1]
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            
            # Create visualization with 3 panels: Original, Heatmap Only, Overlay
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Load original image if path is available
            original_img_array = None
            if i < len(img_paths) and img_paths[i]:
                try:
                    from PIL import Image
                    original_img = Image.open(img_paths[i])
                    if original_img.mode != 'RGB':
                        original_img = original_img.convert('RGB')
                    original_img = original_img.resize((args.image_size, args.image_size))
                    original_img_array = np.array(original_img)
                except Exception as e:
                    logger.warning(f"Could not load image {img_paths[i]}: {str(e)}")
                    original_img_array = None
            
            # Panel 1: Original Image
            if original_img_array is not None:
                axes[0].imshow(original_img_array)
                axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
                axes[0].axis('off')
            else:
                axes[0].text(0.5, 0.5, 'Original Image\nNot Available', 
                           ha='center', va='center', transform=axes[0].transAxes,
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
                axes[0].axis('off')
            
            # Panel 2: Heatmap Only
            im1 = axes[1].imshow(anomaly_map, cmap='jet', interpolation='bilinear')
            axes[1].set_title('Anomaly Heatmap', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], shrink=0.8)
            
            # Panel 3: Overlay
            if original_img_array is not None:
                # Resize anomaly map to match original image size if needed
                if anomaly_map.shape != original_img_array.shape[:2]:
                    from skimage.transform import resize
                    anomaly_map_resized = resize(anomaly_map, original_img_array.shape[:2], 
                                               mode='reflect', anti_aliasing=True)
                else:
                    anomaly_map_resized = anomaly_map
                
                # Show original image with heatmap overlay
                axes[2].imshow(original_img_array, alpha=0.8)  # Slightly more opaque original
                im2 = axes[2].imshow(anomaly_map_resized, cmap='hot', alpha=0.6, interpolation='bilinear')
                axes[2].set_title('Overlay (Original + Heatmap)', fontsize=14, fontweight='bold')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], shrink=0.8)
            else:
                # Fallback to heatmap only
                im2 = axes[2].imshow(anomaly_map, cmap='hot', interpolation='bilinear')
                axes[2].set_title('Anomaly Heatmap (No Original)', fontsize=14, fontweight='bold')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], shrink=0.8)
            
            # Add main title with ground truth info
            gt_label = 'Anomaly' if i < len(gt_sp) and gt_sp[i] == 1 else 'Normal'
            fig.suptitle(f'{obj} - Sample {i+1} (GT: {gt_label})', fontsize=16, fontweight='bold')
            
            plt.tight_layout(pad=2.0)
            
            # Save
            save_path = os.path.join(vis_dir, f'sample_{i+1:03d}_visualization.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            saved_count += 1
            
        except Exception as e:
            logger.warning(f"Error saving visualization for {obj} sample {i}: {str(e)}")
            continue
    
    logger.info(f"Saved {saved_count} visualizations for {obj} in {vis_dir}")


def save_overlay_images(anomaly_maps, gt_sp, img_paths, obj, pixel_auroc, args, logger):
    """Save clean overlay images (original + heatmap) as single files"""
    
    # Create directory for overlay images
    overlay_dir = os.path.join(args.save_path, 'overlays', obj)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Convert anomaly maps to numpy if needed
    if hasattr(anomaly_maps, 'detach'):
        if len(anomaly_maps) > 0:
            anomaly_maps = torch.cat(anomaly_maps).detach().cpu().numpy()
        else:
            anomaly_maps = np.array(anomaly_maps)
    
    # Save overlay images (save up to 15 samples)
    num_samples = min(15, len(anomaly_maps))
    saved_count = 0
    
    for i in range(num_samples):
        try:
            anomaly_map = anomaly_maps[i]
            
            # Handle different tensor dimensions
            if len(anomaly_map.shape) == 3:
                anomaly_map = anomaly_map.squeeze()
            elif len(anomaly_map.shape) == 1:
                # If it's flattened, try to reshape to square
                side_length = int(np.sqrt(len(anomaly_map)))
                if side_length * side_length == len(anomaly_map):
                    anomaly_map = anomaly_map.reshape(side_length, side_length)
                else:
                    logger.warning(f"Cannot reshape anomaly map for {obj} sample {i}")
                    continue
            
            # Normalize to [0, 1]
            if anomaly_map.max() > anomaly_map.min():
                anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
            
            # Load original image if path is available
            original_img_array = None
            if i < len(img_paths) and img_paths[i]:
                try:
                    from PIL import Image
                    original_img = Image.open(img_paths[i])
                    if original_img.mode != 'RGB':
                        original_img = original_img.convert('RGB')
                    original_img = original_img.resize((args.image_size, args.image_size))
                    original_img_array = np.array(original_img)
                except Exception as e:
                    logger.warning(f"Could not load image {img_paths[i]}: {str(e)}")
                    continue
            else:
                continue  # Skip if no original image
            
            # Resize anomaly map to match original image size if needed
            if anomaly_map.shape != original_img_array.shape[:2]:
                from skimage.transform import resize
                anomaly_map_resized = resize(anomaly_map, original_img_array.shape[:2], 
                                           mode='reflect', anti_aliasing=True)
            else:
                anomaly_map_resized = anomaly_map
            
            # Create clean overlay image
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Show original image with heatmap overlay
            ax.imshow(original_img_array)
            im = ax.imshow(anomaly_map_resized, cmap='hot', alpha=0.5, interpolation='bilinear')
            
            # Add title with ground truth info
            gt_label = 'Anomaly' if i < len(gt_sp) and gt_sp[i] == 1 else 'Normal'
            ax.set_title(f'{obj} - Sample {i+1} (GT: {gt_label}, Pixel AUROC: {pixel_auroc:.1f}%)', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label('Anomaly Score', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save clean overlay image
            overlay_path = os.path.join(overlay_dir, f'{obj}_sample_{i+1:03d}_overlay.png')
            plt.savefig(overlay_path, dpi=200, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', pad_inches=0.1)
            plt.close()
            
            saved_count += 1
            
        except Exception as e:
            logger.warning(f"Error saving overlay for {obj} sample {i}: {str(e)}")
            continue
    
    logger.info(f"Saved {saved_count} overlay images for {obj} in {overlay_dir}")


def analyze_concept_importance(cbm, logger):
    """Analyze which concepts are most important for anomaly detection"""
    
    with torch.no_grad():
        # Get ReLU-activated association matrix weights
        activated_weights = F.relu(cbm.association_matrix)
        
        # Difference between anomaly and normal weights
        concept_importance = activated_weights[1] - activated_weights[0]
        
        # Sort concepts by importance
        sorted_indices = torch.argsort(concept_importance, descending=True)
        
        logger.info("\n" + "="*50)
        logger.info("CONCEPT IMPORTANCE ANALYSIS (Max Selection Method)")
        logger.info("="*50)
        
        logger.info("\nTop 15 concepts for anomaly detection:")
        for i in range(min(15, len(sorted_indices))):
            idx = sorted_indices[i]
            concept_name = cbm.all_concepts[idx]
            importance = concept_importance[idx].item()
            normal_weight = activated_weights[0, idx].item()
            anomaly_weight = activated_weights[1, idx].item()
            category = cbm.concept_to_category.get(concept_name, "unknown")
            logger.info(f"{i+1:2d}. {concept_name:<30} | Importance: {importance:7.4f} | Normal: {normal_weight:6.4f} | Anomaly: {anomaly_weight:6.4f} | Category: {category}")
        
        logger.info("\nTop 10 concepts for normal detection:")
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[-(i+1)]
            concept_name = cbm.all_concepts[idx]
            importance = concept_importance[idx].item()
            normal_weight = activated_weights[0, idx].item()
            anomaly_weight = activated_weights[1, idx].item()
            category = cbm.concept_to_category.get(concept_name, "unknown")
            logger.info(f"{i+1:2d}. {concept_name:<30} | Importance: {importance:7.4f} | Normal: {normal_weight:6.4f} | Anomaly: {anomaly_weight:6.4f} | Category: {category}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP-CBM Test (Finetune)", add_help=True)
    
    # Paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results_cbm_finetune', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to trained CBM checkpoint")
    parser.add_argument("--concepts_file", type=str, default="./concepts_raw.json", help="path to concepts JSON file")
    
    # Model parameters
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset name")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian filter sigma")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--visualization", action="store_true", help="save visualization heatmaps for objects with pixel AUROC >= 80")
    
    args = parser.parse_args()
    print(args)
    
    setup_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    test(args)
