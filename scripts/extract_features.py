import argparse
import os
from pathlib import Path
import logging
import traceback
from typing import Dict, List
import psutil
import gc

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from src.feature_extraction.efficientnet import EfficientNetFeatureExtractor
from src.feature_extraction.color_features import ColorFeatureExtractor
from src.feature_extraction.fcm_clustering import FCMClustering
from src.feature_extraction.dimensionality import PCAReducer
from src.config import config
from src.utils.data_handling import load_patches, save_features
from src.utils.visualization import PipelineVisualizer

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'feature_extraction.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_optimal_batch_size(patch_shape, available_memory, safety_factor=0.7):
    """Calculate optimal batch size based on patch size and available memory"""
    patch_memory = np.prod(patch_shape) * 4  # 4 bytes per float32
    safe_memory = int(available_memory * safety_factor)
    return max(16, min(128, safe_memory // patch_memory))

def extract_features(patches: List[np.ndarray], 
                    image_name: str,
                    output_dir: str,
                    deep_extractor: EfficientNetFeatureExtractor,
                    color_extractor: ColorFeatureExtractor,
                    fcm: FCMClustering,
                    pca: PCAReducer,
                    visualizer: PipelineVisualizer = None) -> None:
    """Extract features from patches with visualization and memory optimization"""
    logger = logging.getLogger(__name__)
    
    feature_paths = {
        'deep': os.path.join(output_dir, 'deep', f"{image_name}_features.npy"),
        'color': os.path.join(output_dir, 'color', f"{image_name}_features.npy"),
        'combined': os.path.join(output_dir, 'combined', f"{image_name}_features.npy"),
        'final': os.path.join(output_dir, f"{image_name}_features.npy")
    }
    
    # Check if all feature files exist
    if all(os.path.exists(path) for path in feature_paths.values()):
        logger.info(f"Features already exist for {image_name}, skipping...")
        return
    
    try:
        # Get optimal batch size based on available memory
        available_memory = psutil.virtual_memory().available
        batch_size = get_optimal_batch_size(patches[0].shape, available_memory)
        logger.info(f"Using batch size: {batch_size}")
        
        # Convert patches list to NumPy array for batch processing
        patches_np = np.stack(patches)  # (N, H, W, C)
        
        # Extract deep features in batches
        logger.info("Extracting deep features")
        deep_features_list = []
        for i in range(0, len(patches_np), batch_size):
            batch = patches_np[i:i + batch_size]
            batch_features = deep_extractor.extract_features(batch)
            deep_features_list.append(batch_features)
            
        deep_features = np.concatenate(deep_features_list, axis=0)
        logger.info(f"Extracted deep features shape: {deep_features.shape}")
        save_features(deep_features, image_name, os.path.join(output_dir, 'deep'))
        
        # Extract color features with parallel processing
        logger.info("Extracting color features")
        with ThreadPoolExecutor(max_workers=48) as executor:  # Using half available cores
            color_features_list = list(executor.map(color_extractor.extract_features, patches))
        
        # Convert color features to array
        color_features_keys = sorted(list(set().union(*[f.keys() for f in color_features_list])))
        color_features = np.zeros((len(patches_np), len(color_features_keys)))
        
        for i, features in enumerate(color_features_list):
            for j, key in enumerate(color_features_keys):
                if key in features:
                    color_features[i, j] = features[key]
        
        logger.info(f"Extracted color features shape: {color_features.shape}")
        save_features(color_features, image_name, os.path.join(output_dir, 'color'))
        
        # Combine features
        logger.info("Combining features")
        combined_features = np.concatenate([deep_features, color_features], axis=1)
        logger.info(f"Combined features shape: {combined_features.shape}")
        save_features(combined_features, image_name, os.path.join(output_dir, 'combined'))
        
        # Apply FCM clustering
        logger.info("Applying FCM clustering")
        cluster_features = fcm.extract_features(combined_features)
        logger.info(f"Cluster features shape: {cluster_features.shape}")
        
        # Apply PCA
        logger.info("Applying PCA")
        reduced_features = pca.fit_transform(cluster_features)
        logger.info(f"Reduced features shape: {reduced_features.shape}")
        
        # Save final features
        logger.info(f"Saving features for {image_name}")
        save_features(reduced_features, image_name, output_dir)
        
        # Visualize features if visualizer is available
        if visualizer:
            visualizer.visualize_features(
                deep_features=deep_features,
                color_features=color_features,
                combined_features=reduced_features,
                image_name=image_name
            )
        
        # Clean up to free memory
        del deep_features, color_features, combined_features, reduced_features
        gc.collect()
        
    except Exception as e:
        logger.error(f"Error extracting features for {image_name}:")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main function for feature extraction"""
    parser = argparse.ArgumentParser(description='Extract features from BCC dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing processed patches')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save extracted features')
    parser.add_argument('--visualization_dir', type=str, default='data/visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--num_workers', type=int, default=48,
                      help='Number of worker processes (default: 48)')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting feature extraction pipeline")
    
    # Create output directories with class-specific subdirs
    splits = ['train', 'val', 'test']
    classes = ['bcc_high_risk', 'bcc_low_risk', 'non_malignant']
    
    for split in splits:
        for class_name in classes:
            for feature_type in ['deep', 'color', 'combined']:
                os.makedirs(os.path.join(args.output_dir, split, class_name, feature_type), exist_ok=True)
    
    # Create visualization directory
    os.makedirs(args.visualization_dir, exist_ok=True)
    
    logger.info(f"Created output directories in {args.output_dir}")
    
    # Initialize visualizer
    visualizer = PipelineVisualizer(args.visualization_dir, n_jobs=args.num_workers)
    
    # Initialize feature extractors with mixed precision
    logger.info("Initializing feature extractors")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    deep_extractor = EfficientNetFeatureExtractor(
        input_shape=config.feature_extraction.input_shape,
        batch_size=config.feature_extraction.batch_size
    )
    
    color_extractor = ColorFeatureExtractor(
        color_spaces=config.feature_extraction.color_spaces
    )
    
    fcm = FCMClustering(
        n_clusters=config.feature_extraction.fcm_clusters,
        max_iter=config.feature_extraction.max_iter,
        error=config.feature_extraction.fcm_error,
        m=config.feature_extraction.fcm_fuzziness
    )
    
    pca = PCAReducer(
        variance_threshold=config.feature_extraction.pca_variance
    )
    
    # Find all patch files recursively
    patch_files = []
    input_path = Path(args.input_dir)
    
    for split in splits:
        split_dir = input_path / split
        if not split_dir.exists():
            continue
            
        for class_name in classes:
            class_dir = split_dir / class_name / 'patches'
            if not class_dir.exists():
                continue
                
            # Find all patch files
            patch_files.extend([
                f for f in class_dir.glob('*_patches_*.npy')
                if not f.name.startswith('.')  # Skip hidden files
            ])
    
    logger.info(f"Found {len(patch_files)} patch files")
    
    # Group patches by original image
    image_groups = {}
    for patch_file in patch_files:
        # Extract original image name from patch filename
        # Format: {image_id}_patches_{batch_number}.npy
        image_id = patch_file.name.split('_patches_')[0]
        split = patch_file.parent.parent.parent.name  # train/val/test
        class_name = patch_file.parent.parent.name    # bcc_high_risk etc
        group_key = f"{split}/{class_name}/{image_id}"
        image_groups.setdefault(group_key, []).append(patch_file)
    
    logger.info(f"Found {len(image_groups)} unique images to process")
    
    # Process each image's patches
    for i, (image_key, patch_paths) in enumerate(image_groups.items(), 1):
        logger.info(f"Processing image {i}/{len(image_groups)}: {image_key}")
        
        # Load all patches for this image
        patches = []
        for patch_path in patch_paths:
            try:
                patch_batch = np.load(str(patch_path))
                if isinstance(patch_batch, np.ndarray):
                    if len(patch_batch.shape) == 4:  # (N, H, W, C)
                        patches.extend([p for p in patch_batch])
                    elif len(patch_batch.shape) == 3:  # Single patch (H, W, C)
                        patches.append(patch_batch)
            except Exception as e:
                logger.error(f"Error loading patches from {patch_path}: {str(e)}")
                continue
        
        if not patches:
            logger.warning(f"No valid patches found for {image_key}")
            continue
            
        logger.info(f"Loaded {len(patches)} patches")
        
        try:
            # Get split and class from image_key
            split, class_name, image_id = image_key.split('/')
            
            # Extract features
            extract_features(
                patches=patches,
                image_name=image_id,
                output_dir=os.path.join(args.output_dir, split, class_name),
                deep_extractor=deep_extractor,
                color_extractor=color_extractor,
                fcm=fcm,
                pca=pca,
                visualizer=visualizer
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_key}: {str(e)}")
            continue
        
        # Clean up memory
        del patches
        gc.collect()
    
    logger.info("Feature extraction completed successfully")

if __name__ == '__main__':
    main()