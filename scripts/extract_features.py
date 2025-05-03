import argparse
import os
from pathlib import Path
import logging
import traceback
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.feature_extraction.efficientnet import EfficientNetFeatureExtractor
from src.feature_extraction.color_features import ColorFeatureExtractor
from src.feature_extraction.fcm_clustering import FCMClustering
from src.feature_extraction.dimensionality import PCAReducer
from src.config import config
from src.utils.data_handling import load_patches, save_features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = info, 2 = warning, 3 = error

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feature_extraction.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def extract_features(patches: List[np.ndarray], 
                    image_name: str,
                    output_dir: str,
                    deep_extractor: EfficientNetFeatureExtractor,
                    color_extractor: ColorFeatureExtractor,
                    fcm: FCMClustering,
                    pca: PCAReducer) -> None:
    """
    Extract features from patches
    
    Args:
        patches: List of image patches
        image_name: Name of source image
        output_dir: Directory to save features
        deep_extractor: Deep feature extractor
        color_extractor: Color feature extractor
        fcm: FCM clustering instance
        pca: PCA reducer instance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Convert patches list to NumPy array for batch processing
        patches_np = np.stack(patches)  # (N, H, W, C)

        # Extract deep features
        logger.info("Extracting deep features")
        deep_features = deep_extractor.extract_features(patches_np)
        logger.info(f"Extracted deep features shape: {deep_features.shape}")
        # Save deep features
        save_features(deep_features, image_name, os.path.join(output_dir, 'deep'))
        
        # Extract color features for each patch and stack
        logger.info("Extracting color features")
        color_features_list = []
        for patch in tqdm(patches_np, desc="Extracting color features"):
            features = color_extractor.extract_features(patch)
            if features:  # Only add non-empty feature dictionaries
                color_features_list.append(features)
        
        if not color_features_list:
            logger.error("No valid color features extracted")
            raise ValueError("No valid color features extracted")
            
        # Get all possible keys from all feature dictionaries
        all_keys = set()
        for features in color_features_list:
            all_keys.update(features.keys())
        color_features_keys = sorted(list(all_keys))
        
        # Create feature matrix with consistent columns
        color_features = np.zeros((len(patches_np), len(color_features_keys)))
        for i, features in enumerate(color_features_list):
            for j, key in enumerate(color_features_keys):
                if key in features:
                    color_features[i, j] = features[key]
                    
        logger.info(f"Extracted color features shape: {color_features.shape}")
        # Save color features
        save_features(color_features, image_name, os.path.join(output_dir, 'color'))
        
        # Combine features
        logger.info("Combining features")
        combined_features = np.concatenate([deep_features, color_features], axis=1)
        logger.info(f"Combined features shape: {combined_features.shape}")
        # Save combined features
        save_features(combined_features, image_name, os.path.join(output_dir, 'combined'))
        
        # Apply FCM clustering
        logger.info("Applying FCM clustering")
        cluster_features = fcm.extract_features(combined_features)
        logger.info(f"Cluster features shape: {cluster_features.shape}")
        
        # Apply PCA
        logger.info("Applying PCA")
        reduced_features = pca.reduce_dimensions(cluster_features)
        logger.info(f"Reduced features shape: {reduced_features.shape}")
        
        # Save features
        logger.info(f"Saving features for {image_name}")
        save_features(reduced_features, image_name, output_dir)
        logger.info(f"Features saved to {output_dir}")
        
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
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting feature extraction pipeline")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'deep'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'combined'), exist_ok=True)
    logger.info(f"Created output directories in {args.output_dir}")
    
    # Initialize feature extractors
    logger.info("Initializing feature extractors")
    deep_extractor = EfficientNetFeatureExtractor(
        input_shape=config.feature_extraction.input_shape,
        batch_size=config.feature_extraction.batch_size
    )
    
    color_extractor = ColorFeatureExtractor(
        color_spaces=config.feature_extraction.color_spaces
    )
    
    fcm = FCMClustering(
        n_clusters=config.feature_extraction.fcm_clusters
    )
    
    pca = PCAReducer(
        variance_threshold=config.feature_extraction.pca_variance
    )
    
    # --- NEW LOGIC: Group .npy files by image prefix ---
    patch_files = list(Path(args.input_dir).glob('*.npy'))
    image_groups = {}
    for patch_file in patch_files:
        fname = patch_file.name
        if '_patch_' in fname:
            prefix = fname.split('_patch_')[0]
            image_groups.setdefault(prefix, []).append(patch_file)
        else:
            logger.warning(f"File {fname} does not match expected pattern and will be skipped.")
    logger.info(f"Found {len(image_groups)} images to process (grouped by prefix)")
    
    for i, (image_name, patch_paths) in enumerate(image_groups.items(), 1):
        logger.info(f"Processing image {i}/{len(image_groups)}: {image_name} with {len(patch_paths)} patches")
        
        # Load patches
        patches = []
        for patch_path in patch_paths:
            try:
                patch = np.load(patch_path)
                patches.append(patch)
            except Exception as e:
                logger.error(f"Failed to load patch {patch_path}: {e}")
        logger.info(f"Loaded {len(patches)} patches for {image_name}")
        if not patches:
            logger.warning(f"No patches loaded for {image_name}, skipping.")
            continue
        
        # Extract features
        extract_features(
            patches=patches,
            image_name=image_name,
            output_dir=args.output_dir,
            deep_extractor=deep_extractor,
            color_extractor=color_extractor,
            fcm=fcm,
            pca=pca
        )
    
    logger.info("Feature extraction completed successfully")

if __name__ == '__main__':
    main() 