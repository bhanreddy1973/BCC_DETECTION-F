import argparse
import os
from pathlib import Path
import logging
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tqdm import tqdm
import cv2

from src.preprocessing.tissue_segmentation import TissueSegmenter
from src.preprocessing.patch_extraction import PatchExtractor
from src.feature_extraction.efficientnet import EfficientNetFeatureExtractor
from src.feature_extraction.color_features import ColorFeatureExtractor
from src.feature_extraction.fcm_clustering import FCMClustering
from src.feature_extraction.dimensionality import PCAReducer
from src.config import config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimized_pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess image"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def optimize_tissue_segmentation(image: np.ndarray, 
                               true_mask: np.ndarray,
                               segmenter: TissueSegmenter) -> Dict:
    """Optimize tissue segmentation parameters"""
    param_grid = {
        'min_tissue_threshold': [0.6, 0.7, 0.8],
        'patch_size': [224, 256, 288],
        'patch_overlap': [0.3, 0.5, 0.7]
    }
    
    best_params = None
    best_score = -1
    
    for min_thresh in param_grid['min_tissue_threshold']:
        for size in param_grid['patch_size']:
            for overlap in param_grid['patch_overlap']:
                segmenter.min_tissue_threshold = min_thresh
                segmenter.patch_size = size
                segmenter.patch_overlap = overlap
                
                # Segment tissue
                pred_mask = segmenter.segment_tissue(image)
                
                # Calculate IoU score
                intersection = np.logical_and(pred_mask, true_mask)
                union = np.logical_or(pred_mask, true_mask)
                iou_score = np.sum(intersection) / np.sum(union)
                
                if iou_score > best_score:
                    best_score = iou_score
                    best_params = {
                        'min_tissue_threshold': min_thresh,
                        'patch_size': size,
                        'patch_overlap': overlap
                    }
    
    return best_params

def optimize_feature_extraction(patches: List[np.ndarray],
                              labels: List[int],
                              deep_extractor: EfficientNetFeatureExtractor,
                              color_extractor: ColorFeatureExtractor,
                              fcm: FCMClustering,
                              pca: PCAReducer) -> Dict:
    """Optimize feature extraction parameters"""
    param_grid = {
        'pca_variance': [0.95, 0.99, 0.999],
        'batch_size': [16, 32, 64],
        'fcm_clusters': [2, 3, 4]
    }
    
    best_params = None
    best_score = -1
    
    for pca_var in param_grid['pca_variance']:
        for batch in param_grid['batch_size']:
            for clusters in param_grid['fcm_clusters']:
                # Update parameters
                deep_extractor.pca_variance = pca_var
                deep_extractor.batch_size = batch
                fcm.n_clusters = clusters
                
                # Extract features
                try:
                    deep_features = deep_extractor.extract_features(np.array(patches))
                    color_features = [color_extractor.extract_features(patch) for patch in patches]
                    
                    # Ensure color_features is a 2D array
                    color_features = np.array(color_features)
                    if color_features.ndim == 1:
                        color_features = color_features[:, np.newaxis]
                    assert deep_features.shape[0] == color_features.shape[0], "Mismatch in number of patches"
                    
                    # Combine features
                    combined_features = np.concatenate([deep_features, color_features], axis=1)
                    
                    # Apply FCM and PCA
                    cluster_features = fcm.extract_features(combined_features)
                    reduced_features = pca.fit_transform(cluster_features)
                    
                    # Evaluate with simple classifier
                    from sklearn.linear_model import LogisticRegression
                    clf = LogisticRegression(max_iter=1000)
                    scores = cross_val_score(clf, reduced_features, labels, cv=5)
                    mean_score = np.mean(scores)
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'pca_variance': pca_var,
                            'batch_size': batch,
                            'fcm_clusters': clusters
                        }
                except Exception as e:
                    logging.error(f"Error in feature extraction: {str(e)}")
                    continue
    
    return best_params

def main():
    parser = argparse.ArgumentParser(description='Optimized BCC detection pipeline')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save outputs')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting optimized pipeline")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'segmentation'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'patches'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'features'), exist_ok=True)
    
    # Initialize components
    segmenter = TissueSegmenter()
    patch_extractor = PatchExtractor(
        patch_size=config.preprocessing.patch_size,
        patch_overlap=config.preprocessing.patch_overlap
    )
    deep_extractor = EfficientNetFeatureExtractor()
    color_extractor = ColorFeatureExtractor(
        color_spaces=config.feature_extraction.color_spaces
    )
    fcm = FCMClustering(
        n_clusters=config.feature_extraction.fcm_clusters
    )
    pca = PCAReducer(
        variance_threshold=config.feature_extraction.pca_variance
    )
    
    # Load and process images
    image_files = list(Path(args.input_dir).glob('*.tif'))
    logger.info(f"Found {len(image_files)} images to process")
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Check if patches for this image already exist
            patch_dir = os.path.join(args.output_dir, 'patches')
            existing_patches = list(Path(patch_dir).glob(f"{image_path.stem}_patch_*.npy"))
            if len(existing_patches) > 0:
                logger.info(f"Skipping {image_path.name} as patches already exist ({len(existing_patches)} found)")
                continue

            # Load image
            image = load_image(str(image_path))
            
            # Optimize tissue segmentation
            # Note: This requires ground truth masks for optimization
            # For now, we'll use default parameters
            mask = segmenter.segment_tissue(image)
            
            # Extract patches
            patches, coordinates = patch_extractor.extract_patches(image, mask)
            logger.info(f"Extracted {len(patches)} patches from {image_path.name}")
            
            # Save patches
            for i, (patch, coord) in enumerate(zip(patches, coordinates)):
                patch_path = os.path.join(args.output_dir, 'patches', 
                                        f"{image_path.stem}_patch_{i}_y{coord[0]}_x{coord[1]}.npy")
                np.save(patch_path, patch)
            
            # Extract features
            deep_features = deep_extractor.extract_features(np.array(patches))
            color_features = [color_extractor.extract_features(patch) for patch in patches]
            
            # Ensure color_features is a 2D array
            color_features = np.array(color_features)
            if color_features.ndim == 1:
                color_features = color_features[:, np.newaxis]
            assert deep_features.shape[0] == color_features.shape[0], "Mismatch in number of patches"
            
            # Combine features
            combined_features = np.concatenate([deep_features, color_features], axis=1)
            
            # Apply FCM and PCA
            cluster_features = fcm.extract_features(combined_features)
            reduced_features = pca.fit_transform(cluster_features)
            
            # Save features
            features_path = os.path.join(args.output_dir, 'features', 
                                       f"{image_path.stem}_features.npy")
            np.save(features_path, reduced_features)
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            continue
    
    logger.info("Pipeline completed successfully")

if __name__ == '__main__':
    main()