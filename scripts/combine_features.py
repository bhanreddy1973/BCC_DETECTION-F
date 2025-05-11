import numpy as np
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Set
import argparse
import re

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_sample_ids(feature_dir: Path, split: str, class_name: str) -> Set[str]:
    """Get all unique sample IDs for a given split and class"""
    pattern = r'([a-f0-9]{32})_features\.npy$'
    sample_ids = set()
    
    # Look in both color and deep feature directories
    for feat_type in ['color', 'deep']:
        feat_path = feature_dir / split / class_name / feat_type
        if feat_path.exists():
            for feat_file in feat_path.glob('*.npy'):
                if match := re.search(pattern, feat_file.name):
                    sample_ids.add(match.group(1))
    
    return sample_ids

def get_target_dimensions() -> Dict[str, int]:
    """
    Define target dimensions for each feature type
    """
    return {
        'color': 48,  # Color features seem consistent
        'deep': 1280  # Standard EfficientNet feature dimension
    }

def pad_or_truncate_features(features: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pad or truncate feature vector to match target dimension
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)
        
    current_dim = features.shape[1]
    if current_dim == target_dim:
        return features
    elif current_dim < target_dim:
        # Pad with zeros
        padding = np.zeros((features.shape[0], target_dim - current_dim))
        return np.hstack([features, padding])
    else:
        # Truncate
        return features[:, :target_dim]

def load_features_for_sample(feature_dir: Path, split: str, class_name: str, 
                           sample_id: str, target_dims: Dict[str, int]) -> np.ndarray:
    """Load and combine color and deep features for a single sample"""
    features = []
    logger = logging.getLogger(__name__)
    
    for feat_type in ['color', 'deep']:
        if feat_type not in target_dims:
            continue
            
        feat_path = feature_dir / split / class_name / feat_type / f'{sample_id}_features.npy'
        if feat_path.exists():
            try:
                feat = np.load(str(feat_path))
                # Pad or truncate to target dimension
                feat = pad_or_truncate_features(feat, target_dims[feat_type])
                features.append(feat)
            except Exception as e:
                logger.warning(f"Error loading {feat_path}: {e}")
    
    if not features:
        return None
        
    # All features for a sample should have the same number of patches
    if not all(f.shape[0] == features[0].shape[0] for f in features):
        logger.error(f"Inconsistent number of patches for sample {sample_id}")
        return None
        
    # Concatenate features horizontally
    return np.hstack(features)

def combine_features(data_dir: Path, splits: List[str]) -> None:
    """Combine features from different sources for each split"""
    logger = logging.getLogger(__name__)
    features_dir = data_dir / 'features'
    
    # Create combined directory if it doesn't exist
    combined_dir = features_dir / 'combined'
    combined_dir.mkdir(exist_ok=True)
    
    # Get target dimensions for features
    target_dims = get_target_dimensions()
    logger.info(f"Target feature dimensions: {target_dims}")
    
    for split in splits:
        logger.info(f"Processing {split} split...")
        split_features = []
        split_labels = []
        
        for class_name in ['bcc_high_risk', 'bcc_low_risk', 'non_malignant']:
            # Get all unique sample IDs for this split/class
            sample_ids = get_sample_ids(features_dir, split, class_name)
            if not sample_ids:
                logger.warning(f"No samples found for {split}/{class_name}")
                continue
                
            logger.info(f"Found {len(sample_ids)} samples for {split}/{class_name}")
            
            # Process each sample
            for sample_id in sample_ids:
                features = load_features_for_sample(features_dir, split, class_name, 
                                                 sample_id, target_dims)
                if features is not None:
                    split_features.append(features)
                    split_labels.extend([1 if 'bcc' in class_name else 0] * len(features))
        
        if split_features:
            # Combine all features and labels for this split
            combined_features = np.vstack(split_features)
            combined_labels = np.array(split_labels)
            
            logger.info(f"Combined {len(combined_features)} samples for {split} "
                       f"with feature shape {combined_features.shape}")
            
            # Save combined features and labels
            np.save(str(combined_dir / f'{split}_features.npy'), combined_features)
            np.save(str(combined_dir / f'{split}_labels.npy'), combined_labels)
            logger.info(f"Saved combined features and labels for {split}")
        else:
            logger.warning(f"No valid features found for {split}")

def main():
    parser = argparse.ArgumentParser(description='Combine features from different sources')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Base data directory')
    args = parser.parse_args()
    
    logger = setup_logging()
    data_dir = Path(args.data_dir)
    
    splits = ['train', 'val', 'test']
    combine_features(data_dir, splits)
    logger.info("Feature combination completed")

if __name__ == '__main__':
    main()