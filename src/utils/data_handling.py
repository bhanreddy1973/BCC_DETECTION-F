import os
from pathlib import Path
import numpy as np
import tifffile
import pandas as pd
from typing import List, Tuple, Dict
import logging
import cv2

def load_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Loaded image as numpy array
    """
    try:
        # Load image with tifffile
        image = tifffile.imread(image_path)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        raise

def load_tiff_image(image_path: str) -> np.ndarray:
    """
    Load TIFF image with memory mapping
    
    Args:
        image_path: Path to TIFF image
        
    Returns:
        Image as numpy array
    """
    return tifffile.imread(image_path)

def save_patches(patches: List[np.ndarray],
                coordinates: List[Tuple[int, int]],
                output_dir: str,
                image_name: str) -> None:
    """
    Save extracted patches
    
    Args:
        patches: List of image patches
        coordinates: List of patch coordinates
        output_dir: Directory to save patches
        image_name: Name of source image
    """
    patches_dir = os.path.join(output_dir, 'patches')
    os.makedirs(patches_dir, exist_ok=True)
    
    for i, (patch, coord) in enumerate(zip(patches, coordinates)):
        patch_name = f"{image_name}_patch_{i}_y{coord[0]}_x{coord[1]}.npy"
        patch_path = os.path.join(patches_dir, patch_name)
        np.save(patch_path, patch)

def load_patches(patches_dir: str) -> List[np.ndarray]:
    """
    Load saved patches
    
    Args:
        patches_dir: Directory containing patches
        
    Returns:
        List of patches
    """
    patches = []
    for patch_path in Path(patches_dir).glob('*.npy'):
        patches.append(np.load(patch_path))
    return patches

def load_labels(labels_path: str) -> np.ndarray:
    """
    Load labels from CSV file
    
    Args:
        labels_path: Path to labels CSV file
        
    Returns:
        Array of labels
    """
    df = pd.read_csv(labels_path)
    return df['label'].values

def save_features(features: np.ndarray, image_name: str, output_dir: str) -> None:
    """
    Save extracted features to disk
    
    Args:
        features: Feature array to save
        image_name: Name/ID of the source image
        output_dir: Base directory to save features
    """
    try:
        # Create full path including split/class subdirectories
        split_dir = os.path.dirname(image_name)  # e.g. 'test/bcc_high_risk'
        features_dir = os.path.join(output_dir, split_dir)
        os.makedirs(features_dir, exist_ok=True)
        
        # Create features filename from image ID
        image_id = os.path.basename(image_name)  # e.g. '361ac0cd4eca3818a749fe57020e0caa'
        features_path = os.path.join(features_dir, f"{image_id}_features.npy")
        
        # Save features
        np.save(features_path, features)
    except Exception as e:
        logging.error(f"Error saving features for {image_name}: {str(e)}")
        raise

def load_features(features_dir: str) -> Dict[str, np.ndarray]:
    """
    Load saved features
    
    Args:
        features_dir: Directory containing features
        
    Returns:
        Dictionary mapping image names to feature vectors
    """
    features = {}
    for feature_path in Path(features_dir).glob('*.npy'):
        image_name = feature_path.stem.replace('_features', '')
        features[image_name] = np.load(feature_path)
    return features

def save_predictions(predictions: Dict,
                   image_name: str,
                   output_dir: str) -> None:
    """
    Save model predictions
    
    Args:
        predictions: Dictionary of predictions
        image_name: Name of source image
        output_dir: Directory to save predictions
    """
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    predictions_path = os.path.join(predictions_dir, f"{image_name}_predictions.npy")
    np.save(predictions_path, predictions)

def load_predictions(predictions_dir: str) -> Dict[str, Dict]:
    """
    Load saved predictions
    
    Args:
        predictions_dir: Directory containing predictions
        
    Returns:
        Dictionary mapping image names to predictions
    """
    predictions = {}
    for pred_path in Path(predictions_dir).glob('*.npy'):
        image_name = pred_path.stem.replace('_predictions', '')
        predictions[image_name] = np.load(pred_path, allow_pickle=True).item()
    return predictions