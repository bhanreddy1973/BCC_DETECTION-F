import numpy as np
from typing import List, Dict
import cv2
from dataclasses import dataclass
import os
import logging

@dataclass
class ColorFeatureExtractor:
    """Class for extracting color-based features"""
    color_spaces: List[str]
    
    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """Validate input image format"""
        if not isinstance(image, np.ndarray):
            self.logger.error(f"Input must be numpy array, got {type(image)}")
            return False
        if len(image.shape) != 3:
            self.logger.error(f"Image must be 3D (H,W,C), got shape {image.shape}")
            return False
        if image.shape[2] != 3:
            self.logger.error(f"Image must have 3 channels (RGB), got {image.shape[2]} channels")
            return False
        return True
    
    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract color features from image
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Dictionary of color features
        """
        try:
            if not self._validate_image(image):
                return {}
                
            features = {}
            
            for color_space in self.color_spaces:
                try:
                    if color_space == "HSV":
                        converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    elif color_space == "Lab":
                        converted = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
                    elif color_space == "YCbCr":
                        converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
                    else:
                        converted = image
                        
                    # Calculate statistics for each channel
                    for i, channel in enumerate(cv2.split(converted)):
                        prefix = f"{color_space}_{i}"
                        features[f"{prefix}_mean"] = np.mean(channel)
                        features[f"{prefix}_std"] = np.std(channel)
                        features[f"{prefix}_skew"] = np.mean((channel - np.mean(channel))**3)
                        features[f"{prefix}_kurtosis"] = np.mean((channel - np.mean(channel))**4)
                except Exception as e:
                    self.logger.error(f"Error processing color space {color_space}: {str(e)}")
                    continue
                    
            return features
            
        except Exception as e:
            self.logger.error(f"Error in extract_features: {str(e)}")
            return {}
    
    def compute_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute texture features using GLCM
        
        Args:
            image: Input grayscale image
            
        Returns:
            Dictionary of texture features
        """
        try:
            from skimage.feature import graycomatrix, graycoprops
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Normalize to [0, 255] if needed
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
                
            # Compute GLCM
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray, distances, angles, symmetric=True, normed=True)
            
            # Compute properties
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            features = {}
            
            for prop in properties:
                values = graycoprops(glcm, prop)
                features[f"glcm_{prop}_mean"] = np.mean(values)
                features[f"glcm_{prop}_std"] = np.std(values)
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error in compute_texture_features: {str(e)}")
            return {}
    
    def combine_features(self, color_features: Dict[str, float],
                        texture_features: Dict[str, float]) -> np.ndarray:
        """
        Combine color and texture features
        
        Args:
            color_features: Color features dictionary
            texture_features: Texture features dictionary
            
        Returns:
            Combined feature vector
        """
        try:
            all_features = {**color_features, **texture_features}
            return np.array(list(all_features.values()))
        except Exception as e:
            self.logger.error(f"Error in combine_features: {str(e)}")
            return np.array([])

    def save_features(self, features: Dict[str, np.ndarray], image_name: str, output_dir: str):
        try:
            for feature_name, feature_value in features.items():
                save_path = os.path.join(output_dir, feature_name)
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, f"{image_name}_features.npy"), feature_value)
        except Exception as e:
            self.logger.error(f"Error saving features: {str(e)}") 