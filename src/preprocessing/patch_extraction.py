import numpy as np
from typing import List, Tuple, Optional
import cv2
from dataclasses import dataclass

@dataclass
class PatchExtractor:
    """Class for extracting patches from tissue-segmented images"""
    patch_size: int
    patch_overlap: float
    min_tissue_ratio: float = 0.5
    
    def extract_patches(self, 
                       image: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract patches from image
        
        Args:
            image: Input image
            mask: Optional tissue mask
            
        Returns:
            Tuple of (patches, coordinates)
        """
        if mask is None:
            mask = np.ones_like(image[:, :, 0] if len(image.shape) == 3 else image)
            
        h, w = image.shape[:2]
        stride = int(self.patch_size * (1 - self.patch_overlap))
        
        patches = []
        coordinates = []
        
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                patch_mask = mask[y:y + self.patch_size, x:x + self.patch_size]
                if np.mean(patch_mask) >= self.min_tissue_ratio:
                    patch = image[y:y + self.patch_size, x:x + self.patch_size]
                    patches.append(patch)
                    coordinates.append((y, x))
        
        return patches, coordinates
    
    def normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Normalize patch for model input
        
        Args:
            patch: Input patch
            
        Returns:
            Normalized patch
        """
        # Convert to float32
        patch = patch.astype(np.float32)
        
        # Scale to [0, 1]
        if patch.max() > 1:
            patch = patch / 255.0
            
        return patch
    
    def augment_patch(self, 
                     patch: np.ndarray,
                     augmentation_type: str) -> np.ndarray:
        """
        Apply augmentation to patch
        
        Args:
            patch: Input patch
            augmentation_type: Type of augmentation
            
        Returns:
            Augmented patch
        """
        if augmentation_type == "flip_h":
            return cv2.flip(patch, 1)
        elif augmentation_type == "flip_v":
            return cv2.flip(patch, 0)
        elif augmentation_type == "rotate90":
            return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        elif augmentation_type == "rotate180":
            return cv2.rotate(patch, cv2.ROTATE_180)
        elif augmentation_type == "rotate270":
            return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return patch 