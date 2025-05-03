import numpy as np
from skimage import io, color, morphology, measure
from skimage.filters import threshold_otsu
from sklearn.model_selection import KFold
import cv2
from typing import Tuple, List, Dict
import logging

class TissueSegmenter:
    def __init__(self, 
                 min_tissue_threshold: float = 0.7,
                 patch_size: int = 224,
                 patch_overlap: float = 0.5,
                 n_folds: int = 5):
        """
        Initialize tissue segmenter with configurable parameters
        
        Args:
            min_tissue_threshold: Minimum tissue content in patch (0-1)
            patch_size: Size of extracted patches
            patch_overlap: Overlap between patches (0-1)
            n_folds: Number of folds for cross-validation
        """
        self.min_tissue_threshold = min_tissue_threshold
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.n_folds = n_folds
        self.logger = logging.getLogger(__name__)

    def segment_tissue(self, image: np.ndarray) -> np.ndarray:
        """
        Segment tissue from background using color deconvolution and Otsu's thresholding
        
        Args:
            image: Input RGB image
            
        Returns:
            Binary mask of tissue regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu's thresholding
        thresh = threshold_otsu(gray)
        binary = gray > thresh
        
        # Morphological operations
        binary = morphology.binary_closing(binary, morphology.disk(5))
        binary = morphology.binary_opening(binary, morphology.disk(5))
        
        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=500)
        
        return binary

    def extract_patches(self, 
                       image: np.ndarray, 
                       mask: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Extract patches from image with tissue mask
        
        Args:
            image: Input RGB image
            mask: Binary tissue mask
            
        Returns:
            List of patches and their coordinates
        """
        patches = []
        coordinates = []
        
        stride = int(self.patch_size * (1 - self.patch_overlap))
        
        for y in range(0, image.shape[0] - self.patch_size + 1, stride):
            for x in range(0, image.shape[1] - self.patch_size + 1, stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patch_mask = mask[y:y+self.patch_size, x:x+self.patch_size]
                
                # Check tissue content
                tissue_ratio = np.mean(patch_mask)
                if tissue_ratio >= self.min_tissue_threshold:
                    patches.append(patch)
                    coordinates.append((y, x))
        
        return patches, coordinates

    def cross_validate_patches(self, 
                             patches: List[np.ndarray],
                             labels: List[int]) -> List[Dict]:
        """
        Perform k-fold cross-validation on patches
        
        Args:
            patches: List of image patches
            labels: List of corresponding labels
            
        Returns:
            List of dictionaries containing fold information
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(patches)):
            fold_info = {
                'fold': fold + 1,
                'train_patches': [patches[i] for i in train_idx],
                'val_patches': [patches[i] for i in val_idx],
                'train_labels': [labels[i] for i in train_idx],
                'val_labels': [labels[i] for i in val_idx]
            }
            fold_results.append(fold_info)
        
        return fold_results

    def optimize_parameters(self,
                          image: np.ndarray,
                          true_mask: np.ndarray) -> Dict:
        """
        Optimize segmentation parameters using grid search
        
        Args:
            image: Input RGB image
            true_mask: Ground truth tissue mask
            
        Returns:
            Dictionary of optimal parameters
        """
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
                    self.min_tissue_threshold = min_thresh
                    self.patch_size = size
                    self.patch_overlap = overlap
                    
                    # Segment tissue
                    pred_mask = self.segment_tissue(image)
                    
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