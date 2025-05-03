import numpy as np
from sklearn.cluster import DBSCAN
from typing import Tuple, Dict, List
import logging

class PatchAggregator:
    def __init__(self,
                 confidence_threshold: float = 0.3,
                 min_cluster_size: int = 5,
                 eps: float = 0.1):
        """
        Initialize patch aggregator
        
        Args:
            confidence_threshold: Minimum confidence for BCC prediction
            min_cluster_size: Minimum size for BCC clusters
            eps: Maximum distance between points in a cluster
        """
        self.confidence_threshold = confidence_threshold
        self.min_cluster_size = min_cluster_size
        self.eps = eps
        self.logger = logging.getLogger(__name__)
        
    def aggregate_predictions(self,
                            patch_predictions: np.ndarray,
                            patch_coordinates: List[Tuple[int, int]]) -> Dict:
        """
        Aggregate patch-level predictions to image-level diagnosis
        
        Args:
            patch_predictions: Patch-level prediction probabilities
            patch_coordinates: Coordinates of patches in original image
            
        Returns:
            Dictionary containing:
            - final_prediction: Binary prediction (0 or 1)
            - confidence_score: Overall confidence score
            - bcc_regions: List of BCC regions with coordinates
        """
        # Apply confidence threshold
        bcc_patches = patch_predictions >= self.confidence_threshold
        
        if not np.any(bcc_patches):
            return {
                'final_prediction': 0,
                'confidence_score': 0.0,
                'bcc_regions': []
            }
        
        # Get coordinates of BCC patches
        bcc_coords = np.array([patch_coordinates[i] for i in range(len(patch_coordinates)) 
                             if bcc_patches[i]])
        
        # Cluster BCC patches
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_cluster_size
        ).fit(bcc_coords)
        
        # Get unique clusters
        unique_labels = np.unique(clustering.labels_)
        bcc_regions = []
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            # Get coordinates of patches in this cluster
            cluster_coords = bcc_coords[clustering.labels_ == label]
            
            # Calculate cluster boundaries
            min_y, min_x = np.min(cluster_coords, axis=0)
            max_y, max_x = np.max(cluster_coords, axis=0)
            
            bcc_regions.append({
                'min_y': min_y,
                'min_x': min_x,
                'max_y': max_y,
                'max_x': max_x
            })
        
        # Calculate final prediction and confidence
        final_prediction = 1 if len(bcc_regions) > 0 else 0
        confidence_score = np.mean(patch_predictions[bcc_patches])
        
        return {
            'final_prediction': final_prediction,
            'confidence_score': confidence_score,
            'bcc_regions': bcc_regions
        }
    
    def optimize_parameters(self,
                          patch_predictions: np.ndarray,
                          patch_coordinates: List[Tuple[int, int]],
                          true_labels: np.ndarray) -> Dict:
        """
        Optimize aggregation parameters using grid search
        
        Args:
            patch_predictions: Patch-level prediction probabilities
            patch_coordinates: Coordinates of patches in original image
            true_labels: Ground truth labels
            
        Returns:
            Dictionary of optimal parameters
        """
        param_grid = {
            'confidence_threshold': [0.2, 0.3, 0.4],
            'min_cluster_size': [3, 5, 7],
            'eps': [0.05, 0.1, 0.2]
        }
        
        best_params = None
        best_score = -1
        
        for conf_thresh in param_grid['confidence_threshold']:
            for min_size in param_grid['min_cluster_size']:
                for eps in param_grid['eps']:
                    self.confidence_threshold = conf_thresh
                    self.min_cluster_size = min_size
                    self.eps = eps
                    
                    # Aggregate predictions
                    results = self.aggregate_predictions(
                        patch_predictions,
                        patch_coordinates
                    )
                    
                    # Calculate F1 score
                    pred = results['final_prediction']
                    f1 = self._calculate_f1_score(pred, true_labels)
                    
                    if f1 > best_score:
                        best_score = f1
                        best_params = {
                            'confidence_threshold': conf_thresh,
                            'min_cluster_size': min_size,
                            'eps': eps
                        }
        
        return best_params
    
    def _calculate_f1_score(self,
                          predictions: np.ndarray,
                          true_labels: np.ndarray) -> float:
        """
        Calculate F1 score
        
        Args:
            predictions: Predicted labels
            true_labels: True labels
            
        Returns:
            F1 score
        """
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1 