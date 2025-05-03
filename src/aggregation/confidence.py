import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ConfidenceScorer:
    """Calculate confidence scores for predictions"""
    spatial_weight: float = 0.5
    cluster_size_weight: float = 0.3
    uncertainty_weight: float = 0.2
    
    def calculate_confidence(self,
                           predictions: List[Dict],
                           cluster_stats: Dict[int, Dict]) -> float:
        """
        Calculate overall confidence score
        
        Args:
            predictions: List of predictions
            cluster_stats: Cluster statistics
            
        Returns:
            Confidence score
        """
        if not predictions:
            return 0.0
            
        # Calculate spatial confidence
        spatial_confidence = self._calculate_spatial_confidence(cluster_stats)
        
        # Calculate prediction confidence
        pred_confidence = np.mean([p['confidence'] for p in predictions])
        
        # Calculate uncertainty-based confidence if available
        if 'uncertainty' in predictions[0]:
            uncertainty_confidence = 1 - np.mean([p['uncertainty'] for p in predictions])
        else:
            uncertainty_confidence = 1.0
            
        # Combine scores
        confidence = (
            self.spatial_weight * spatial_confidence +
            self.cluster_size_weight * pred_confidence +
            self.uncertainty_weight * uncertainty_confidence
        )
        
        return float(confidence)
    
    def _calculate_spatial_confidence(self, cluster_stats: Dict[int, Dict]) -> float:
        """
        Calculate confidence based on spatial clustering
        
        Args:
            cluster_stats: Cluster statistics
            
        Returns:
            Spatial confidence score
        """
        if not cluster_stats:
            return 0.0
            
        # Calculate weighted average of cluster confidences
        total_weight = 0
        weighted_confidence = 0
        
        for stats in cluster_stats.values():
            size = stats['size']
            confidence = stats['mean_confidence']
            
            # Weight by cluster size
            weight = size
            weighted_confidence += weight * confidence
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        return weighted_confidence / total_weight
    
    def get_confidence_breakdown(self,
                               predictions: List[Dict],
                               cluster_stats: Dict[int, Dict]) -> Dict:
        """
        Get detailed confidence breakdown
        
        Args:
            predictions: List of predictions
            cluster_stats: Cluster statistics
            
        Returns:
            Dictionary of confidence components
        """
        spatial_confidence = self._calculate_spatial_confidence(cluster_stats)
        pred_confidence = np.mean([p['confidence'] for p in predictions])
        
        if 'uncertainty' in predictions[0]:
            uncertainty_confidence = 1 - np.mean([p['uncertainty'] for p in predictions])
        else:
            uncertainty_confidence = 1.0
            
        return {
            'spatial_confidence': float(spatial_confidence),
            'prediction_confidence': float(pred_confidence),
            'uncertainty_confidence': float(uncertainty_confidence),
            'overall_confidence': float(
                self.spatial_weight * spatial_confidence +
                self.cluster_size_weight * pred_confidence +
                self.uncertainty_weight * uncertainty_confidence
            )
        } 