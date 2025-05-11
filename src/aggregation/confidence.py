import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class ConfidenceScorer:
    """Calculate confidence scores for predictions"""
    spatial_weight: float = 0.4
    cluster_size_weight: float = 0.3
    density_weight: float = 0.2
    uncertainty_weight: float = 0.1
    
    def calculate_confidence(self,
                           predictions: List[Dict],
                           cluster_stats: Dict[int, Dict]) -> float:
        """
        Calculate overall confidence score
        
        Args:
            predictions: List of predictions with cluster info
            cluster_stats: Statistics for each cluster
            
        Returns:
            Confidence score between 0 and 1
        """
        if not predictions:
            return 0.0
            
        # Calculate spatial confidence from clusters
        spatial_confidence = self._calculate_spatial_confidence(cluster_stats)
        
        # Calculate cluster size confidence
        size_confidence = self._calculate_size_confidence(cluster_stats)
        
        # Calculate density confidence
        density_confidence = self._calculate_density_confidence(cluster_stats)
        
        # Calculate prediction confidence
        pred_confidence = np.mean([p['confidence'] for p in predictions])
        
        # Calculate uncertainty-based confidence
        uncertainty_confidence = self._calculate_uncertainty_confidence(predictions)
        
        # Combine scores with weights
        confidence = (
            self.spatial_weight * spatial_confidence +
            self.cluster_size_weight * size_confidence +
            self.density_weight * density_confidence +
            self.uncertainty_weight * uncertainty_confidence
        )
        
        return float(min(1.0, confidence))
    
    def _calculate_spatial_confidence(self, cluster_stats: Dict[int, Dict]) -> float:
        """Calculate confidence based on spatial coherence"""
        if not cluster_stats:
            return 0.0
            
        # Weight by cluster size and mean confidence
        total_weight = 0
        weighted_confidence = 0
        
        for stats in cluster_stats.values():
            size = stats['size']
            confidence = stats['mean_confidence']
            
            weight = size
            weighted_confidence += weight * confidence
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        return weighted_confidence / total_weight
    
    def _calculate_size_confidence(self, cluster_stats: Dict[int, Dict]) -> float:
        """Calculate confidence based on cluster sizes"""
        if not cluster_stats:
            return 0.0
            
        # Get total area covered by clusters
        total_size = sum(stats['size'] for stats in cluster_stats.values())
        
        # Normalize by expected tumor size range (500-5000 pixels)
        size_confidence = np.clip(total_size / 2000, 0, 1)
        
        return size_confidence
    
    def _calculate_density_confidence(self, cluster_stats: Dict[int, Dict]) -> float:
        """Calculate confidence based on cluster density"""
        if not cluster_stats:
            return 0.0
            
        # Weight density by cluster size
        total_weight = 0
        weighted_density = 0
        
        for stats in cluster_stats.values():
            size = stats['size']
            density = stats.get('density', 0)
            
            weight = size
            weighted_density += weight * density
            total_weight += weight
            
        if total_weight == 0:
            return 0.0
            
        # Normalize density score
        density_confidence = np.clip(weighted_density / total_weight / 2, 0, 1)
        
        return density_confidence
    
    def _calculate_uncertainty_confidence(self, predictions: List[Dict]) -> float:
        """Calculate confidence based on prediction uncertainty"""
        if not predictions:
            return 0.0
            
        # Use prediction confidence as inverse uncertainty
        confidences = [p['confidence'] for p in predictions]
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        # Penalize high variance in confidence
        uncertainty_confidence = mean_confidence * (1 - std_confidence)
        
        return float(uncertainty_confidence)
    
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
        size_confidence = self._calculate_size_confidence(cluster_stats)
        density_confidence = self._calculate_density_confidence(cluster_stats)
        uncertainty_confidence = self._calculate_uncertainty_confidence(predictions)
        
        overall_confidence = (
            self.spatial_weight * spatial_confidence +
            self.cluster_size_weight * size_confidence +
            self.density_weight * density_confidence +
            self.uncertainty_weight * uncertainty_confidence
        )
        
        return {
            'spatial_confidence': float(spatial_confidence),
            'size_confidence': float(size_confidence),
            'density_confidence': float(density_confidence),
            'uncertainty_confidence': float(uncertainty_confidence),
            'overall_confidence': float(min(1.0, overall_confidence))
        }