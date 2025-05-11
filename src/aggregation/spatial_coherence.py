import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import logging

@dataclass
class SpatialCoherenceEnhancer:
    """Enhance spatial coherence of patch predictions"""
    eps: float
    min_samples: int
    confidence_threshold: float
    neighborhood_size: int = 3
    
    def enhance_predictions(self,
                          predictions: List[Dict],
                          coordinates: List[Tuple[int, int]]) -> List[Dict]:
        """
        Enhance predictions using spatial coherence
        
        Args:
            predictions: List of patch predictions
            coordinates: List of patch coordinates
            
        Returns:
            Enhanced predictions with spatial coherence
        """
        logger = logging.getLogger(__name__)
        
        # First apply neighborhood voting
        enhanced_preds = self._apply_neighborhood_voting(predictions, coordinates)
        
        # Then perform cluster analysis
        positive_indices = []
        for i, pred in enumerate(enhanced_preds):
            if (pred['prediction'] == 1 and 
                pred['confidence'] >= self.confidence_threshold):
                positive_indices.append(i)
                
        if not positive_indices:
            logger.info("No high-confidence positive predictions found")
            return enhanced_preds
        
        # Get coordinates of positive predictions
        positive_coords = np.array([coordinates[i] for i in positive_indices])
        
        # Perform clustering
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(positive_coords)
        
        # Process clusters
        final_predictions = enhanced_preds.copy()
        clusters = set(clustering.labels_)
        
        if -1 in clusters:  # Remove noise label
            clusters.remove(-1)
            
        logger.info(f"Found {len(clusters)} spatial clusters")
        
        # Enhance predictions based on clusters
        for cluster_id in clusters:
            cluster_mask = clustering.labels_ == cluster_id
            cluster_indices = [positive_indices[i] for i, is_in_cluster in 
                             enumerate(cluster_mask) if is_in_cluster]
            
            cluster_stats = self._calculate_cluster_stats(
                final_predictions, cluster_indices
            )
            
            # Update predictions in cluster
            self._update_cluster_predictions(
                final_predictions, 
                cluster_indices,
                cluster_stats,
                cluster_id
            )
            
        return final_predictions
    
    def _apply_neighborhood_voting(self,
                                 predictions: List[Dict],
                                 coordinates: List[Tuple[int, int]]) -> List[Dict]:
        """Apply weighted majority voting in local neighborhoods"""
        enhanced_preds = predictions.copy()
        coords_array = np.array(coordinates)
        
        for i, (pred, coord) in enumerate(zip(predictions, coordinates)):
            # Find neighbors within neighborhood_size
            distances = np.linalg.norm(coords_array - coord, axis=1)
            neighbor_indices = np.where(distances <= self.neighborhood_size)[0]
            
            if len(neighbor_indices) < 2:  # Skip if no neighbors found
                continue
                
            # Get weighted votes
            neighbor_preds = [predictions[j]['prediction'] for j in neighbor_indices]
            neighbor_confs = [predictions[j]['confidence'] for j in neighbor_indices]
            
            # Calculate weighted average
            weighted_pred = np.average(
                neighbor_preds,
                weights=neighbor_confs
            )
            
            # Update prediction if confidence improved
            if abs(weighted_pred - 0.5) > abs(pred['confidence'] - 0.5):
                enhanced_preds[i]['prediction'] = int(weighted_pred > 0.5)
                enhanced_preds[i]['confidence'] = abs(weighted_pred - 0.5) * 2
                
        return enhanced_preds
    
    def _calculate_cluster_stats(self,
                               predictions: List[Dict],
                               cluster_indices: List[int]) -> Dict:
        """Calculate statistics for a cluster"""
        confidences = [predictions[i]['confidence'] for i in cluster_indices]
        sizes = len(cluster_indices)
        
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'size': sizes,
            'density': sizes / (np.pi * self.eps ** 2)  # Approximate density
        }
    
    def _update_cluster_predictions(self,
                                  predictions: List[Dict],
                                  cluster_indices: List[int],
                                  cluster_stats: Dict,
                                  cluster_id: int) -> None:
        """Update predictions for a cluster based on its statistics"""
        mean_conf = cluster_stats['mean_confidence']
        density = cluster_stats['density']
        
        # Boost confidence based on cluster properties
        confidence_boost = min(0.2, density * 0.1)  # Cap boost at 0.2
        
        for idx in cluster_indices:
            predictions[idx].update({
                'confidence': min(1.0, mean_conf + confidence_boost),
                'cluster_id': int(cluster_id),
                'cluster_size': cluster_stats['size'],
                'cluster_density': density
            })