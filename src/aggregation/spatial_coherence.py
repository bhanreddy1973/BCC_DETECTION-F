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
    
    def enhance_predictions(self,
                          predictions: List[Dict],
                          coordinates: List[Tuple[int, int]]) -> List[Dict]:
        """
        Enhance predictions using spatial coherence
        
        Args:
            predictions: List of patch predictions
            coordinates: List of patch coordinates
            
        Returns:
            Enhanced predictions
        """
        logger = logging.getLogger(__name__)
        
        # Filter high-confidence positive predictions
        positive_indices = []
        for i, pred in enumerate(predictions):
            if (pred['prediction'] == 1 and 
                pred['confidence'] >= self.confidence_threshold):
                positive_indices.append(i)
                
        if not positive_indices:
            logger.info("No high-confidence positive predictions found")
            return predictions
        
        # Get coordinates of positive predictions
        positive_coords = np.array([coordinates[i] for i in positive_indices])
        
        # Perform clustering
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        ).fit(positive_coords)
        
        # Process clusters
        enhanced_predictions = predictions.copy()
        clusters = set(clustering.labels_)
        
        if -1 in clusters:  # Remove noise label
            clusters.remove(-1)
            
        logger.info(f"Found {len(clusters)} spatial clusters")
        
        # Enhance predictions based on clusters
        for cluster_id in clusters:
            # Get indices of predictions in cluster
            cluster_mask = clustering.labels_ == cluster_id
            cluster_indices = [positive_indices[i] for i, is_in_cluster in 
                             enumerate(cluster_mask) if is_in_cluster]
            
            # Calculate cluster statistics
            cluster_confidences = [predictions[i]['confidence'] 
                                 for i in cluster_indices]
            mean_confidence = np.mean(cluster_confidences)
            
            # Enhance predictions in cluster
            for idx in cluster_indices:
                enhanced_predictions[idx]['confidence'] = mean_confidence
                enhanced_predictions[idx]['cluster_id'] = int(cluster_id)
                
        return enhanced_predictions
    
    def get_cluster_statistics(self,
                             predictions: List[Dict]) -> Dict[int, Dict]:
        """
        Get statistics for each cluster
        
        Args:
            predictions: Enhanced predictions with cluster IDs
            
        Returns:
            Dictionary mapping cluster IDs to statistics
        """
        cluster_stats = {}
        
        for pred in predictions:
            if 'cluster_id' in pred:
                cluster_id = pred['cluster_id']
                if cluster_id not in cluster_stats:
                    cluster_stats[cluster_id] = {
                        'size': 0,
                        'confidences': []
                    }
                    
                cluster_stats[cluster_id]['size'] += 1
                cluster_stats[cluster_id]['confidences'].append(pred['confidence'])
                
        # Calculate summary statistics
        for cluster_id in cluster_stats:
            confidences = cluster_stats[cluster_id]['confidences']
            cluster_stats[cluster_id].update({
                'mean_confidence': np.mean(confidences),
                'std_confidence': np.std(confidences)
            })
            
        return cluster_stats 