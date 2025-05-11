import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Dict, Optional
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

from src.config import config

class PatchAggregator:
    def __init__(self,
                 smoothing_sigma: float = 2.0,
                 confidence_threshold: float = 0.5,
                 bcc_threshold: float = 0.3,  # Threshold for BCC classification
                 n_jobs: int = 48,
                 batch_size: int = 64,
                 visualizer = None):
        """
        Initialize patch aggregation with visualization support
        
        Args:
            smoothing_sigma: Gaussian smoothing parameter
            confidence_threshold: Threshold for confident predictions
            bcc_threshold: Threshold for BCC-positive classification
            n_jobs: Number of parallel jobs
            batch_size: Batch size for processing
            visualizer: PipelineVisualizer instance
        """
        self.smoothing_sigma = smoothing_sigma
        self.confidence_threshold = confidence_threshold
        self.bcc_threshold = bcc_threshold
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.visualizer = visualizer
        self.logger = logging.getLogger(__name__)
        
        # Initialize helper classes
        self.spatial_enhancer = SpatialCoherenceEnhancer(
            eps=50,  # Maximum distance between patches in a cluster
            min_samples=5,  # Minimum patches for a cluster
            confidence_threshold=confidence_threshold
        )
        self.confidence_scorer = ConfidenceScorer()
        
    def _process_patch_batch(self, 
                           patch_predictions: np.ndarray,
                           patch_coordinates: List[Tuple[int, int]],
                           confidence_scores: np.ndarray,
                           output_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch of patches in parallel"""
        prediction_map = np.zeros(output_shape)
        confidence_map = np.zeros(output_shape)
        weight_map = np.zeros(output_shape)
        
        patch_size = config.preprocessing.patch_size
        
        for pred, (x, y), conf in zip(patch_predictions, patch_coordinates, confidence_scores):
            # Apply gaussian weights centered on patch
            y_grid, x_grid = np.ogrid[-patch_size//2:patch_size//2, -patch_size//2:patch_size//2]
            weights = np.exp(-(x_grid**2 + y_grid**2)/(2*self.smoothing_sigma**2))
            weights = weights / weights.sum()
            
            # Update maps
            prediction_map[y:y+patch_size, x:x+patch_size] += pred * weights * conf
            confidence_map[y:y+patch_size, x:x+patch_size] += conf * weights
            weight_map[y:y+patch_size, x:x+patch_size] += weights
        
        # Normalize predictions by weights
        valid_mask = weight_map > 0
        prediction_map[valid_mask] /= weight_map[valid_mask]
        confidence_map[valid_mask] /= weight_map[valid_mask]
        
        return prediction_map, confidence_map

    def aggregate_predictions(self,
                            patch_predictions: List[Dict],
                            patch_coordinates: List[Tuple[int, int]],
                            output_shape: Tuple[int, int],
                            original_image: Optional[np.ndarray] = None,
                            image_name: Optional[str] = None) -> Dict:
        """
        Aggregate patch predictions with spatial coherence
        
        Args:
            patch_predictions: List of patch prediction dictionaries
            patch_coordinates: (x,y) coordinates for each patch
            output_shape: Shape of output probability map
            original_image: Optional original image for visualization
            image_name: Optional image name for visualization
            
        Returns:
            Dictionary containing aggregation results
        """
        # Extract raw predictions and confidence scores
        predictions = np.array([p['prediction'] for p in patch_predictions])
        confidences = np.array([p['confidence'] for p in patch_predictions])
        
        # Apply spatial coherence enhancement
        enhanced_predictions = self.spatial_enhancer.enhance_predictions(
            patch_predictions,
            patch_coordinates
        )
        
        # Split data into batches for parallel processing
        n_patches = len(enhanced_predictions)
        n_batches = (n_patches + self.batch_size - 1) // self.batch_size
        
        prediction_maps = []
        confidence_maps = []
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, n_patches)
                
                future = executor.submit(
                    self._process_patch_batch,
                    predictions[start_idx:end_idx],
                    patch_coordinates[start_idx:end_idx],
                    confidences[start_idx:end_idx],
                    output_shape
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                pred_map, conf_map = future.result()
                prediction_maps.append(pred_map)
                confidence_maps.append(conf_map)
        
        # Combine maps
        final_prediction_map = np.mean(prediction_maps, axis=0)
        final_confidence_map = np.mean(confidence_maps, axis=0)
        
        # Apply spatial smoothing
        final_prediction_map = gaussian_filter(final_prediction_map, sigma=self.smoothing_sigma)
        final_confidence_map = gaussian_filter(final_confidence_map, sigma=self.smoothing_sigma)
        
        # Calculate uncertainty map
        uncertainty_map = 1 - final_confidence_map
        
        # Get cluster statistics
        cluster_stats = {
            p['cluster_id']: {
                'size': p['cluster_size'],
                'mean_confidence': p['confidence'],
                'density': p.get('cluster_density', 0)
            }
            for p in enhanced_predictions 
            if 'cluster_id' in p
        }
        
        # Calculate slide-level confidence
        slide_confidence = self.confidence_scorer.calculate_confidence(
            enhanced_predictions,
            cluster_stats
        )
        
        # Get confidence breakdown
        confidence_breakdown = self.confidence_scorer.get_confidence_breakdown(
            enhanced_predictions,
            cluster_stats
        )
        
        # Make final BCC classification
        is_bcc = np.mean(final_prediction_map > 0.5) >= self.bcc_threshold
        
        # Generate region proposals
        region_proposals = self.get_region_proposals(
            final_prediction_map,
            final_confidence_map
        )
        
        # Visualize if visualizer is available
        if self.visualizer and original_image is not None and image_name is not None:
            self.visualizer.visualize_aggregation(
                original_image=original_image,
                final_prediction_mask=final_prediction_map,
                uncertainty_mask=uncertainty_map,
                image_name=image_name
            )
            
            # Create summary report
            metrics = {
                'mean_confidence': float(np.mean(final_confidence_map)),
                'uncertainty_level': float(np.mean(uncertainty_map)),
                'positive_ratio': float(np.mean(final_prediction_map > 0.5)),
                'high_confidence_ratio': float(np.mean(final_confidence_map > 0.8))
            }
            
            processing_time = {
                'aggregation': 0.0,
                'smoothing': 0.0
            }
            
            memory_usage = {
                'peak_memory': float(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
            }
            
            self.visualizer.create_summary_report(
                metrics=metrics,
                processing_time=processing_time,
                memory_usage=memory_usage,
                image_name=image_name
            )
        
        # Clean up intermediate results
        del prediction_maps, confidence_maps
        gc.collect()
        
        return {
            'prediction_map': final_prediction_map,
            'confidence_map': final_confidence_map,
            'uncertainty_map': uncertainty_map,
            'is_bcc': bool(is_bcc),
            'slide_confidence': float(slide_confidence),
            'confidence_breakdown': confidence_breakdown,
            'region_proposals': region_proposals
        }

    def get_region_proposals(self,
                           prediction_map: np.ndarray,
                           confidence_map: np.ndarray,
                           min_size: int = 1000,
                           threshold: float = 0.5) -> List[Dict]:
        """Extract region proposals for high-confidence BCC regions"""
        from skimage import measure
        
        # Threshold prediction map
        binary_pred = prediction_map > threshold
        
        # Find connected components
        labels = measure.label(binary_pred)
        regions = measure.regionprops(labels, intensity_image=confidence_map)
        
        proposals = []
        for region in regions:
            if region.area >= min_size:
                props = {
                    'bbox': region.bbox,
                    'area': region.area,
                    'centroid': region.centroid,
                    'mean_confidence': region.mean_intensity,
                    'max_confidence': region.max_intensity
                }
                proposals.append(props)
        
        return proposals

    def refine_predictions(self,
                          prediction_map: np.ndarray,
                          uncertainty_map: np.ndarray,
                          high_uncertainty_threshold: float = 0.7) -> np.ndarray:
        """Refine predictions using uncertainty information"""
        # Identify high uncertainty regions
        high_uncertainty = uncertainty_map > high_uncertainty_threshold
        
        # Apply stricter threshold in high uncertainty regions
        refined_predictions = prediction_map.copy()
        refined_predictions[high_uncertainty] = 0.5  # Neutral prediction for uncertain regions
        
        return refined_predictions