import numpy as np
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
import logging
from pathlib import Path
import cv2

from .model import BCCClassifier
from ..config import config
from ..aggregation import PatchAggregator, SpatialCoherenceEnhancer
from ..utils.visualization import PipelineVisualizer

def predict_slide(model: BCCClassifier,
                 image: np.ndarray,
                 patch_coordinates: List[Tuple[int, int]],
                 features: np.ndarray,
                 output_shape: Tuple[int, int],
                 visualizer: Optional[PipelineVisualizer] = None,
                 use_uncertainty: bool = True,
                 batch_size: Optional[int] = None) -> Dict:
    """
    Make predictions for a whole slide image
    
    Args:
        model: Trained BCCClassifier instance
        image: Original WSI image
        patch_coordinates: List of (x,y) coordinates for each patch
        features: Extracted features for each patch
        output_shape: Shape of the output prediction map
        visualizer: Optional visualizer for results
        use_uncertainty: Whether to use MC dropout for uncertainty estimation
        batch_size: Batch size for predictions
        
    Returns:
        Dictionary containing prediction results and metadata
    """
    logger = logging.getLogger(__name__)
    
    # Get patch predictions with uncertainty if requested
    if use_uncertainty:
        patch_predictions = predict_with_uncertainty(
            model, features, n_iterations=10
        )
    else:
        patch_predictions = predict(
            model, features, batch_size=batch_size
        )
    
    # Initialize aggregator
    aggregator = PatchAggregator(
        smoothing_sigma=config.aggregation.smoothing_sigma,
        confidence_threshold=config.aggregation.confidence_threshold,
        bcc_threshold=config.aggregation.bcc_threshold,
        visualizer=visualizer
    )
    
    # Aggregate predictions
    results = aggregator.aggregate_predictions(
        patch_predictions=patch_predictions,
        patch_coordinates=patch_coordinates,
        output_shape=output_shape,
        original_image=image,
        image_name='prediction_results'
    )
    
    return results

def predict(model: BCCClassifier,
           features: np.ndarray,
           batch_size: Optional[int] = None) -> List[Dict]:
    """
    Make predictions using trained model
    
    Args:
        model: Trained BCCClassifier instance
        features: Input features
        batch_size: Batch size for prediction
        
    Returns:
        List of prediction dictionaries
    """
    logger = logging.getLogger(__name__)
    
    if batch_size is None:
        batch_size = config.classification.batch_size
    
    # Make predictions
    logger.info("Making predictions...")
    probabilities = model.predict(
        features,
        batch_size=batch_size,
        return_confidence=True
    )
    
    # Convert to predictions
    predictions = []
    for prob, conf in zip(probabilities[0], probabilities[1]):
        prediction = {
            'prediction': int(prob[1] >= 0.5),  # Class 1 probability
            'confidence': float(conf),
            'raw_probabilities': prob.tolist()
        }
        predictions.append(prediction)
    
    logger.info(f"Made predictions for {len(predictions)} samples")
    return predictions

def predict_with_uncertainty(model: BCCClassifier,
                           features: np.ndarray,
                           n_iterations: int = 10) -> List[Dict]:
    """
    Make predictions with uncertainty estimation using MC Dropout
    
    Args:
        model: Trained BCCClassifier instance with dropout
        features: Input features
        n_iterations: Number of Monte Carlo iterations
        
    Returns:
        List of prediction dictionaries with uncertainty
    """
    logger = logging.getLogger(__name__)
    
    # Enable dropout at inference
    tf.keras.backend.set_learning_phase(1)
    
    # Make multiple predictions
    logger.info(f"Making {n_iterations} predictions for uncertainty estimation...")
    all_probabilities = []
    
    for _ in range(n_iterations):
        probabilities = model.predict(features)
        all_probabilities.append(probabilities)
    
    # Calculate statistics
    all_probabilities = np.array(all_probabilities)
    mean_probabilities = np.mean(all_probabilities, axis=0)
    std_probabilities = np.std(all_probabilities, axis=0)
    
    # Calculate entropy-based uncertainty
    entropy = -np.sum(mean_probabilities * np.log(mean_probabilities + 1e-10), axis=1)
    max_entropy = -np.log(1/mean_probabilities.shape[1])  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy
    
    # Convert to predictions
    predictions = []
    for mean_prob, std_prob, ent in zip(mean_probabilities, std_probabilities, normalized_entropy):
        confidence = 1 - ent  # Use inverse entropy as confidence
        prediction = {
            'prediction': int(mean_prob[1] >= 0.5),  # Class 1 probability
            'confidence': float(confidence),
            'uncertainty': float(ent),
            'std_dev': float(np.mean(std_prob)),
            'raw_probabilities': mean_prob.tolist()
        }
        predictions.append(prediction)
    
    # Disable dropout
    tf.keras.backend.set_learning_phase(0)
    
    logger.info(f"Made predictions with uncertainty for {len(predictions)} samples")
    return predictions

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained model
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded TensorFlow model
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load model directly using TensorFlow
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None