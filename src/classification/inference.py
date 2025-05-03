import numpy as np
from typing import Dict, List
import tensorflow as tf
import logging

from .model import BCCClassifier
from ..config import config

def predict(model: BCCClassifier,
           features: np.ndarray,
           batch_size: int = None) -> List[Dict]:
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
        batch_size=batch_size
    )
    
    # Convert to predictions
    predictions = []
    for prob in probabilities:
        prediction = {
            'prediction': int(prob >= 0.5),
            'confidence': float(prob if prob >= 0.5 else 1 - prob)
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
    model.enable_dropout()
    
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
    
    # Convert to predictions
    predictions = []
    for mean_prob, std_prob in zip(mean_probabilities, std_probabilities):
        prediction = {
            'prediction': int(mean_prob >= 0.5),
            'confidence': float(mean_prob if mean_prob >= 0.5 else 1 - mean_prob),
            'uncertainty': float(std_prob)
        }
        predictions.append(prediction)
    
    # Disable dropout
    model.disable_dropout()
    
    logger.info(f"Made predictions with uncertainty for {len(predictions)} samples")
    return predictions 