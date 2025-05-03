import argparse
import os
import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

from src.feature_extraction.efficientnet import EfficientNetFeatureExtractor
from src.classification.model import BCCClassifier
from src.config import config
from src.utils.data_handling import load_patches, load_labels
from src.utils.optimization import enable_mixed_precision

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data(data_dir: str) -> tuple:
    """
    Load training data
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        Tuple of (patches, labels)
    """
    logger = logging.getLogger(__name__)
    
    # Load patches
    patches_dir = os.path.join(data_dir, 'patches')
    patches = load_patches(patches_dir)
    logger.info(f"Loaded {len(patches)} patches")
    
    # Load labels
    labels_path = os.path.join(data_dir, 'labels.csv')
    labels = load_labels(labels_path)
    logger.info(f"Loaded {len(labels)} labels")
    
    return patches, labels

def main():
    """Main function for training model"""
    parser = argparse.ArgumentParser(description='Train BCC detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing processed data')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save trained model')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Enable mixed precision training
    if config.training.mixed_precision:
        enable_mixed_precision()
        logger.info("Enabled mixed precision training")
    
    # Load data
    patches, labels = load_data(args.data_dir)
    
    # Initialize feature extractor
    feature_extractor = EfficientNetFeatureExtractor(
        input_shape=config.feature_extraction.input_shape,
        pca_variance=config.feature_extraction.pca_variance,
        batch_size=config.feature_extraction.batch_size
    )
    
    # Extract features
    logger.info("Extracting features...")
    features = feature_extractor.extract_features(np.array(patches))
    logger.info(f"Extracted features with shape: {features.shape}")
    
    # Initialize classifier
    classifier = BCCClassifier(
        input_dim=features.shape[1],
        learning_rate=config.classification.learning_rate,
        dropout_rate=config.classification.dropout_rate,
        batch_size=config.classification.batch_size
    )
    
    # Setup TensorBoard
    tensorboard_callback = TensorBoard(
        log_dir=os.path.join(args.model_dir, 'logs'),
        histogram_freq=1
    )
    
    # Train model
    logger.info("Starting model training...")
    history = classifier.train(
        features, labels,
        epochs=config.classification.epochs,
        callbacks=[tensorboard_callback]
    )
    
    # Save model
    model_path = os.path.join(args.model_dir, 'best_model.h5')
    classifier.model.save(model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save training history
    history_path = os.path.join(args.model_dir, 'training_history.npy')
    np.save(history_path, history)
    logger.info(f"Saved training history to {history_path}")
    
    logger.info("Training completed successfully")

if __name__ == '__main__':
    main() 