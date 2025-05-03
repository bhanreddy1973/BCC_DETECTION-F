import argparse
import os
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.preprocessing.tissue_segmentation import TissueSegmenter
from src.feature_extraction.efficientnet import EfficientNetFeatureExtractor
from src.classification.model import BCCClassifier
from src.aggregation.patch_aggregation import PatchAggregator
from src.config import config
from src.utils.data_handling import load_tiff_image
from src.utils.visualization import visualize_results

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    """Main function for making predictions"""
    parser = argparse.ArgumentParser(description='Make BCC predictions')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image
    logger.info(f"Loading image: {args.image_path}")
    image = load_tiff_image(args.image_path)
    
    # Initialize components
    segmenter = TissueSegmenter(
        min_tissue_threshold=config.preprocessing.min_tissue_threshold,
        patch_size=config.preprocessing.patch_size,
        patch_overlap=config.preprocessing.patch_overlap
    )
    
    feature_extractor = EfficientNetFeatureExtractor(
        input_shape=config.feature_extraction.input_shape,
        pca_variance=config.feature_extraction.pca_variance,
        batch_size=config.feature_extraction.batch_size
    )
    
    classifier = BCCClassifier(
        input_dim=config.classification.input_dim,
        learning_rate=config.classification.learning_rate,
        dropout_rate=config.classification.dropout_rate,
        batch_size=config.classification.batch_size
    )
    
    aggregator = PatchAggregator(
        confidence_threshold=config.aggregation.confidence_threshold,
        min_cluster_size=config.aggregation.min_cluster_size,
        eps=config.aggregation.eps
    )
    
    # Load trained model
    logger.info(f"Loading model from {args.model_path}")
    classifier.model = tf.keras.models.load_model(args.model_path)
    
    # Process image
    logger.info("Segmenting tissue...")
    mask = segmenter.segment_tissue(image)
    
    logger.info("Extracting patches...")
    patches, coordinates = segmenter.extract_patches(image, mask)
    
    logger.info("Extracting features...")
    features = feature_extractor.extract_features(np.array(patches))
    
    logger.info("Making predictions...")
    patch_predictions = classifier.predict(features)
    
    logger.info("Aggregating predictions...")
    results = aggregator.aggregate_predictions(patch_predictions, coordinates)
    
    # Save results
    output_path = os.path.join(args.output_dir, 'results.npy')
    np.save(output_path, results)
    logger.info(f"Saved results to {output_path}")
    
    # Generate visualization
    vis_path = os.path.join(args.output_dir, 'visualization.png')
    visualize_results(image, results, vis_path)
    logger.info(f"Saved visualization to {vis_path}")
    
    # Print results
    logger.info(f"Final prediction: {'BCC detected' if results['final_prediction'] else 'No BCC detected'}")
    logger.info(f"Confidence score: {results['confidence_score']:.2f}")
    logger.info(f"Number of BCC regions: {len(results['bcc_regions'])}")
    
    logger.info("Prediction completed successfully")

if __name__ == '__main__':
    main() 