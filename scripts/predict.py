import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
import tifffile
import cv2
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
import traceback

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    MODEL_DIR, BEST_MODEL_PATH, CONFIDENCE_THRESHOLD,
    TOP_K_PREDICTIONS, CLASS_NAMES, PATCH_SIZE,
    VISUALIZATION_DIR, LOGS_DIR, PATCH_STRIDE, MIN_TISSUE_RATIO
)
from src.preprocessing.tissue_segmentation import TissueSegmenter
from src.preprocessing.patch_extraction import PatchExtractor
from src.models.model import create_model, load_tensorflow_model, load_pytorch_model

def parse_args():
    parser = argparse.ArgumentParser(description='BCC Detection Prediction')
    parser.add_argument('--model_path', type=str, default=str(BEST_MODEL_PATH),
                      help='Path to the trained model')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Path to the input image')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save predictions and visualizations')
    parser.add_argument('--use_uncertainty', action='store_true',
                      help='Enable uncertainty estimation')
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of Monte Carlo samples for uncertainty estimation')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for prediction')
    return parser.parse_args()

def load_model(model_path):
    """Load the trained model."""
    try:
        if model_path.endswith('.h5'):
            # Load TensorFlow model
            model = load_tensorflow_model(model_path)
        else:
            # Load PyTorch model
            model = load_pytorch_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image_path, segmenter, patch_extractor):
    """Preprocess image for prediction."""
    try:
        # Load and segment image
        image = segmenter.load_image(image_path)
        if image is None:
            return None, None, None

        # Segment tissue
        mask, _ = segmenter.segment_tissue(image)
        
        # Extract patches
        patches, coordinates, patch_metrics = patch_extractor.extract_patches(image, mask)
        logger.info(f"Extracted {len(patches)} patches")
        
        if not patches:
            logger.warning(f"No valid patches extracted from {image_path}")
            return None, None, None
            
        return patches, coordinates, patch_metrics
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None, None, None

def predict_patches(model, patches, device, use_uncertainty=False, num_samples=10):
    """Make predictions on patches with optional uncertainty estimation."""
    try:
        # Convert patches to appropriate format
        if isinstance(model, tf.keras.Model):
            # TensorFlow model
            patches_tensor = np.array(patches)
            if use_uncertainty:
                # Monte Carlo Dropout for uncertainty estimation
                predictions = []
                for _ in range(num_samples):
                    outputs = model(patches_tensor, training=True)  # Enable dropout
                    probs = tf.nn.softmax(outputs, axis=1)
                    predictions.append(probs.numpy())
                
                # Stack predictions and calculate mean and std
                predictions = np.stack(predictions)
                mean_probs = np.mean(predictions, axis=0)
                std_probs = np.std(predictions, axis=0)
                
                return mean_probs, std_probs
            else:
                # Standard prediction
                outputs = model(patches_tensor, training=False)
                probabilities = tf.nn.softmax(outputs, axis=1)
                return probabilities.numpy(), None
        else:
            # PyTorch model
            patches_tensor = torch.from_numpy(np.array(patches)).float()
            patches_tensor = patches_tensor.permute(0, 3, 1, 2)  # Change to NCHW format
            patches_tensor = patches_tensor.to(device)
            
            if use_uncertainty:
                # Monte Carlo Dropout for uncertainty estimation
                model.train()  # Enable dropout
                predictions = []
                for _ in range(num_samples):
                    with torch.no_grad():
                        outputs = model(patches_tensor)
                        probs = F.softmax(outputs, dim=1)
                        predictions.append(probs)
                model.eval()  # Disable dropout
                
                # Stack predictions and calculate mean and std
                predictions = torch.stack(predictions)
                mean_probs = predictions.mean(dim=0)
                std_probs = predictions.std(dim=0)
                
                return mean_probs.cpu().numpy(), std_probs.cpu().numpy()
            else:
                # Standard prediction
                with torch.no_grad():
                    outputs = model(patches_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                return probabilities.cpu().numpy(), None
            
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return None, None

def aggregate_predictions(probabilities, uncertainties, coordinates, metrics):
    """Aggregate patch predictions into image-level prediction."""
    try:
        # Calculate weighted average based on tissue ratio
        weights = np.array([m['tissue_ratio'] for m in metrics])
        weights = weights / weights.sum()
        
        weighted_probs = np.average(probabilities, weights=weights, axis=0)
        
        if uncertainties is not None:
            weighted_uncertainties = np.average(uncertainties, weights=weights, axis=0)
        else:
            weighted_uncertainties = None
        
        # Get top predictions
        top_indices = np.argsort(weighted_probs)[-TOP_K_PREDICTIONS:][::-1]
        top_predictions = [
            {
                'class': CLASS_NAMES[idx],
                'probability': float(weighted_probs[idx]),
                'uncertainty': float(weighted_uncertainties[idx]) if weighted_uncertainties is not None else None
            }
            for idx in top_indices
        ]
        
        return top_predictions
    except Exception as e:
        logger.error(f"Error aggregating predictions: {str(e)}")
        return None

def visualize_predictions(image, patches, coordinates, probabilities, uncertainties, output_path):
    """Visualize predictions on the image."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(131)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Patches with predictions
        plt.subplot(132)
        plt.imshow(image)
        for (x, y), prob in zip(coordinates, probabilities):
            color = 'red' if prob[1] > CONFIDENCE_THRESHOLD else 'green'
            rect = plt.Rectangle((x, y), PATCH_SIZE, PATCH_SIZE,
                               fill=False, edgecolor=color, linewidth=1)
            plt.gca().add_patch(rect)
        plt.title('Patch Predictions')
        plt.axis('off')
        
        # Heatmap with uncertainty
        plt.subplot(133)
        heatmap = np.zeros(image.shape[:2])
        uncertainty_map = np.zeros(image.shape[:2])
        for (x, y), prob, unc in zip(coordinates, probabilities, uncertainties if uncertainties is not None else [None] * len(probabilities)):
            heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob[1]
            if unc is not None:
                uncertainty_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += unc[1]
        
        plt.imshow(heatmap, cmap='hot')
        if uncertainties is not None:
            plt.colorbar(label='Prediction Confidence')
            plt.title('Prediction Heatmap with Uncertainty')
        else:
            plt.colorbar(label='Prediction Confidence')
            plt.title('Prediction Heatmap')
        plt.axis('off')
        
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        logger.error(f"Error visualizing predictions: {str(e)}")

def predict_image(image_path, model, segmenter, patch_extractor, device, use_uncertainty=False, num_samples=10):
    """Make predictions on a single image."""
    try:
        # Preprocess image
        patches, coordinates, patch_metrics = preprocess_image(image_path, segmenter, patch_extractor)
        if patches is None:
            return None
            
        # Make predictions
        probabilities, uncertainties = predict_patches(model, patches, device, use_uncertainty, num_samples)
        if probabilities is None:
            return None
            
        # Aggregate predictions
        predictions = aggregate_predictions(probabilities, uncertainties, coordinates, patch_metrics)
        if predictions is None:
            return None
            
        # Prepare result dictionary with all necessary data
        result = {
            'image_path': str(image_path),
            'predictions': predictions,
            'patches': patches.tolist() if isinstance(patches, np.ndarray) else patches,
            'coordinates': coordinates.tolist() if isinstance(coordinates, np.ndarray) else coordinates,
            'patch_metrics': patch_metrics,
            'probabilities': probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities,
            'uncertainties': uncertainties.tolist() if isinstance(uncertainties, np.ndarray) else uncertainties
        }
        
        return result
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'prediction.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting BCC prediction pipeline...")
        
        # Check if model exists
        if not os.path.exists(args.model_path):
            logger.error(f"Model not found at {args.model_path}")
            return
        
        # Check if image exists
        if not os.path.exists(args.image_path):
            logger.error(f"Image not found at {args.image_path}")
            return
        
        logger.info(f"Loading image: {args.image_path}")
        image = cv2.imread(args.image_path)
        if image is None:
            logger.error(f"Failed to load image: {args.image_path}")
            return
        
        logger.info("Initializing pipeline components...")
        segmenter = TissueSegmenter(
            min_tissue_area=1000,
            kernel_size=5
        )
        
        patch_extractor = PatchExtractor(
            patch_size=224,
            stride=112,
            min_tissue_ratio=0.7,
            n_jobs=4
        )
        
        logger.info(f"Loading model from {args.model_path}")
        model = load_model(args.model_path)
        
        logger.info("Segmenting tissue...")
        mask = segmenter.segment_tissue(image)
        
        logger.info("Extracting patches...")
        patches, coordinates, patch_metrics = patch_extractor.extract_patches(image, mask)
        logger.info(f"Extracted {len(patches)} patches")
        
        # Save intermediate results
        intermediate_results = {
            'patches': patches,
            'coordinates': coordinates,
            'patch_metrics': patch_metrics
        }
        np.save(os.path.join(args.output_dir, 'intermediate_results.npy'), intermediate_results)
        
        # Make predictions
        predictions = []
        uncertainties = []
        
        # Process patches in batches
        for i in range(0, len(patches), args.batch_size):
            batch_patches = patches[i:i + args.batch_size]
            batch_predictions = model.predict(np.array(batch_patches))
            predictions.extend(batch_predictions)
            
            if args.use_uncertainty:
                batch_uncertainties = model.predict(np.array(batch_patches), return_uncertainty=True)
                uncertainties.extend(batch_uncertainties)
        
        # Save results
        results = {
            'image_path': args.image_path,
            'predictions': predictions,
            'uncertainties': uncertainties if args.use_uncertainty else None,
            'coordinates': coordinates,
            'patch_metrics': patch_metrics
        }
        
        with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        if args.visualize:
            logger.info("Generating visualizations...")
            visualize_results(image, mask, patches, coordinates, predictions, uncertainties, args.output_dir)
        
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return

if __name__ == "__main__":
    main() 