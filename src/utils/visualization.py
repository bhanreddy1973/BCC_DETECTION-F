import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import cv2
import seaborn as sns
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from typing import Optional
import logging
import json
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def visualize_patches(image: np.ndarray,
                     patches: List[np.ndarray],
                     coordinates: List[Tuple[int, int]],
                     output_path: str) -> None:
    """
    Visualize extracted patches on original image
    
    Args:
        image: Original image
        patches: List of patches
        coordinates: List of patch coordinates
        output_path: Path to save visualization
    """
    # Create RGB image if grayscale
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Convert to uint8 for visualization
    image = (image * 255).astype(np.uint8)
    
    # Draw patch boundaries
    for patch, (y, x) in zip(patches, coordinates):
        h, w = patch.shape[:2]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(f"Extracted Patches ({len(patches)} total)")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_features(features: np.ndarray,
                      output_path: str) -> None:
    """
    Visualize feature vectors using PCA
    
    Args:
        features: Feature vectors
        output_path: Path to save visualization
    """
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
    plt.title("Feature Space Visualization (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_predictions(image: np.ndarray,
                        predictions: Dict,
                        output_path: str) -> None:
    """
    Visualize model predictions on image
    
    Args:
        image: Original image
        predictions: Dictionary containing predictions
        output_path: Path to save visualization
    """
    # Create RGB image if grayscale
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    
    # Convert to uint8 for visualization
    image = (image * 255).astype(np.uint8)
    
    # Draw predicted regions
    for region in predictions['regions']:
        y, x = region['coordinates']
        h, w = region['size']
        confidence = region['confidence']
        
        # Color based on confidence
        color = (0, int(255 * confidence), 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Add confidence score
        text = f"{confidence:.2f}"
        cv2.putText(image, text, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(f"BCC Detection Results\nConfidence: {predictions['confidence']:.2f}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_training_history(history: Dict,
                        output_path: str) -> None:
    """
    Plot training history
    
    Args:
        history: Training history dictionary
        output_path: Path to save plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

class PipelineVisualizer:
    def __init__(self, output_dir: str, n_jobs: int = 48):
        """Initialize visualizer with output directory and parallel processing"""
        self.output_dir = Path(output_dir)
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)
        
        # Create visualization directories
        for subdir in ['preprocessing', 'features', 'training', 'predictions', 'reports']:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        # Set style for plots
        plt.style.use('seaborn')
        sns.set_theme(style="whitegrid")

    def visualize_preprocessing(self, 
                              original_image: np.ndarray,
                              segmentation_mask: np.ndarray,
                              patches: List[np.ndarray],
                              coordinates: List[Tuple[int, int]],
                              image_name: str) -> None:
        """Visualize preprocessing stage results"""
        fig = plt.figure(figsize=(20, 10))
        
        # Original image
        plt.subplot(231)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Segmentation mask
        plt.subplot(232)
        plt.imshow(segmentation_mask, cmap='gray')
        plt.title('Tissue Segmentation')
        plt.axis('off')
        
        # Patch coverage
        plt.subplot(233)
        coverage = np.zeros_like(segmentation_mask)
        for (x, y) in coordinates:
            coverage[y:y+224, x:x+224] += 1
        plt.imshow(coverage, cmap='hot')
        plt.title('Patch Coverage')
        plt.colorbar(label='Overlap count')
        plt.axis('off')
        
        # Sample patches
        if patches:
            for i, patch in enumerate(patches[:3], 1):
                plt.subplot(234 + i)
                plt.imshow(patch)
                plt.title(f'Sample Patch {i}')
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'preprocessing' / f'{image_name}_preprocessing.png')
        plt.close()

    def visualize_features(self,
                          deep_features: np.ndarray,
                          color_features: np.ndarray,
                          combined_features: np.ndarray,
                          image_name: str) -> None:
        """Visualize extracted features using dimensionality reduction"""
        # Create dimensionality reduction visualizations in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            
            # UMAP visualization
            futures.append(executor.submit(
                self._create_feature_embedding,
                combined_features,
                'umap',
                f'{image_name}_umap'
            ))
            
            # t-SNE visualization
            futures.append(executor.submit(
                self._create_feature_embedding,
                combined_features,
                'tsne',
                f'{image_name}_tsne'
            ))
            
            # Feature correlation heatmap
            futures.append(executor.submit(
                self._create_correlation_heatmap,
                combined_features,
                f'{image_name}_correlation'
            ))
            
            # Wait for all visualizations to complete
            for future in futures:
                future.result()

    def _create_feature_embedding(self,
                                features: np.ndarray,
                                method: str,
                                filename: str) -> None:
        """Create dimensionality reduction visualization"""
        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42)
            
        embedded = reducer.fit_transform(features)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(embedded[:, 0], embedded[:, 1], alpha=0.5)
        plt.title(f'Feature Embedding ({method.upper()})')
        plt.savefig(self.output_dir / 'features' / f'{filename}.png')
        plt.close()

    def _create_correlation_heatmap(self,
                                  features: np.ndarray,
                                  filename: str) -> None:
        """Create feature correlation heatmap"""
        corr = np.corrcoef(features.T)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig(self.output_dir / 'features' / f'{filename}.png')
        plt.close()

    def visualize_training_history(self,
                                 history: Dict[str, List[float]],
                                 metrics: List[str],
                                 image_name: str) -> None:
        """Visualize training metrics history"""
        n_metrics = len(metrics)
        fig = plt.figure(figsize=(15, 5 * ((n_metrics + 1) // 2)))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot((n_metrics + 1) // 2, 2, i)
            plt.plot(history[metric], label='Training')
            plt.plot(history[f'val_{metric}'], label='Validation')
            plt.title(f'{metric.capitalize()} Over Time')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training' / f'{image_name}_training.png')
        plt.close()
        
        # Create interactive plotly visualization
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Scatter(y=history[metric],
                                   name=f'Training {metric}',
                                   mode='lines'))
            fig.add_trace(go.Scatter(y=history[f'val_{metric}'],
                                   name=f'Validation {metric}',
                                   mode='lines'))
            
        fig.update_layout(title='Training History',
                         xaxis_title='Epoch',
                         yaxis_title='Metric Value')
        
        fig.write_html(self.output_dir / 'training' / f'{image_name}_training_interactive.html')

    def visualize_layer_activations(self,
                                  activations: np.ndarray,
                                  layer_name: str) -> None:
        """Visualize neural network layer activations"""
        n_neurons = min(16, activations.shape[-1])
        fig = plt.figure(figsize=(20, 4))
        
        for i in range(n_neurons):
            plt.subplot(2, 8, i+1)
            plt.hist(activations[:, i], bins=50)
            plt.title(f'Neuron {i+1}')
            
        plt.suptitle(f'Layer: {layer_name} Activations')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training' / f'{layer_name}_activations.png')
        plt.close()

    def visualize_predictions(self, image: np.ndarray,
                        predictions: Dict,
                        output_path: str) -> None:
        """
        Visualize model predictions on image
        
        Args:
            image: Original image
            predictions: Dictionary containing predictions
            output_path: Path to save visualization
        """
        # Create RGB image if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Convert to uint8 for visualization
        image = (image * 255).astype(np.uint8)
        
        # Draw predicted regions
        for region in predictions['regions']:
            y, x = region['coordinates']
            h, w = region['size']
            confidence = region['confidence']
            
            # Color based on confidence
            color = (0, int(255 * confidence), 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Add confidence score
            text = f"{confidence:.2f}"
            cv2.putText(image, text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(f"BCC Detection Results\nConfidence: {predictions['confidence']:.2f}")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_aggregation(self,
                            original_image: np.ndarray,
                            final_prediction_mask: np.ndarray,
                            uncertainty_mask: np.ndarray,
                            image_name: str) -> None:
        """Visualize aggregated predictions and uncertainty"""
        fig = plt.figure(figsize=(20, 10))
        
        # Original image with prediction overlay
        plt.subplot(131)
        plt.imshow(original_image)
        plt.imshow(final_prediction_mask, cmap='RdYlBu', alpha=0.5)
        plt.title('Predictions Overlay')
        plt.axis('off')
        
        # Prediction heatmap
        plt.subplot(132)
        plt.imshow(final_prediction_mask, cmap='RdYlBu')
        plt.colorbar(label='Prediction Confidence')
        plt.title('Prediction Heatmap')
        plt.axis('off')
        
        # Uncertainty map
        plt.subplot(133)
        plt.imshow(uncertainty_mask, cmap='viridis')
        plt.colorbar(label='Uncertainty Level')
        plt.title('Uncertainty Map')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'predictions' / f'{image_name}_predictions.png')
        plt.close()

    def create_summary_report(self,
                            metrics: Dict[str, float],
                            processing_time: Dict[str, float],
                            memory_usage: Dict[str, float],
                            image_name: str) -> None:
        """Create JSON summary report with metrics and resource usage"""
        report = {
            'image_name': image_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'processing_time': processing_time,
            'memory_usage': memory_usage
        }
        
        report_path = self.output_dir / 'reports' / f'{image_name}_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Create performance visualization
        fig = plt.figure(figsize=(15, 5))
        
        # Metrics plot
        plt.subplot(131)
        plt.bar(metrics.keys(), metrics.values())
        plt.xticks(rotation=45)
        plt.title('Performance Metrics')
        
        # Processing time plot
        plt.subplot(132)
        plt.bar(processing_time.keys(), processing_time.values())
        plt.xticks(rotation=45)
        plt.title('Processing Time (s)')
        
        # Memory usage plot
        plt.subplot(133)
        plt.bar(memory_usage.keys(), memory_usage.values())
        plt.xticks(rotation=45)
        plt.title('Memory Usage (MB)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reports' / f'{image_name}_performance.png')
        plt.close()

    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      image_name: str) -> None:
        """Plot ROC curve for model evaluation"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plt.savefig(self.output_dir / 'training' / f'{image_name}_roc.png')
        plt.close()

    @staticmethod
    def get_gradcam_heatmap(model: tf.keras.Model,
                           image: np.ndarray,
                           layer_name: str) -> np.ndarray:
        """Generate Grad-CAM heatmap for model interpretability"""
        grad_model = tf.keras.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(image)
            class_channel = predictions[:, 0]
            
        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def visualize_model_interpretability(self,
                                       model: tf.keras.Model,
                                       image: np.ndarray,
                                       layer_name: str,
                                       image_name: str) -> None:
        """Create model interpretability visualization using Grad-CAM"""
        heatmap = self.get_gradcam_heatmap(model, image, layer_name)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(superimposed)
        plt.title('Grad-CAM Visualization')
        plt.axis('off')
        
        plt.savefig(self.output_dir / 'classification' / f'{image_name}_interpretability.png')
        plt.close()