import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import cv2

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
    from sklearn.decomposition import PCA
    
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