import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf

def calculate_metrics(model: tf.keras.Model,
                    X: np.ndarray,
                    y_true: np.ndarray) -> Dict:
    """
    Calculate classification metrics
    
    Args:
        model: Trained model
        X: Input features
        y_true: True labels (one-hot encoded)
        
    Returns:
        Dictionary of metrics
    """
    # Get model predictions
    y_pred = model.predict(X)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    metrics = {
        'accuracy': accuracy_score(y_true_classes, y_pred_classes),
        'precision': precision_score(y_true_classes, y_pred_classes, average='weighted'),
        'recall': recall_score(y_true_classes, y_pred_classes, average='weighted'),
        'f1': f1_score(y_true_classes, y_pred_classes, average='weighted'),
        'roc_auc': roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovo')
    }
    return metrics

def plot_confusion_matrix(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        output_path: str) -> None:
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_roc_curve(y_true: np.ndarray,
                  y_prob: np.ndarray,
                  output_path: str) -> None:
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        output_path: Path to save plot
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def evaluate_patch_predictions(patch_predictions: List[Dict],
                            patch_labels: List[int]) -> Dict:
    """
    Evaluate patch-level predictions
    
    Args:
        patch_predictions: List of patch predictions
        patch_labels: List of patch labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_true = np.array(patch_labels)
    y_pred = np.array([p['prediction'] for p in patch_predictions])
    y_prob = np.array([p['confidence'] for p in patch_predictions])
    
    return calculate_metrics(y_true, y_pred, y_prob)

def evaluate_image_predictions(image_predictions: List[Dict],
                            image_labels: List[int]) -> Dict:
    """
    Evaluate image-level predictions
    
    Args:
        image_predictions: List of image predictions
        image_labels: List of image labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_true = np.array(image_labels)
    y_pred = np.array([p['prediction'] for p in image_predictions])
    y_prob = np.array([p['confidence'] for p in image_predictions])
    
    return calculate_metrics(y_true, y_pred, y_prob)

def generate_evaluation_report(patch_metrics: Dict,
                            image_metrics: Dict,
                            output_path: str) -> None:
    """
    Generate evaluation report
    
    Args:
        patch_metrics: Patch-level metrics
        image_metrics: Image-level metrics
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write("BCC Detection System Evaluation Report\n")
        f.write("=====================================\n\n")
        
        f.write("Patch-Level Performance\n")
        f.write("----------------------\n")
        for metric, value in patch_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\nImage-Level Performance\n")
        f.write("----------------------\n")
        for metric, value in image_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

def plot_training_curves(history: Dict,
                        output_dir: str = 'data/visualizations/training') -> None:
    """
    Plot training curves from model history
    
    Args:
        history: Training history dictionary containing metrics
        output_dir: Directory to save the plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/loss_curves.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/accuracy_curves.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot other metrics if available
    for metric in ['precision', 'recall', 'f1', 'auc']:
        if metric in history:
            plt.figure(figsize=(10, 6))
            plt.plot(history[metric], label=f'Training {metric.upper()}')
            if f'val_{metric}' in history:
                plt.plot(history[f'val_{metric}'], label=f'Validation {metric.upper()}')
            plt.title(f'Model {metric.upper()} Over Time')
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{output_dir}/{metric}_curves.png', bbox_inches='tight', dpi=300)
            plt.close()