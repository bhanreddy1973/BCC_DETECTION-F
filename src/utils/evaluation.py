import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray) -> Dict:
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
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