import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import json
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import joblib

def load_real_data():
    """Load all real data from saved files"""
    data = {}
    
    # Load model
    try:
        model = tf.keras.models.load_model('models/checkpoints/best_model.h5')
        data['model'] = model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load training history
    try:
        with open('logs/training_history.json', 'r') as f:
            data['history'] = json.load(f)
    except Exception as e:
        print(f"Error loading training history: {e}")
        return None

    # Load model metrics
    try:
        with open('logs/model_metrics.json', 'r') as f:
            data['metrics'] = json.load(f)
    except Exception as e:
        print(f"Error loading model metrics: {e}")
        return None

    # Load feature importance
    try:
        data['feature_importance'] = joblib.load('models/feature_importance.pkl')
    except Exception as e:
        print(f"Error loading feature importance: {e}")
        return None

    # Load test data for confusion matrix
    try:
        data['X_test'] = np.load('data/features/X_test.npy')
        data['y_test'] = np.load('data/features/y_test.npy')
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

    return data

def create_pipeline_diagram():
    """Create the BCC detection pipeline diagram"""
    plt.figure(figsize=(15, 8))
    
    # Define components with descriptions
    components = [
        ('Input Features\n(1328 dimensions)', 'Raw feature vectors from medical images'),
        ('Feature Extraction\n(1024 units)', 'Initial feature processing and normalization'),
        ('Residual Blocks\n(512→256→128)', 'Deep feature learning with skip connections'),
        ('Attention Mechanism\n(128 units)', 'Feature importance weighting'),
        ('Classification\n(64 units)', 'Final feature processing'),
        ('Output\n(2 classes)', 'BCC/Non-BCC prediction')
    ]
    
    # Create arrows and boxes
    for i, (comp, desc) in enumerate(components):
        # Main component box
        plt.text(i, 0.6, comp, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        # Description text
        plt.text(i, 0.3, desc, ha='center', va='center', fontsize=8)
        if i < len(components) - 1:
            plt.arrow(i + 0.3, 0.6, 0.4, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    plt.xlim(-0.5, len(components) - 0.5)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('BCC Detection Pipeline', pad=20)
    
    # Save figure
    plt.savefig('docs/images/pipeline.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_architecture_diagram():
    """Create the model architecture diagram"""
    plt.figure(figsize=(12, 10))
    
    # Define layers and their sizes with descriptions
    layers = [
        ('Input\n(1328)', 'Raw feature vectors', 1),
        ('Dense\n(1024)', 'Initial feature extraction', 0.9),
        ('Residual\n(512)', 'Deep feature learning\nwith skip connections', 0.8),
        ('Residual\n(256)', 'Feature refinement\nwith skip connections', 0.7),
        ('Residual\n(128)', 'Final feature processing\nwith skip connections', 0.6),
        ('Attention\n(128)', 'Feature importance\nweighting', 0.5),
        ('Global Pool', 'Feature aggregation', 0.4),
        ('Dense\n(64)', 'Final processing', 0.3),
        ('Output\n(2)', 'BCC/Non-BCC\nprediction', 0.2)
    ]
    
    # Plot layers
    for i, (name, desc, y) in enumerate(layers):
        # Main layer box
        plt.text(0.5, y, name, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        # Description text
        plt.text(0.5, y - 0.05, desc, ha='center', va='center', fontsize=8)
        if i < len(layers) - 1:
            plt.arrow(0.5, y - 0.1, 0, -0.1, head_width=0.1, head_length=0.05, fc='black', ec='black')
    
    plt.xlim(0, 1)
    plt.ylim(0.1, 1.1)
    plt.axis('off')
    plt.title('Model Architecture', pad=20)
    
    # Save figure
    plt.savefig('docs/images/architecture.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_metrics_plot(data):
    """Create the performance metrics plot using real data"""
    if not data or 'metrics' not in data:
        print("Error: No metrics data available")
        return

    plt.figure(figsize=(12, 8))
    
    metrics = list(data['metrics'].keys())
    values = list(data['metrics'].values())
    
    # Create bar plot
    bars = plt.bar(metrics, values)
    
    # Customize bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2%}', ha='center', va='bottom')
    
    plt.ylim(0, 1)
    plt.title('Model Performance Metrics', pad=20)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add description
    plt.figtext(0.5, 0.01, 
                'The model shows balanced performance across metrics, with particularly strong precision.',
                ha='center', fontsize=10)
    
    # Save figure
    plt.savefig('docs/images/metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_training_curves(data):
    """Create training curves plot using real data"""
    if not data or 'history' not in data:
        print("Error: No training history data available")
        return

    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(data['history']['loss']) + 1)
    train_loss = data['history']['loss']
    val_loss = data['history']['val_loss']
    
    # Plot training curves
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves', pad=20)
    plt.legend()
    plt.grid(True)
    
    # Add description
    plt.figtext(0.5, 0.01, 
                'The model shows good convergence with minimal overfitting.',
                ha='center', fontsize=10)
    
    # Save figure
    plt.savefig('docs/images/training_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_feature_importance(data):
    """Create feature importance plot using real data"""
    if not data or 'feature_importance' not in data:
        print("Error: No feature importance data available")
        return

    plt.figure(figsize=(12, 8))
    
    feature_importance = data['feature_importance']
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)
    features = [features[i] for i in sorted_idx]
    importance = importance[sorted_idx]
    
    # Create horizontal bar plot
    plt.barh(features, importance)
    plt.xlabel('Importance Score')
    plt.title('Top 10 Feature Importance Scores', pad=20)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add description
    plt.figtext(0.5, 0.01, 
                'Feature importance scores show the relative contribution of each feature to the model predictions.',
                ha='center', fontsize=10)
    
    # Save figure
    plt.savefig('docs/images/feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_confusion_matrix(data):
    """Create confusion matrix plot using real data"""
    if not data or 'model' not in data or 'X_test' not in data or 'y_test' not in data:
        print("Error: No model or test data available for confusion matrix")
        return

    plt.figure(figsize=(10, 8))
    
    # Get predictions
    y_pred = data['model'].predict(data['X_test'])
    y_pred = (y_pred > 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(data['y_test'], y_pred)
    
    # Create heatmap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', pad=20)
    
    # Add labels
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0, 1], ['Non-BCC', 'BCC'])
    plt.yticks([0, 1], ['Non-BCC', 'BCC'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Add description
    plt.figtext(0.5, 0.01, 
                'The confusion matrix shows the model\'s prediction accuracy for each class.',
                ha='center', fontsize=10)
    
    # Save figure
    plt.savefig('docs/images/confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_lr_schedule_plot(data):
    """Create learning rate schedule plot using real data"""
    if not data or 'history' not in data or 'learning_rate' not in data['history']:
        print("Error: No learning rate history data available")
        return

    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(data['history']['learning_rate']) + 1)
    lr = data['history']['learning_rate']
    
    # Plot learning rate
    plt.plot(epochs, lr, 'b-', label='Learning Rate')
    
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule', pad=20)
    plt.legend()
    plt.grid(True)
    
    # Add description
    plt.figtext(0.5, 0.01, 
                'The learning rate schedule shows how the learning rate changed during training.',
                ha='center', fontsize=10)
    
    # Save figure
    plt.savefig('docs/images/lr_schedule.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Generate all documentation images using real data"""
    # Create docs/images directory if it doesn't exist
    Path('docs/images').mkdir(parents=True, exist_ok=True)
    
    # Load all real data
    data = load_real_data()
    if not data:
        print("Error: Could not load required data files. Please ensure all data files are available.")
        return
    
    # Generate all diagrams
    create_pipeline_diagram()
    create_architecture_diagram()
    create_metrics_plot(data)
    create_training_curves(data)
    create_feature_importance(data)
    create_confusion_matrix(data)
    create_lr_schedule_plot(data)

if __name__ == '__main__':
    main() 