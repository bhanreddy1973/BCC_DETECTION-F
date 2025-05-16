# BCC (Basal Cell Carcinoma) Detection System

## Overview
This project implements a deep learning-based system for detecting Basal Cell Carcinoma (BCC) from medical images. The system uses a sophisticated neural network architecture with residual connections and attention mechanisms to achieve high accuracy in BCC detection.

![BCC Detection Pipeline](docs/images/pipeline.png)

## Architecture

### Model Structure
The model follows a hierarchical architecture with multiple components:

1. **Feature Extraction**
   - Input layer processes 1328-dimensional feature vectors
   - Initial dense layer (1024 units) with batch normalization
   - Dropout for regularization
   - Purpose: Transform raw features into meaningful representations

2. **Residual Blocks**
   - Three blocks with decreasing dimensions (512 → 256 → 128)
   - Each block includes:
     - Dense layers with batch normalization
     - Skip connections for better gradient flow
     - Dropout for regularization
   - Purpose: Enable deep feature learning while maintaining gradient flow

3. **Attention Mechanism**
   - Feature importance weighting (128 units)
   - Context-aware processing
   - Purpose: Focus on relevant features for BCC detection

4. **Classification Head**
   - Global context processing
   - Final dense layer (64 units)
   - Output layer (2 units for BCC/Non-BCC)
   - Purpose: Make final classification decision

![Model Architecture](docs/images/architecture.png)

## Features

### 1. Advanced Architecture
- Residual connections for better gradient flow
- Attention mechanism for feature importance
- Multiple regularization techniques
- Batch normalization and dropout

### 2. Training Stability
- Learning rate warmup
- Cosine decay with restarts
- Early stopping
- Class weight balancing

### 3. Monitoring and Visualization
- Comprehensive metrics tracking
- Layer activation visualization
- Training history monitoring

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/bcc-detection.git
cd bcc-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```python
from src.classification.model import BCCClassifier

# Initialize model
model = BCCClassifier(
    input_dim=1328,  # Your feature dimension
    learning_rate=0.001,
    dropout_rate=0.3
)

# Train model
history = model.train(
    X_train=train_features,
    y_train=train_labels,
    X_val=val_features,
    y_val=val_labels
)
```

### Prediction
```python
# Make predictions
predictions = model.predict(
    X=test_features,
    return_confidence=True
)
```

## Model Performance

### Metrics
- Accuracy: 65.97%
- Precision: 72.28%
- Recall: 65.97%
- F1 Score: 66.95%
- ROC AUC: 71.43%

![Performance Metrics](docs/images/metrics.png)

### Training Progress
The model shows good convergence with minimal overfitting, as demonstrated by the training curves:

![Training Curves](docs/images/training_curves.png)

### Feature Importance
The model identifies key features that contribute most to BCC detection:

![Feature Importance](docs/images/feature_importance.png)

### Confusion Matrix
Detailed breakdown of model predictions:

![Confusion Matrix](docs/images/confusion_matrix.png)

## Project Structure
```
bcc-detection/
├── src/
│   ├── classification/
│   │   ├── model.py          # Model architecture and training
│   │   ├── training.py       # Training utilities
│   │   └── inference.py      # Prediction utilities
│   ├── config.py             # Configuration settings
│   └── utils/                # Utility functions
├── data/
│   ├── features/             # Extracted features
│   └── visualizations/       # Training visualizations
├── models/
│   └── checkpoints/          # Model checkpoints
├── logs/                     # Training logs
└── docs/
    └── images/              # Documentation images
```

## Key Components

### 1. ResidualBlock
```python
class ResidualBlock(layers.Layer):
    def __init__(self, units, dropout_rate, l2_reg, name=None):
        # Implementation details...
```
Purpose: Enables deep feature learning while maintaining gradient flow through skip connections.

### 2. AttentionBlock
```python
class AttentionBlock(layers.Layer):
    def __init__(self, units, name=None):
        # Implementation details...
```
Purpose: Weights features based on their importance for BCC detection.

## Training Process

### Learning Rate Schedule
1. **Warmup Phase**
   - 10% of total epochs
   - Linear increase from 0 to initial learning rate
   - Purpose: Stabilize initial training

2. **Cosine Decay with Restarts**
   - Periodic learning rate adjustments
   - Helps escape local minima
   - Purpose: Fine-tune model parameters

