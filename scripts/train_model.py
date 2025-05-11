import argparse
import os
import logging
from pathlib import Path
import json
import itertools
from typing import Dict, List, Tuple, Optional

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.classification.model import BCCClassifier
from src.config import config
from src.utils.optimization import configure_performance
from src.utils.visualization import PipelineVisualizer
from src.utils.evaluation import calculate_metrics, plot_training_curves

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_features(data_dir: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load features from either combined directory or individual files
    
    Args:
        data_dir: Base directory containing features
        
    Returns:
        Tuple of (features, labels) dictionaries for train/val/test splits
    """
    logger = logging.getLogger(__name__)
    features = {}
    labels = {}
    
    # Risk levels to process
    risk_levels = ['bcc_high_risk', 'bcc_low_risk', 'non_malignant']
    
    for split in ['train', 'val', 'test']:
        logger.info(f"Loading {split} features...")
        try:
            # First try loading from combined directory
            combined_features_path = data_dir / 'combined' / f'{split}_features.npy'
            combined_labels_path = data_dir / 'combined' / f'{split}_labels.npy'
            
            if combined_features_path.exists() and combined_labels_path.exists():
                logger.info(f"Loading combined features for {split} split")
                features[split] = np.load(combined_features_path)
                labels[split] = np.load(combined_labels_path)
                logger.info(f"{split} labels shape after loading: {labels[split].shape}")
                # One-hot encode labels
                labels[split] = tf.keras.utils.to_categorical(labels[split], num_classes=2)
                logger.info(f"{split} labels shape after one-hot encoding: {labels[split].shape}")
                logger.info(f"Loaded {len(features[split])} {split} samples "
                          f"with {features[split].shape[1]} dimensions")
                continue
            
            # If combined features not found, load individual features
            logger.info(f"Loading individual features for {split} split")
            split_features = []
            split_labels = []
            
            for risk_level in risk_levels:
                risk_dir = data_dir / split / risk_level
                if not risk_dir.exists():
                    logger.warning(f"Directory not found: {risk_dir}")
                    continue
                
                # Get all feature files for this risk level
                feature_files = list(risk_dir.glob('*_features.npy'))
                if not feature_files:
                    logger.warning(f"No feature files found in {risk_dir}")
                    continue
                
                # Load and combine features for each file
                for feature_file in feature_files:
                    try:
                        # Load features
                        file_features = np.load(feature_file)
                        
                        # Create labels based on risk level
                        file_labels = np.zeros(len(file_features))
                        if risk_level != 'non_malignant':
                            file_labels[:] = 1  # BCC class
                        
                        split_features.append(file_features)
                        split_labels.append(file_labels)
                        
                        logger.info(f"Loaded {len(file_features)} samples from {feature_file.name}")
                        
                    except Exception as e:
                        logger.error(f"Error loading {feature_file}: {e}")
                        continue
            
            if not split_features:
                logger.error(f"No features found for {split} split")
                features[split] = np.array([])
                labels[split] = np.array([])
                continue
            
            # Combine features from all risk levels
            features[split] = np.concatenate(split_features, axis=0)
            labels[split] = np.concatenate(split_labels, axis=0)
            logger.info(f"{split} labels shape after loading: {labels[split].shape}")
            # One-hot encode labels
            labels[split] = tf.keras.utils.to_categorical(labels[split], num_classes=2)
            logger.info(f"{split} labels shape after one-hot encoding: {labels[split].shape}")
            
            logger.info(f"Total {split} samples: {len(features[split])} "
                       f"with {features[split].shape[1]} dimensions")
            
        except Exception as e:
            logger.error(f"Error loading {split} data: {e}")
            features[split] = np.array([])
            labels[split] = np.array([])
            
    return features, labels

def create_callbacks(model_dir: Path) -> List[tf.keras.callbacks.Callback]:
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.model.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / 'best_model.h5'),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    return callbacks

def grid_search_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: Path,
    visualizer: Optional[PipelineVisualizer] = None
) -> Tuple[BCCClassifier, Dict, float]:
    """Perform grid search for hyperparameter tuning"""
    logger = logging.getLogger(__name__)
    
    # Define hyperparameter grid
    param_grid = {
        'dropout_rate': [0.3, 0.4, 0.5],
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'batch_size': [32, 64, 128],
        'l2_reg': [1e-4, 1e-5, 1e-6]
    }
    
    best_val_auc = 0
    best_params = None
    best_model = None
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in itertools.product(*param_grid.values())]
    
    logger.info(f"Starting grid search with {len(param_combinations)} combinations")
    
    for params in param_combinations:
        logger.info(f"Testing parameters: {params}")
        
        # Initialize model with current parameters
        model = BCCClassifier(
            input_dim=X_train.shape[1],
            dropout_rate=params['dropout_rate'],
            learning_rate=params['learning_rate'],
            l2_reg=params['l2_reg'],
            checkpoint_dir=model_dir,
            visualizer=visualizer
        )
        
        # Train model
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=params['batch_size'],
            callbacks=create_callbacks(model_dir)
        )
        
        # Get best validation AUC
        val_auc = max(history['val_auc'])
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_params = params
            best_model = model
            logger.info(f"New best parameters found! Val AUC: {val_auc:.4f}")
    
    return best_model, best_params, best_val_auc

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    model_dir: Path = None,
    visualizer: Optional[PipelineVisualizer] = None
) -> List[Dict]:
    """Perform k-fold cross-validation"""
    logger = logging.getLogger(__name__)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        logger.info(f"Training fold {fold}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create fold-specific directories
        fold_model_dir = model_dir / f"fold_{fold}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform grid search for this fold
        best_model, best_params, best_val_auc = grid_search_hyperparameters(
            X_train, y_train, X_val, y_val,
            fold_model_dir,
            visualizer
        )
        
        cv_scores.append({
            'fold': fold,
            'val_auc': best_val_auc,
            'params': best_params
        })
        
        logger.info(f"Fold {fold} completed with val AUC: {best_val_auc:.4f}")
    
    return cv_scores

def main():
    """Main function for training model"""
    parser = argparse.ArgumentParser(description='Train BCC detection model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Base directory containing features')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save trained model')
    parser.add_argument('--visualization_dir', type=str, default='data/visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--n_splits', type=int, default=5,
                      help='Number of cross-validation folds')
    args = parser.parse_args()
    
    # Convert paths to Path objects
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    visualization_dir = Path(args.visualization_dir)
    
    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(model_dir)
    
    # Configure performance
    configure_performance()
    
    # Initialize visualizer
    visualizer = PipelineVisualizer(visualization_dir)
    
    try:
        # Load features
        features, labels = load_features(data_dir)
        
        if len(features['train']) == 0 or len(labels['train']) == 0:
            raise ValueError("No training data found")
        
        # Combine train and val for cross-validation
        X = np.concatenate([features['train'], features['val']])
        y = np.concatenate([labels['train'], labels['val']])
        
        # Perform cross-validation
        cv_scores = cross_validate(
            X, y,
            n_splits=args.n_splits,
            model_dir=model_dir,
            visualizer=visualizer
        )
        
        # Save cross-validation results
        cv_results = {
            'cv_scores': cv_scores,
            'mean_val_auc': np.mean([score['val_auc'] for score in cv_scores]),
            'std_val_auc': np.std([score['val_auc'] for score in cv_scores])
        }
        
        with open(model_dir / 'cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=4)
        
        logger.info(f"Cross-validation completed. Mean val AUC: {cv_results['mean_val_auc']:.4f} ± {cv_results['std_val_auc']:.4f}")
        
        # Train final model on full training set
        logger.info("Training final model on full training set...")
        final_model = BCCClassifier(
            input_dim=X.shape[1],
            checkpoint_dir=model_dir,
            visualizer=visualizer
        )
        
        history = final_model.train(
            X_train=X,
            y_train=y,
            X_val=features['test'],
            y_val=labels['test'],
            callbacks=create_callbacks(model_dir)
        )
        
        # Save training history
        with open(model_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=4)
        
        # Calculate and save final metrics
        predictions = final_model.predict(features['test'])
        metrics = calculate_metrics(labels['test'], predictions)
        
        with open(model_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Final model metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()