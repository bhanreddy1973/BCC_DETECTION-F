import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import GridSearchCV, KFold
from typing import Tuple, Dict, List
import numpy as np
import logging

class BCCClassifier:
    def __init__(self,
                 input_dim: int,
                 learning_rate: float = 1e-4,
                 dropout_rate: float = 0.3,
                 batch_size: int = 32):
        """
        Initialize BCC classifier
        
        Args:
            input_dim: Input feature dimension
            learning_rate: Initial learning rate
            dropout_rate: Dropout rate
            batch_size: Training batch size
        """
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build the classification model
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Dense(512, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        return model
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              epochs: int = 100) -> Dict:
        """
        Train the classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'models/checkpoints/best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=self._get_class_weights(y_train)
        )
        
        return history.history
    
    def optimize_parameters(self,
                          X: np.ndarray,
                          y: np.ndarray) -> Dict:
        """
        Optimize model parameters using grid search with cross-validation
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary of optimal parameters
        """
        param_grid = {
            'learning_rate': [1e-3, 1e-4, 1e-5],
            'dropout_rate': [0.2, 0.3, 0.4],
            'batch_size': [16, 32, 64]
        }
        
        best_params = None
        best_score = -1
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for lr in param_grid['learning_rate']:
            for dropout in param_grid['dropout_rate']:
                for batch in param_grid['batch_size']:
                    self.learning_rate = lr
                    self.dropout_rate = dropout
                    self.batch_size = batch
                    
                    # Rebuild model with new parameters
                    self.model = self._build_model()
                    
                    # Cross-validation
                    fold_scores = []
                    for train_idx, val_idx in kf.split(X):
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        history = self.train(
                            X_train, y_train,
                            X_val, y_val,
                            epochs=50
                        )
                        
                        fold_scores.append(history['val_auc'][-1])
                    
                    mean_score = np.mean(fold_scores)
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'learning_rate': lr,
                            'dropout_rate': dropout,
                            'batch_size': batch
                        }
        
        return best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        return self.model.predict(X)
    
    def _get_class_weights(self, y: np.ndarray) -> Dict:
        """
        Calculate class weights for imbalanced data
        
        Args:
            y: Training labels
            
        Returns:
            Dictionary of class weights
        """
        class_counts = np.bincount(y.astype(int))
        total = np.sum(class_counts)
        weights = total / (len(class_counts) * class_counts)
        return {i: w for i, w in enumerate(weights)} 