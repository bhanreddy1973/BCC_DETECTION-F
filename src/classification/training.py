import numpy as np
from typing import Dict, Optional, List
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import logging

from .model import BCCClassifier
from ..config import config

def train_model(model: BCCClassifier,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None,
               callbacks: Optional[List] = None) -> Dict:
    """
    Train the BCC classifier
    
    Args:
        model: BCCClassifier instance
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        callbacks: Optional list of callbacks
        
    Returns:
        Training history
    """
    logger = logging.getLogger(__name__)
    
    # Setup validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # Setup callbacks
    if callbacks is None:
        callbacks = []
        
    # Add default callbacks
    callbacks.extend([
        ModelCheckpoint(
            filepath=f"{config.training.checkpoint_dir}/best_model.h5",
            monitor='val_loss' if validation_data else 'loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=config.classification.early_stopping_patience,
            restore_best_weights=True
        ),
        TensorBoard(
            log_dir=f"{config.training.checkpoint_dir}/logs",
            histogram_freq=1
        )
    ])
    
    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=config.classification.epochs,
        batch_size=config.classification.batch_size,
        callbacks=callbacks
    )
    
    logger.info("Training completed")
    return history.history 