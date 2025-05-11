import numpy as np
from typing import Dict, Optional, List
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import logging
from sklearn.utils import shuffle

from .model import BCCClassifier
from ..config import config

class StratifiedBatchGenerator(tf.keras.utils.Sequence):
    """Custom data generator for balanced batch sampling"""
    def __init__(self, X, y, batch_size=32):
        self.X = X.numpy() if isinstance(X, tf.Tensor) else X
        self.y = y.numpy() if isinstance(y, tf.Tensor) else y
        self.batch_size = batch_size
        
        # Convert one-hot encoded y back to class indices if needed
        if len(self.y.shape) > 1:
            self.y = np.argmax(self.y, axis=1)
        
        # Split data by class
        self.class_indices = [np.where(self.y == i)[0] for i in range(2)]
        self.samples_per_class = self.batch_size // 2
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = []
        
        # Sample equally from each class
        for class_idx in range(2):
            indices = np.random.choice(
                self.class_indices[class_idx],
                size=self.samples_per_class,
                replace=len(self.class_indices[class_idx]) < self.samples_per_class
            )
            batch_indices.extend(indices)
            
        # Shuffle the batch indices
        batch_indices = np.random.permutation(batch_indices)
        
        # Create the balanced batch
        batch_X = tf.convert_to_tensor(self.X[batch_indices], dtype=tf.float32)
        batch_y = tf.keras.utils.to_categorical(self.y[batch_indices], num_classes=2)
        batch_y = tf.cast(batch_y, tf.float32)  # Ensure float32 dtype
        
        return batch_X, batch_y
    
    def on_epoch_end(self):
        """Called at the end of every epoch"""
        pass  # We don't need to do anything here since we randomly sample each batch

def train_model(model: BCCClassifier,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None,
               callbacks: Optional[List] = None) -> Dict:
    """
    Train the BCC classifier with the specified training protocol
    
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
    
    # Convert inputs to float32
    X_train = tf.cast(X_train, tf.float32)
    if X_val is not None:
        X_val = tf.cast(X_val, tf.float32)
    
    # Convert labels to one-hot encoding for categorical crossentropy
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    if y_val is not None:
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)
    
    # Update learning rate schedule with actual training data size
    model.update_learning_rate_schedule(len(X_train))
    
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
            monitor='val_auc' if validation_data else 'auc',
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_auc' if validation_data else 'auc',
            patience=20,  # As specified in requirements
            restore_best_weights=True,
            mode='max'
        ),
        TensorBoard(
            log_dir=f"{config.training.checkpoint_dir}/logs",
            histogram_freq=1,
            update_freq='epoch'
        )
    ])
    
    # Create stratified batch generator for balanced training
    train_generator = StratifiedBatchGenerator(
        X_train, y_train,
        batch_size=config.training.batch_size
    )
    
    # Train model using the underlying TensorFlow model
    logger.info("Starting model training...")
    history = model.model.fit(
        train_generator,
        validation_data=validation_data,
        epochs=100,  # Maximum epochs as specified
        batch_size=config.training.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Training completed")
    return history.history