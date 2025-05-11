import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
from typing import Dict, Optional, List, Tuple
import logging
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from src.config import config

class ResidualBlock(layers.Layer):
    def __init__(self, units, dropout_rate, l2_reg, name=None):
        super(ResidualBlock, self).__init__(name=name)
        self.units = units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
        self.dense1 = layers.Dense(
            units,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            dtype=tf.float32
        )
        self.bn1 = layers.BatchNormalization(dtype=tf.float32)
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.dense2 = layers.Dense(
            units,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            dtype=tf.float32
        )
        self.bn2 = layers.BatchNormalization(dtype=tf.float32)
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Skip connection
        self.skip_dense = layers.Dense(
            units,
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
            dtype=tf.float32
        )
        
    def call(self, inputs, training=False):
        # Main path
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        skip = self.skip_dense(inputs)
        
        # Combine paths
        x = tf.nn.relu(x + skip)
        x = self.dropout2(x, training=training)
        
        return x

class AttentionBlock(layers.Layer):
    def __init__(self, units, name=None):
        super(AttentionBlock, self).__init__(name=name)
        self.units = units
        self.attention = layers.Dense(units, activation='tanh')
        self.context = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        # Compute attention weights
        attention_weights = self.attention(inputs)
        attention_weights = self.context(attention_weights)
        
        # Apply attention
        return inputs * attention_weights

class BCCClassifier:
    def __init__(self, 
                 input_dim: int,
                 learning_rate: float = None,
                 dropout_rate: float = None,
                 l2_reg: float = None,
                 batch_size: int = None,
                 checkpoint_dir: str = None,
                 visualizer = None):
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate or config.model.learning_rate
        self.dropout_rate = dropout_rate or config.model.dropout_rate
        self.l2_reg = l2_reg or config.model.weight_decay
        self.batch_size = batch_size or config.training.batch_size
        self.checkpoint_dir = Path(checkpoint_dir or config.model_dir / "checkpoints")
        self.visualizer = visualizer
        self.model = self._build_model()
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _build_model(self) -> tf.keras.Model:
        """Build the classification model with residual connections and attention"""
        # Configure memory growth for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        inputs = tf.keras.Input(shape=(self.input_dim,), dtype=tf.float32)
        
        # Initial feature extraction
        x = layers.Dense(1024, 
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                        dtype=tf.float32)(inputs)
        x = layers.BatchNormalization(dtype=tf.float32)(x)
        x = layers.ReLU(dtype=tf.float32)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Residual blocks with decreasing dimensions
        x = ResidualBlock(512, self.dropout_rate, self.l2_reg, name='res_block_1')(x)
        x = ResidualBlock(256, self.dropout_rate, self.l2_reg, name='res_block_2')(x)
        x = ResidualBlock(128, self.dropout_rate, self.l2_reg, name='res_block_3')(x)
        
        # Attention mechanism
        x = AttentionBlock(128, name='attention_block')(x)
        
        # Global context
        x = layers.GlobalAveragePooling1D()(x)
        
        # Final classification layers
        x = layers.Dense(64,
                        kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                        dtype=tf.float32)(x)
        x = layers.BatchNormalization(dtype=tf.float32)(x)
        x = layers.ReLU(dtype=tf.float32)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer with softmax activation for binary classification
        outputs = layers.Dense(config.model.num_classes, 
                             activation='softmax',
                             dtype=tf.float32)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Cosine decay learning rate schedule with warmup
        initial_learning_rate = self.initial_learning_rate
        decay_steps = config.model.num_epochs
        warmup_steps = int(decay_steps * 0.1)  # 10% warmup
        
        cosine_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps=decay_steps,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.001
        )
        
        warmup_schedule = tf.keras.optimizers.schedules.LinearSchedule(
            initial_learning_rate=0.0,
            final_learning_rate=initial_learning_rate,
            total_steps=warmup_steps
        )
        
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[warmup_steps],
            values=[warmup_schedule, cosine_decay]
        )
        
        optimizer = Adam(learning_rate=lr_schedule)
        
        # Compile with categorical crossentropy and metrics
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score', threshold=0.5)
            ]
        )
        
        return model
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = None,
             batch_size: int = None,
             callbacks: List[tf.keras.callbacks.Callback] = None) -> Dict:
        """
        Train the classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        epochs = epochs or config.model.num_epochs
        batch_size = batch_size or self.batch_size
        
        # Calculate class weights for imbalanced dataset
        class_weights = self._get_class_weights(y_train)
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                ModelCheckpoint(
                    str(self.checkpoint_dir / 'model_{epoch:02d}_{val_auc:.4f}.h5'),
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_auc',
                    patience=config.model.early_stopping_patience,
                    restore_best_weights=True,
                    mode='max',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # Add TensorBoard callback if visualization is enabled
        if self.visualizer:
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(self.checkpoint_dir / 'logs'),
                    histogram_freq=1,
                    update_freq='epoch'
                )
            )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Visualize training results if visualizer is available
        if self.visualizer:
            self.visualizer.visualize_training_history(
                history=history.history,
                metrics=['loss', 'accuracy', 'auc', 'precision', 'recall'],
                image_name='training_history'
            )
        
        return history.history
    
    def predict(self,
               X: np.ndarray,
               batch_size: Optional[int] = None,
               return_confidence: bool = False) -> np.ndarray:
        """Make predictions with optional confidence scores"""
        predictions = self.model.predict(X, batch_size=batch_size)
        
        if return_confidence:
            # Calculate confidence as distance from decision boundary
            confidence = np.abs(predictions - 0.5) * 2
            return predictions, confidence
        
        return predictions
    
    def save(self, filepath: str) -> None:
        """Save the model"""
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the model"""
        self.model = tf.keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        
    def visualize_layer_activations(self, 
                                  X: np.ndarray,
                                  layer_names: Optional[List[str]] = None) -> None:
        """Visualize layer activations if visualizer is available"""
        if not self.visualizer:
            return
            
        if layer_names is None:
            layer_names = [layer.name for layer in self.model.layers if isinstance(layer, Dense)]
        
        for layer_name in layer_names:
            activation_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer(layer_name).output
            )
            activations = activation_model.predict(X)
            
            self.visualizer.visualize_layer_activations(
                activations=activations,
                layer_name=layer_name
            )
    
    def _get_class_weights(self, y: np.ndarray) -> Dict:
        """Compute class weights for imbalanced datasets. Accepts one-hot or class index labels."""
        # If y is one-hot encoded, convert to class indices
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_indices),
            y=y_indices
        )
        return {i: w for i, w in enumerate(class_weights)}
    
    def update_learning_rate_schedule(self, train_size: int) -> None:
        """Update the learning rate schedule with actual training data size"""
        steps_per_epoch = train_size // self.batch_size
        decay_steps = config.training.epochs * steps_per_epoch
        
        # Update optimizer learning rate schedule
        self.model.optimizer.learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            self.initial_learning_rate, decay_steps
        )