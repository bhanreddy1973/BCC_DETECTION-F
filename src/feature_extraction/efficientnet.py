import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Dict, List
import numpy as np
import logging
import gc

class EfficientNetFeatureExtractor:
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 pca_variance: float = 0.99,
                 batch_size: int = 32):
        """
        Initialize EfficientNet feature extractor
        
        Args:
            input_shape: Input image shape
            pca_variance: Variance to preserve in PCA
            batch_size: Batch size for feature extraction
        """
        self.input_shape = input_shape
        self.pca_variance = pca_variance
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize base model
        self.base_model = EfficientNetB7(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Add global average pooling
        self.model = Model(
            inputs=self.base_model.input,
            outputs=GlobalAveragePooling2D()(self.base_model.output)
        )
        
        # Initialize PCA
        self.pca = PCA(n_components=pca_variance)
        
    def extract_features(self, 
                        images: np.ndarray) -> np.ndarray:
        """
        Extract features from images using EfficientNet
        
        Args:
            images: Batch of input images
            
        Returns:
            Extracted features
        """
        try:
            # Calculate number of chunks
            n_images = len(images)
            chunk_size = min(1000, n_images)  # Process at most 1000 images at a time
            n_chunks = (n_images + chunk_size - 1) // chunk_size
            
            self.logger.info(f"Processing {n_images} images in {n_chunks} chunks of size {chunk_size}")
            
            all_features = []
            
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_images)
                
                self.logger.info(f"Processing chunk {i+1}/{n_chunks} (images {start_idx}-{end_idx})")
                
                # Get current chunk
                chunk = images[start_idx:end_idx]
                
                # Preprocess images
                preprocessed = tf.keras.applications.efficientnet.preprocess_input(chunk)
                
                # Extract features
                features = self.model.predict(preprocessed, batch_size=self.batch_size)
                all_features.append(features)
                
                # Clear memory
                del preprocessed
                del features
                gc.collect()
                tf.keras.backend.clear_session()
            
            # Combine all features
            all_features = np.vstack(all_features)
            
            # Apply PCA
            self.logger.info("Applying PCA to reduce dimensions")
            reduced_features = self.pca.fit_transform(all_features)
            
            return reduced_features
            
        except Exception as e:
            self.logger.error(f"Error in extract_features: {str(e)}")
            raise
    
    def optimize_parameters(self,
                          train_images: np.ndarray,
                          train_labels: np.ndarray) -> Dict:
        """
        Optimize feature extraction parameters using grid search
        
        Args:
            train_images: Training images
            train_labels: Training labels
            
        Returns:
            Dictionary of optimal parameters
        """
        param_grid = {
            'pca_variance': [0.95, 0.99, 0.999],
            'batch_size': [16, 32, 64]
        }
        
        best_params = None
        best_score = -1
        
        for pca_var in param_grid['pca_variance']:
            for batch in param_grid['batch_size']:
                self.pca_variance = pca_var
                self.batch_size = batch
                
                # Extract features
                features = self.extract_features(train_images)
                
                # Train simple classifier
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=1000)
                scores = cross_val_score(clf, features, train_labels, cv=5)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'pca_variance': pca_var,
                        'batch_size': batch
                    }
        
        return best_params
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from PCA components
        
        Returns:
            Feature importance scores
        """
        return self.pca.explained_variance_ratio_ 