import numpy as np
from typing import Optional
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@dataclass
class PCAReducer:
    """PCA-based dimensionality reduction"""
    n_components: Optional[int] = None
    variance_threshold: float = 0.99
    
    def __post_init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit PCA model
        
        Args:
            X: Input features
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize PCA
        if self.n_components is None:
            # Find number of components for desired variance
            temp_pca = PCA()
            temp_pca.fit(X_scaled)
            cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using PCA
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
            
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform features
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        self.fit(X)
        return self.transform(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores
        
        Returns:
            Array of importance scores
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
            
        return self.pca.explained_variance_ratio_
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform features
        
        Args:
            X: Transformed features
            
        Returns:
            Reconstructed features
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
            
        X_original_scaled = self.pca.inverse_transform(X)
        return self.scaler.inverse_transform(X_original_scaled) 