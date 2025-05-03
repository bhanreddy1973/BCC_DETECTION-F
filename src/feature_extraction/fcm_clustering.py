import numpy as np
from typing import Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class FCMClustering:
    """Fuzzy C-Means clustering for feature extraction"""
    n_clusters: int
    max_iter: int = 100
    m: float = 2.0  # Fuzziness parameter
    error: float = 1e-5
    
    def __post_init__(self):
        self.scaler = StandardScaler()
        
    def initialize_membership(self, n_samples: int) -> np.ndarray:
        """Initialize membership matrix"""
        membership = np.random.rand(n_samples, self.n_clusters)
        return membership / membership.sum(axis=1, keepdims=True)
    
    def update_centroids(self, X: np.ndarray, membership: np.ndarray) -> np.ndarray:
        """Update cluster centroids"""
        membership_m = membership ** self.m
        return (membership_m.T @ X) / membership_m.sum(axis=0)[:, np.newaxis]
    
    def update_membership(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Update membership matrix"""
        n_samples = X.shape[0]
        membership = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            if np.any(distances == 0):
                membership[i] = (distances == 0).astype(float)
            else:
                sum_inv = np.sum((distances[:, np.newaxis] / distances) ** (2 / (self.m - 1)), axis=1)
                membership[i] = 1 / sum_inv
                
        return membership
    
    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform FCM clustering
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (membership matrix, centroids)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize
        membership = self.initialize_membership(X.shape[0])
        prev_membership = np.zeros_like(membership)
        
        # Iterate until convergence
        for _ in range(self.max_iter):
            centroids = self.update_centroids(X_scaled, membership)
            membership = self.update_membership(X_scaled, centroids)
            
            # Check convergence
            if np.linalg.norm(membership - prev_membership) < self.error:
                break
                
            prev_membership = membership.copy()
            
        return membership, centroids
    
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features using FCM clustering
        
        Args:
            X: Input features
            
        Returns:
            Cluster membership features
        """
        membership, _ = self.fit_predict(X)
        
        # Compute additional features
        features = []
        features.append(membership)  # Cluster memberships
        features.append(np.max(membership, axis=1, keepdims=True))  # Max membership
        features.append(np.std(membership, axis=1, keepdims=True))  # Membership spread
        
        return np.hstack(features) 