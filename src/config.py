from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing stage"""
    min_tissue_threshold: float = 0.7
    patch_size: int = 224
    patch_overlap: float = 0.5
    n_folds: int = 5
    morph_operations: List[str] = field(default_factory=lambda: ["closing", "opening"])
    morph_kernel_size: int = 5
    min_object_size: int = 500

@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction stage"""
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    pca_variance: float = 0.99
    batch_size: int = 32
    color_spaces: List[str] = field(default_factory=lambda: ["HSV", "Lab", "YCbCr"])
    fcm_clusters: int = 3

@dataclass
class ClassificationConfig:
    """Configuration for classification stage"""
    input_dim: int = 512
    learning_rate: float = 1e-4
    dropout_rate: float = 0.3
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    dense_units: List[int] = field(default_factory=lambda: [512, 256, 128])

@dataclass
class AggregationConfig:
    """Configuration for aggregation stage"""
    confidence_threshold: float = 0.3
    min_cluster_size: int = 5
    eps: float = 0.1
    spatial_weight: float = 0.5

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    train_val_split: float = 0.8
    random_seed: int = 42
    num_workers: int = 4
    mixed_precision: bool = True
    checkpoint_dir: str = "models/checkpoints"
    model_dir: str = "models/final"

@dataclass
class DataConfig:
    """Configuration for data handling"""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    features_dir: str = "data/features"
    predictions_dir: str = "data/predictions"
    visualizations_dir: str = "data/visualizations"

@dataclass
class Config:
    """Main configuration class"""
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

# Create global config instance
config = Config() 