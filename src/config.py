import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Union

@dataclass
class PreprocessingConfig:
    patch_size: int = 224
    patch_stride: int = 112  # 50% overlap
    min_tissue_ratio: float = 0.7
    batch_size: int = 64
    num_workers: int = 48  # Adjust based on available CPU cores

@dataclass
class ModelConfig:
    name: str = "efficientnet-b0"
    num_classes: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    dropout_rate: float = 0.3
    l2_reg: float = 1e-4

@dataclass
class TrainingConfig:
    batch_size: int = 32
    val_batch_size: int = 32
    test_batch_size: int = 32
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

@dataclass
class AugmentationConfig:
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = True
    fill_mode: str = 'nearest'

@dataclass
class Config:
    # Base directories
    base_dir: Path = Path("data")
    raw_dir: Path = base_dir / "raw"
    processed_dir: Path = base_dir / "processed"
    model_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    
    # Sub-configurations
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Inference parameters
    confidence_threshold: float = 0.5
    top_k_predictions: int = 5
    
    # Visualization parameters
    visualization_dir: Path = Path("visualizations")
    
    # GPU parameters
    cuda_visible_devices: str = "0"  # Set to -1 for CPU only
    mixed_precision: bool = True
    
    # Export parameters
    export_formats: List[str] = field(default_factory=lambda: ["onnx", "torchscript"])
    export_dir: Path = model_dir / "exported"
    
    # Class names
    class_names: List[str] = field(default_factory=lambda: ["normal", "bcc"])
    
    def __post_init__(self):
        # Create directories
        for dir_path in [self.base_dir, self.raw_dir, self.processed_dir, 
                        self.model_dir, self.logs_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

# Create global config instance
config = Config()

# Export commonly used variables for backward compatibility
PATCH_SIZE = config.preprocessing.patch_size
PATCH_STRIDE = config.preprocessing.patch_stride
MIN_TISSUE_RATIO = config.preprocessing.min_tissue_ratio
BATCH_SIZE = config.preprocessing.batch_size
NUM_WORKERS = config.preprocessing.num_workers
MODEL_NAME = config.model.name
NUM_CLASSES = config.model.num_classes
LEARNING_RATE = config.model.learning_rate
WEIGHT_DECAY = config.model.weight_decay
NUM_EPOCHS = config.model.num_epochs
EARLY_STOPPING_PATIENCE = config.model.early_stopping_patience
TRAIN_BATCH_SIZE = config.training.batch_size
VAL_BATCH_SIZE = config.training.val_batch_size
TEST_BATCH_SIZE = config.training.test_batch_size
AUGMENTATION = {
    'rotation_range': config.augmentation.rotation_range,
    'width_shift_range': config.augmentation.width_shift_range,
    'height_shift_range': config.augmentation.height_shift_range,
    'horizontal_flip': config.augmentation.horizontal_flip,
    'vertical_flip': config.augmentation.vertical_flip,
    'fill_mode': config.augmentation.fill_mode
}
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
CHECKPOINT_DIR = config.model_dir / "checkpoints"
BEST_MODEL_PATH = config.model_dir / "best_model.h5"
LAST_MODEL_PATH = config.model_dir / "last_model.h5"
CONFIDENCE_THRESHOLD = config.confidence_threshold
TOP_K_PREDICTIONS = config.top_k_predictions
VISUALIZATION_DIR = config.visualization_dir
CUDA_VISIBLE_DEVICES = config.cuda_visible_devices
MIXED_PRECISION = config.mixed_precision
EXPORT_FORMATS = config.export_formats
EXPORT_DIR = config.export_dir
TRAIN_SPLIT = config.training.train_split
VAL_SPLIT = config.training.val_split
TEST_SPLIT = config.training.test_split
CLASS_NAMES = config.class_names 