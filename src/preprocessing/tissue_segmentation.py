import os
import cv2
import numpy as np
import tifffile as tiff
import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TissueSegmenter:
    def __init__(self, 
                 otsu_threshold: Optional[int] = None,
                 min_tissue_area: int = 1000,
                 kernel_size: int = 5,
                 visualizer = None):
        """
        Initialize the tissue segmenter.
        
        Args:
            otsu_threshold: Optional manual threshold value. If None, Otsu's method will be used.
            min_tissue_area: Minimum area of tissue regions to keep (in pixels)
            kernel_size: Size of the morphological operation kernel
            visualizer: PipelineVisualizer instance for visualization
        """
        self.otsu_threshold = otsu_threshold
        self.min_tissue_area = min_tissue_area
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.visualizer = visualizer

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load a TIFF image with error handling."""
        try:
            image = tiff.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess the image for tissue segmentation."""
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        return blurred

    def segment_tissue(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Segment tissue regions from the image.
        
        Returns:
            Tuple containing:
            - Binary mask of tissue regions
            - Threshold value used
        """
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Apply Otsu's thresholding
        if self.otsu_threshold is None:
            threshold, binary = cv2.threshold(processed, 0, 255, 
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            threshold = self.otsu_threshold
            _, binary = cv2.threshold(processed, threshold, 255, cv2.THRESH_BINARY)
        
        # Invert the binary image (tissue should be white)
        binary = cv2.bitwise_not(binary)
        
        # Apply morphological operations
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_tissue_area:
                mask[labels == i] = 255
        
        return mask, threshold

    def process_image(self, image_path: str, output_dir: str, label: str, split: str) -> Optional[dict]:
        """Process a single image and save the tissue mask."""
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Segment tissue
        mask, threshold = self.segment_tissue(image)
        
        # Save mask
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_name = f"{base_name}_mask.tif"
        mask_path = os.path.join(output_dir, split, label, mask_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        
        # Save mask
        tiff.imwrite(mask_path, mask)
        
        # Calculate tissue ratio
        tissue_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        
        # Visualize if visualizer is available
        if self.visualizer:
            self.visualizer.visualize_preprocessing(
                original_image=image,
                segmentation_mask=mask,
                patches=[],  # Will be filled by patch extractor
                coordinates=[],
                image_name=f"{split}_{label}_{base_name}"
            )
        
        return {
            'original_image': base_name,
            'mask_name': mask_name,
            'label': label,
            'split': split,
            'threshold': threshold,
            'tissue_ratio': tissue_ratio
        }

def main():
    # Base directories
    base_dir = "data"
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    
    # Create segmenter instance
    segmenter = TissueSegmenter()
    
    # Get list of images
    image_paths = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(raw_dir, split)
        if os.path.exists(split_dir):
            for label in os.listdir(split_dir):
                label_dir = os.path.join(split_dir, label)
                if os.path.isdir(label_dir):
                    for image_file in os.listdir(label_dir):
                        if image_file.endswith('.tif'):
                            image_paths.append({
                                'path': os.path.join(label_dir, image_file),
                                'label': label,
                                'split': split
                            })
    
    # Process images
    mask_info = []
    for img_info in image_paths:
        logger.info(f"Processing {img_info['path']}")
        result = segmenter.process_image(
            img_info['path'],
            processed_dir,
            img_info['label'],
            img_info['split']
        )
        if result:
            mask_info.append(result)
    
    # Save mask information
    mask_df = pd.DataFrame(mask_info)
    mask_info_path = os.path.join(processed_dir, "mask_info.csv")
    mask_df.to_csv(mask_info_path, index=False)
    logger.info(f"Saved mask information to {mask_info_path}")
    
    # Print summary
    print("\nTissue Segmentation Summary:")
    for split in ['train', 'val', 'test']:
        split_masks = mask_df[mask_df['split'] == split]
        print(f"\n{split.capitalize()} Set:")
        for label in mask_df['label'].unique():
            count = len(split_masks[split_masks['label'] == label])
            avg_tissue_ratio = split_masks[split_masks['label'] == label]['tissue_ratio'].mean()
            print(f"  {label}: {count} masks")
            print(f"    Avg tissue ratio: {avg_tissue_ratio:.2f}")
        print(f"  Total: {len(split_masks)} masks")
    
    print(f"\nTotal masks created: {len(mask_df)}")
    print(f"Average tissue ratio across all masks: {mask_df['tissue_ratio'].mean():.2f}")

if __name__ == "__main__":
    main()
