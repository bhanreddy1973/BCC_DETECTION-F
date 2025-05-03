import argparse
import os
from pathlib import Path
import logging
import traceback
from typing import List
import random

from src.preprocessing.tissue_segmentation import TissueSegmenter
from src.config import config
from src.utils.data_handling import load_tiff_image, save_patches

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_image(image_path: str, output_dir: str, segmenter: TissueSegmenter) -> None:
    """
    Process a single image
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save processed data
        segmenter: Tissue segmenter instance
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load image
        logger.info(f"Loading image: {image_path}")
        image = load_tiff_image(image_path)
        logger.info(f"Image loaded successfully. Shape: {image.shape}, dtype: {image.dtype}")
        
        # Segment tissue
        logger.info("Starting tissue segmentation")
        mask = segmenter.segment_tissue(image)
        logger.info(f"Tissue segmentation completed. Mask shape: {mask.shape}")
        
        # Skip if mask shape is too large
        if mask.shape[0] > 35000 or mask.shape[1] > 35000:
            logger.warning(f"Skipping image {image_path} due to large mask shape: {mask.shape}")
            return
        
        # Extract patches
        logger.info("Starting patch extraction")
        patches, coordinates = segmenter.extract_patches(image, mask)
        logger.info(f"Extracted {len(patches)} patches")
        
        # Save patches
        image_name = Path(image_path).stem
        logger.info(f"Saving patches for {image_name}")
        save_patches(patches, coordinates, output_dir, image_name)
        logger.info(f"Patches saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing {image_path}:")
        logger.error(traceback.format_exc())
        raise

def main():
    """Main function for preprocessing dataset"""
    parser = argparse.ArgumentParser(description='Preprocess BCC dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input TIFF images')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save processed data')
    parser.add_argument('--num_images', type=int, default=10,
                      help='Number of random images to process')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting preprocessing pipeline")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'segmented'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'patches'), exist_ok=True)
    logger.info(f"Created output directories in {args.output_dir}")
    
    # Initialize tissue segmenter
    logger.info("Initializing tissue segmenter")
    segmenter = TissueSegmenter(
        min_tissue_threshold=config.preprocessing.min_tissue_threshold,
        patch_size=config.preprocessing.patch_size,
        patch_overlap=config.preprocessing.patch_overlap,
        n_folds=config.preprocessing.n_folds
    )
    
    # Get list of all images and select random subset
    all_image_paths = list(Path(args.input_dir).glob('*.tif'))
    logger.info(f"Found {len(all_image_paths)} total images")
    
    # Select random subset
    random.seed(42)  # For reproducibility
    selected_image_paths = random.sample(all_image_paths, min(args.num_images, len(all_image_paths)))
    logger.info(f"Selected {len(selected_image_paths)} random images to process")
    
    # Process selected images
    for i, image_path in enumerate(selected_image_paths, 1):
        logger.info(f"Processing image {i}/{len(selected_image_paths)}: {image_path}")
        process_image(str(image_path), args.output_dir, segmenter)
    
    logger.info("Preprocessing completed successfully")

if __name__ == '__main__':
    main() 