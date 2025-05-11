import argparse
import os
from pathlib import Path
import logging
import traceback
from typing import List
import random
import psutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from src.preprocessing.tissue_segmentation import TissueSegmenter
from src.preprocessing.patch_extraction import PatchExtractor
from src.config import config
from src.utils.data_handling import load_tiff_image, save_patches
from src.utils.visualization import PipelineVisualizer

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'preprocessing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_optimal_batch_size(image_shape, available_memory):
    """Calculate optimal batch size based on image size and available memory"""
    # Estimate memory per image (assuming float32)
    image_memory = np.prod(image_shape) * 4  # 4 bytes per float32
    # Use 70% of available memory
    safe_memory = int(available_memory * 0.7)
    return max(1, safe_memory // image_memory)

def process_image(image_path: str, output_dir: str, segmenter: TissueSegmenter, extractor: PatchExtractor) -> None:
    """Process a single image through the pipeline"""
    logger = logging.getLogger(__name__)
    
    try:
        # Load image
        logger.info(f"Loading image: {image_path}")
        image = load_tiff_image(image_path)
        logger.info(f"Image loaded successfully. Shape: {image.shape}, dtype: {image.dtype}")
        
        # Get optimal batch size for this image
        available_memory = psutil.virtual_memory().available
        batch_size = get_optimal_batch_size(image.shape, available_memory)
        logger.info(f"Using batch size: {batch_size}")
        
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
        patches, coordinates = extractor.extract_patches(image, mask)
        logger.info(f"Extracted {len(patches)} patches")
        
        # Save patches with coordinates
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
    parser.add_argument('--num_workers', type=int, default=48,
                      help='Number of worker processes (default: 48)')
    parser.add_argument('--visualization_dir', type=str, default='data/visualizations',
                      help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting preprocessing pipeline")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'segmented'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'patches'), exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)
    logger.info(f"Created output directories in {args.output_dir}")
    
    # Initialize visualizer
    visualizer = PipelineVisualizer(args.visualization_dir, n_jobs=args.num_workers)
    
    # Initialize tissue segmenter with visualizer
    logger.info("Initializing tissue segmenter")
    segmenter = TissueSegmenter(
        min_tissue_threshold=config.preprocessing.min_tissue_threshold,
        patch_size=config.preprocessing.patch_size,
        patch_overlap=config.preprocessing.patch_overlap,
        n_folds=config.preprocessing.n_folds,
        visualizer=visualizer
    )
    
    # Initialize patch extractor with visualizer
    logger.info("Initializing patch extractor")
    extractor = PatchExtractor(
        patch_size=config.preprocessing.patch_size,
        stride=int(config.preprocessing.patch_size * (1 - config.preprocessing.patch_overlap)),
        min_tissue_ratio=config.preprocessing.min_tissue_threshold,
        n_jobs=args.num_workers,
        batch_size=64,
        visualizer=visualizer
    )
    
    # Get list of all images and select random subset
    all_image_paths = list(Path(args.input_dir).glob('*.tif'))
    logger.info(f"Found {len(all_image_paths)} total images")
    
    # Select random subset
    random.seed(42)  # For reproducibility
    selected_image_paths = random.sample(all_image_paths, min(args.num_images, len(all_image_paths)))
    logger.info(f"Selected {len(selected_image_paths)} random images to process")
    
    # Process selected images in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(process_image, str(path), args.output_dir, segmenter, extractor)
            for path in selected_image_paths
        ]
        
        # Wait for all tasks to complete
        for i, future in enumerate(futures, 1):
            try:
                future.result()
                logger.info(f"Completed processing image {i}/{len(selected_image_paths)}")
            except Exception as e:
                logger.error(f"Failed to process image {i}: {str(e)}")
    
    logger.info("Preprocessing completed successfully")

if __name__ == '__main__':
    main()