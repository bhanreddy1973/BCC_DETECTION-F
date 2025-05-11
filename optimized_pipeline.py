import os
import numpy as np
import cv2
import logging
from pathlib import Path
import tifffile
import json
import time
from datetime import datetime
import gc
from tqdm import tqdm
from src.preprocessing.tissue_segmentation import TissueSegmenter
from src.preprocessing.patch_extraction import PatchExtractor

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Update paths
BASE_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def validate_and_save_image(img: np.ndarray, save_path: Path) -> bool:
    """Validate image before saving and ensure it's in the correct format."""
    try:
        if img is None:
            return False
        
        # Ensure image is in uint8 format
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        
        # Ensure image is 2D for masks
        if len(img.shape) > 2 and img.shape[2] > 1:
            img = img[:, :, 0]
        
        # Use tifffile for saving large images
        if img.size > 100000000:  # If image is larger than 100MP
            tifffile.imwrite(str(save_path.with_suffix('.tif')), img)
            return True
        else:
            # Try to save with OpenCV for smaller images
            cv2.imwrite(str(save_path), img)
            
            # Verify the saved image can be read back
            verification = cv2.imread(str(save_path))
            if verification is None:
                logger.error(f"Failed to verify saved image: {save_path}")
                if save_path.exists():
                    save_path.unlink()
                return False
            
        return True
    except Exception as e:
        logger.error(f"Error saving image {save_path}: {str(e)}")
        if save_path.exists():
            save_path.unlink()
        return False

def process_image_in_chunks(image_path: Path, output_dir: Path, segmenter: TissueSegmenter, patch_extractor: PatchExtractor):
    """Process a large TIFF image in chunks to manage memory."""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Check if already processed
        metadata_path = output_dir / "patches" / f"{image_path.stem}_metadata.json"
        if metadata_path.exists():
            logger.info(f"Skipping {image_path.name} - already processed")
            return True
            
        # Create output directories
        patches_dir = output_dir / "patches"
        segmented_dir = output_dir / "segmented"
        patches_dir.mkdir(parents=True, exist_ok=True)
        segmented_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if output directories are writable
        if not os.access(str(patches_dir), os.W_OK) or not os.access(str(segmented_dir), os.W_OK):
            raise PermissionError(f"No write permission for output directories")
        
        # Use tifffile to read large TIFF images
        with tifffile.TiffFile(image_path) as tif:
            image = tif.asarray()
            
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Process in smaller chunks
        chunk_size = (1024, 1024)  # Adjust based on available memory
        height, width = image.shape[:2]
        all_patches = []
        all_coords = []
        all_metrics = []
        
        # Perform tissue segmentation on downsampled image for memory efficiency
        scale = 0.25  # 1/4 resolution for initial segmentation
        small_image = cv2.resize(image, None, fx=scale, fy=scale)
        mask, _ = segmenter.segment_tissue(small_image)
        mask = cv2.resize(mask.astype(np.uint8), (width, height)) > 0
        
        # Save segmentation mask with validation
        mask_path = segmented_dir / f"{image_path.stem}_mask.png"
        if not validate_and_save_image(mask.astype(np.uint8) * 255, mask_path):
            raise ValueError(f"Failed to save valid mask for {image_path.name}")
        
        # Process image in chunks where tissue is present
        total_tissue_pixels = np.sum(mask)
        if total_tissue_pixels == 0:
            logger.warning(f"No tissue detected in {image_path.name}")
            return False
            
        for y in range(0, height, chunk_size[0]):
            for x in range(0, width, chunk_size[1]):
                # Extract chunk coordinates
                y2 = min(y + chunk_size[0], height)
                x2 = min(x + chunk_size[1], width)
                
                # Check if chunk contains tissue
                if not mask[y:y2, x:x2].any():
                    continue
                
                # Process chunk
                chunk = image[y:y2, x:x2]
                chunk_mask = mask[y:y2, x:x2]
                
                # Extract patches from chunk
                patches, coordinates, metrics = patch_extractor.extract_patches(chunk, chunk_mask)
                
                if patches is not None and len(patches) > 0:
                    # Adjust coordinates to full image space
                    coordinates = [(coord[0] + x, coord[1] + y) for coord in coordinates]
                    
                    all_patches.extend(patches)
                    all_coords.extend(coordinates)
                    all_metrics.extend(metrics)
                
                # Clear chunk from memory
                del chunk, patches, coordinates, metrics
                gc.collect()
        
        if len(all_patches) == 0:
            logger.warning(f"No valid patches extracted from {image_path.name}")
            return False
            
        # Convert lists to numpy arrays
        all_patches = np.array(all_patches)
        all_coords = np.array(all_coords)
        
        # Save patches in batches
        batch_size = 1000
        num_batches = (len(all_patches) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_patches))
            
            batch_patches = all_patches[start_idx:end_idx]
            batch_coords = all_coords[start_idx:end_idx]
            batch_metrics = all_metrics[start_idx:end_idx]
            
            # Save batch with validation
            patches_path = patches_dir / f"{image_path.stem}_patches_{i}.npy"
            coords_path = patches_dir / f"{image_path.stem}_coords_{i}.npy"
            metrics_path = patches_dir / f"{image_path.stem}_metrics_{i}.json"
            
            try:
                np.save(str(patches_path), batch_patches)
                np.save(str(coords_path), batch_coords)
                with open(metrics_path, 'w') as f:
                    json.dump(batch_metrics, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving batch {i}: {str(e)}")
                # Clean up failed batch
                for path in [patches_path, coords_path, metrics_path]:
                    if path.exists():
                        path.unlink()
                continue
        
        # Save metadata
        metadata = {
            'image_name': image_path.name,
            'num_patches': len(all_patches),
            'num_batches': num_batches,
            'patch_size': patch_extractor.patch_size,
            'timestamp': datetime.now().isoformat(),
            'processing_info': {
                'chunk_size': chunk_size,
                'initial_segmentation_scale': scale,
                'total_tissue_area': int(np.sum(mask)),
                'total_patches': len(all_patches)
            }
        }
        
        metadata_path = patches_dir / f"{image_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully processed {image_path.name} - extracted {len(all_patches)} patches")
        return True
            
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
        # Clean up any partial results
        try:
            for path in patches_dir.glob(f"{image_path.stem}*"):
                path.unlink()
            for path in segmented_dir.glob(f"{image_path.stem}*"):
                path.unlink()
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up after failed processing: {str(cleanup_error)}")
        return False
    finally:
        # Clear memory
        gc.collect()

def preprocess_dataset(data_dir: Path, output_dir: Path):
    """Preprocess all images in the dataset."""
    logger.info("=== Starting Preprocessing Pipeline ===")
    
    # Initialize preprocessing components
    segmenter = TissueSegmenter(
        otsu_threshold=None,
        min_tissue_area=1000,
        kernel_size=5
    )
    
    patch_extractor = PatchExtractor(
        patch_size=224,
        stride=112,  # 50% overlap
        min_tissue_ratio=0.7
    )
    
    # Process images sequentially to manage memory
    successful = 0
    failed = 0
    total_images = 0
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        for class_name in ["bcc_high_risk", "bcc_low_risk", "non_malignant"]:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            image_paths = list(class_dir.glob("*.tif"))
            total_images += len(image_paths)
            
            output_class_dir = output_dir / split / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing {split}/{class_name} images...")
            for image_path in tqdm(image_paths, desc=f"{split}/{class_name}"):
                if process_image_in_chunks(image_path, output_class_dir, segmenter, patch_extractor):
                    successful += 1
                else:
                    failed += 1
                    
                # Clear memory between images
                gc.collect()
    
    # Generate summary
    summary = {
        'total_images': total_images,
        'successful': successful,
        'failed': failed,
        'success_rate': (successful / total_images * 100) if total_images > 0 else 0
    }
    
    with open(output_dir / 'preprocessing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=== Processing Summary ===")
    logger.info(f"Total images: {total_images}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed to process: {failed}")
    logger.info(f"Success rate: {summary['success_rate']:.2f}%")

if __name__ == "__main__":
    try:
        logger.info("Starting BCC Detection preprocessing pipeline...")
        preprocess_dataset(BASE_DIR, PROCESSED_DIR)
    except Exception as e:
        logger.error("Pipeline failed with error", exc_info=True)
    finally:
        logger.info("Pipeline execution completed")