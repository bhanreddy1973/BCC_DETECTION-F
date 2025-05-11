import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import tifffile as tiff
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import gc
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PatchExtractor:
    def __init__(self, 
                 patch_size=224, 
                 stride=112,  # 50% overlap as per requirements
                 min_tissue_ratio=0.7,  # 70% tissue content requirement
                 n_jobs=48,  # Using half of available cores
                 batch_size=64,
                 visualizer=None):
        self.patch_size = patch_size
        self.stride = stride
        self.min_tissue_ratio = min_tissue_ratio
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.visualizer = visualizer
        self.logger = logging.getLogger(__name__)

    def load_image(self, image_path):
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

    def load_mask(self, mask_path):
        """Load a tissue mask with error handling."""
        try:
            mask = tiff.imread(mask_path)
            if mask is None:
                logger.error(f"Failed to load mask: {mask_path}")
                return None
            return mask > 0  # Convert to boolean mask
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {str(e)}")
            return None

    def extract_patches(self, image: np.ndarray, mask: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[Dict]]:
        """Extract patches using parallel processing for large images"""
        height, width = image.shape[:2]
        
        # Calculate grid positions for parallel processing
        positions = []
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                positions.append((x, y))

        # Split positions into batches for parallel processing
        position_batches = [positions[i:i + self.batch_size] 
                          for i in range(0, len(positions), self.batch_size)]

        patches = []
        coordinates = []
        patch_metrics = []

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for batch in position_batches:
                future = executor.submit(
                    self._process_patch_batch,
                    image, mask, batch
                )
                futures.append(future)

            # Collect results
            for future in tqdm(futures, desc="Processing patches"):
                batch_results = future.result()
                if batch_results:
                    batch_patches, batch_coords, batch_metrics = batch_results
                    patches.extend(batch_patches)
                    coordinates.extend(batch_coords)
                    patch_metrics.extend(batch_metrics)

        return patches, coordinates, patch_metrics

    def _process_patch_batch(self, 
                           image: np.ndarray, 
                           mask: np.ndarray, 
                           positions: List[Tuple[int, int]]) -> Tuple[List[np.ndarray], List[Tuple[int, int]], List[Dict]]:
        """Process a batch of patches in parallel"""
        batch_patches = []
        batch_coordinates = []
        batch_metrics = []

        for x, y in positions:
            # Extract patch and corresponding mask
            patch = image[y:y+self.patch_size, x:x+self.patch_size]
            patch_mask = mask[y:y+self.patch_size, x:x+self.patch_size]

            # Calculate tissue ratio
            tissue_ratio = np.sum(patch_mask) / (self.patch_size * self.patch_size)

            if tissue_ratio > self.min_tissue_ratio:
                # Calculate quality metrics
                quality_metrics = self.calculate_patch_quality(patch)
                metrics = {
                    'tissue_ratio': tissue_ratio,
                    'x': x,
                    'y': y,
                    **quality_metrics
                }

                batch_patches.append(patch)
                batch_coordinates.append((x, y))
                batch_metrics.append(metrics)

        return batch_patches, batch_coordinates, batch_metrics

    def calculate_patch_quality(self, patch):
        """Calculate quality metrics for a patch."""
        # Calculate mean and std of pixel intensities
        mean_intensity = np.mean(patch)
        std_intensity = np.std(patch)
        
        # Calculate contrast
        contrast = std_intensity / (mean_intensity + 1e-6)
        
        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'contrast': contrast
        }

    def visualize_patches(self, image, mask, patches, coordinates, output_path):
        """Visualize and save the patch extraction result."""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('Tissue Mask')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(image, cmap='gray')
        for (x, y) in coordinates:
            rect = plt.Rectangle((x, y), self.patch_size, self.patch_size, 
                               fill=False, edgecolor='red', linewidth=1)
            plt.gca().add_patch(rect)
        plt.title('Extracted Patches')
        plt.axis('off')
        
        plt.savefig(output_path)
        plt.close()

    def process_image(self, 
                     image_path: str, 
                     mask_path: str, 
                     output_dir: str, 
                     label: str, 
                     split: str,
                     memory_mapped: bool = True) -> Optional[List[Dict]]:
        """Process a single image with memory mapping for large files"""
        try:
            # Load image and mask with memory mapping for large files
            if memory_mapped:
                image = tiff.memmap(image_path, mode='r')
                mask = tiff.memmap(mask_path, mode='r')
            else:
                image = self.load_image(image_path)
                mask = self.load_mask(mask_path)

            if image is None or mask is None:
                return None

            # Extract patches
            patches, coordinates, patch_metrics = self.extract_patches(image, mask)
            
            # Save patches and collect information
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            patch_info = []

            # Create output directory structure
            patch_dir = os.path.join(output_dir, 'patches', split, label)
            os.makedirs(patch_dir, exist_ok=True)

            # Save patches in batches
            for i, (patch, (x, y), metrics) in enumerate(zip(patches, coordinates, patch_metrics)):
                patch_name = f"{base_name}_patch_{i}_x{x}_y{y}.npy"
                patch_path = os.path.join(patch_dir, patch_name)
                
                # Save patch as numpy array
                np.save(patch_path, patch)

                patch_info.append({
                    'original_image': base_name,
                    'patch_name': patch_name,
                    'x': x,
                    'y': y,
                    'label': label,
                    'split': split,
                    **metrics
                })

            # Visualize if visualizer is available
            if self.visualizer:
                self.visualizer.visualize_preprocessing(
                    original_image=image,
                    segmentation_mask=mask,
                    patches=patches[:min(len(patches), 100)],  # Visualize up to 100 patches
                    coordinates=coordinates[:min(len(coordinates), 100)],
                    image_name=f"{split}_{label}_{base_name}"
                )

            # Clean up memory-mapped files
            if memory_mapped:
                del image
                del mask
                gc.collect()

            return patch_info

        except Exception as e:
            self.logger.error(f"Error processing {image_path}:")
            self.logger.error(traceback.format_exc())
            return None

def process_batch(args):
    """Process a batch of images in parallel."""
    image_path, mask_path, output_dir, label, split, extractor = args
    return extractor.process_image(image_path, mask_path, output_dir, label, split)

def main():
    # Base directories
    base_dir = "data"
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    
    # Create extractor instance
    extractor = PatchExtractor()
    
    # Read the mask information
    mask_df = pd.read_csv(os.path.join(processed_dir, "mask_info.csv"))
    
    # Prepare arguments for parallel processing
    process_args = []
    for _, row in mask_df.iterrows():
        image_path = os.path.join(raw_dir, row['original_image'] + ".tif")
        mask_path = os.path.join(processed_dir, row['split'], row['label'], row['mask_name'])
        label = row['label']
        split = row['split']
        process_args.append((image_path, mask_path, processed_dir, label, split, extractor))
    
    # Process images in parallel
    all_patch_info = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_batch, process_args), 
                          total=len(process_args),
                          desc="Processing images"))
    
    # Flatten results and filter out None
    all_patch_info = [info for result in results if result is not None for info in result]
    
    # Save patch information
    patch_df = pd.DataFrame(all_patch_info)
    patch_info_path = os.path.join(processed_dir, "patch_info.csv")
    patch_df.to_csv(patch_info_path, index=False)
    logger.info(f"Saved patch information to {patch_info_path}")
    
    # Print summary
    print("\nPatch Extraction Summary:")
    for split in ['train', 'val', 'test']:
        split_patches = patch_df[patch_df['split'] == split]
        print(f"\n{split.capitalize()} Set:")
        for label in patch_df['label'].unique():
            count = len(split_patches[split_patches['label'] == label])
            avg_tissue_ratio = split_patches[split_patches['label'] == label]['tissue_ratio'].mean()
            avg_contrast = split_patches[split_patches['label'] == label]['contrast'].mean()
            print(f"  {label}: {count} patches")
            print(f"    Avg tissue ratio: {avg_tissue_ratio:.2f}")
            print(f"    Avg contrast: {avg_contrast:.2f}")
        print(f"  Total: {len(split_patches)} patches")
    
    print(f"\nTotal patches extracted: {len(patch_df)}")
    print(f"Average tissue ratio across all patches: {patch_df['tissue_ratio'].mean():.2f}")
    print(f"Average contrast across all patches: {patch_df['contrast'].mean():.2f}")

if __name__ == "__main__":
    main()