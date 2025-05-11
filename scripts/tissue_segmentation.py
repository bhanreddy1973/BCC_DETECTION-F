import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from skimage import morphology
from skimage.filters import threshold_otsu
import tifffile as tiff

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image(image_path):
    """Load a TIFF image."""
    try:
        image = tiff.imread(image_path)
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for tissue segmentation."""
    # Convert to grayscale if it's RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize to 0-255
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    return blurred

def segment_tissue(image):
    """Segment tissue from background using Otsu's thresholding."""
    # Apply Otsu's thresholding
    thresh = threshold_otsu(image)
    binary = image > thresh
    
    # Remove small objects
    cleaned = morphology.remove_small_objects(binary, min_size=5000)
    
    # Fill holes
    filled = morphology.remove_small_holes(cleaned, area_threshold=5000)
    
    return filled

def extract_patches(image, mask, patch_size=256, stride=128):
    """Extract patches from the image where tissue is present."""
    patches = []
    coordinates = []
    
    height, width = image.shape[:2]
    
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # Check if the patch contains enough tissue
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            tissue_ratio = np.sum(patch_mask) / (patch_size * patch_size)
            
            if tissue_ratio > 0.5:  # At least 50% tissue
                patch = image[y:y+patch_size, x:x+patch_size]
                patches.append(patch)
                coordinates.append((x, y))
    
    return patches, coordinates

def process_image(image_path, output_dir, label, split):
    """Process a single image: segment tissue and extract patches."""
    # Check if patches already exist for this image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    patch_dir = os.path.join(output_dir, split, label)
    existing_patches = []
    
    # Check if directory exists and contains patches for this image
    if os.path.exists(patch_dir):
        existing_patches = [f for f in os.listdir(patch_dir) if f.startswith(base_name)]
    
    # If patches exist, skip processing
    if existing_patches:
        logger.info(f"Patches already exist for {base_name}, skipping...")
        # Return patch info for existing patches
        patch_info = []
        for patch_name in existing_patches:
            # Extract coordinates from filename (assuming format base_name_patch_i.tif)
            patch_num = int(patch_name.split('_patch_')[1].split('.')[0])
            patch_info.append({
                'original_image': base_name,
                'patch_name': patch_name,
                'x': None,  # Coordinates not available for existing patches
                'y': None,
                'label': label,
                'split': split
            })
        return patch_info

    # Load image
    image = load_image(image_path)
    if image is None:
        return

    # Preprocess
    processed = preprocess_image(image)
    
    # Segment tissue
    tissue_mask = segment_tissue(processed)
    
    # Extract patches
    patches, coordinates = extract_patches(image, tissue_mask)
    
    # Save patches
    patch_info = []
    
    for i, (patch, (x, y)) in enumerate(zip(patches, coordinates)):
        patch_name = f"{base_name}_patch_{i}.tif"
        patch_path = os.path.join(output_dir, split, label, patch_name)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(patch_path), exist_ok=True)
        
        # Save patch
        tiff.imwrite(patch_path, patch)
        
        # Add patch information
        patch_info.append({
            'original_image': base_name,
            'patch_name': patch_name,
            'x': x,
            'y': y,
            'label': label,
            'split': split
        })
    
    return patch_info

def main():
    # Base directories
    base_dir = "data"
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    
    # Read the mapping file
    mapping_df = pd.read_csv(os.path.join(raw_dir, "labels/mapping.csv"))
    
    # Initialize list to store patch information
    all_patch_info = []
    
    # Process each image
    for _, row in mapping_df.iterrows():
        image_path = os.path.join(raw_dir, row['image_path'])
        label = row['label']
        split = row['split']
        
        logger.info(f"Processing {image_path}...")
        patch_info = process_image(image_path, processed_dir, label, split)
        
        if patch_info:
            all_patch_info.extend(patch_info)
    
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
            print(f"  {label}: {count} patches")
        print(f"  Total: {len(split_patches)} patches")
    
    print(f"\nTotal patches extracted: {len(patch_df)}")

if __name__ == "__main__":
    main()