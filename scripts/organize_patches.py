import os
import shutil
from pathlib import Path
import logging
import json
import numpy as np
from tqdm import tqdm

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('organize_patches.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def organize_patches(input_dir: str, output_dir: str):
    """
    Organize patches into appropriate class directories
    
    Args:
        input_dir: Base directory containing patches
        output_dir: Directory to save organized patches
    """
    logger = logging.getLogger(__name__)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in ['bcc_high_risk', 'bcc_low_risk', 'non_malignant']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_dir = Path(input_dir) / split
        if not split_dir.exists():
            continue
            
        logger.info(f"Processing {split} split")
        
        for class_name in ['bcc_high_risk', 'bcc_low_risk', 'non_malignant']:
            class_dir = split_dir / class_name / 'patches'
            if not class_dir.exists():
                continue
                
            logger.info(f"Processing {class_name} class")
            output_class_dir = Path(output_dir) / split / class_name
            
            # Find all patch files for this class
            patch_files = list(class_dir.glob('*_patches_*.npy'))
            for patch_file in tqdm(patch_files, desc=f"Processing {class_name} patches"):
                try:
                    # Load patches
                    patches = np.load(str(patch_file))
                    image_id = patch_file.name.split('_patches_')[0]
                    batch_num = int(patch_file.name.split('_patches_')[1].split('.')[0])
                    
                    # Get corresponding metadata if exists
                    metadata_file = class_dir / f"{image_id}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}
                    
                    # Create directory for this image
                    image_dir = output_class_dir / image_id
                    os.makedirs(image_dir, exist_ok=True)
                    
                    # Save each patch as a separate file
                    for i, patch in enumerate(patches):
                        patch_name = f"patch_{batch_num}_{i:04d}.npy"
                        patch_path = image_dir / patch_name
                        np.save(str(patch_path), patch)
                        
                        # Save patch metadata
                        if metadata:
                            patch_metadata = {
                                'original_batch': batch_num,
                                'patch_index': i,
                                'image_metadata': metadata
                            }
                            metadata_path = image_dir / f"{patch_name.replace('.npy', '_metadata.json')}"
                            with open(metadata_path, 'w') as f:
                                json.dump(patch_metadata, f, indent=2)
                    
                except Exception as e:
                    logger.error(f"Error processing {patch_file}: {str(e)}")
                    continue
    
    logger.info("Patch organization completed")

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Organize patches into individual files')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing processed patches')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save organized patches')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting patch organization")
    
    # Organize patches
    organize_patches(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()