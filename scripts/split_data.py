import os
import random
import shutil
from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict, Tuple

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_split.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_image_size(image_path: str) -> float:
    """Get image size in MB"""
    return os.path.getsize(image_path) / (1024 * 1024)

def split_data(input_dir: str, 
               labels_file: str,
               output_dir: str,
               train_size: int = 10,
               val_size: int = 3,
               test_size: int = 2) -> None:
    """
    Split data into train, validation, and test sets
    
    Args:
        input_dir: Directory containing input images
        labels_file: Path to labels CSV file
        output_dir: Directory to save split data
        train_size: Number of images for training
        val_size: Number of images for validation
        test_size: Number of images for testing
    """
    logger = setup_logging()
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    logger.info(f"Loaded labels for {len(labels_df)} images")
    
    # Filter images by size and class
    image_files = []
    for img_path in Path(input_dir).glob('*.tif'):
        size_mb = get_image_size(str(img_path))
        if size_mb < 350:
            img_name = img_path.stem
            label_row = labels_df[labels_df['uuid'] == img_name]
            if not label_row.empty:
                is_bcc = label_row['cancer'].iloc[0] == 1
                risk_level = label_row['bcc_risk'].iloc[0]
                image_files.append((str(img_path), size_mb, is_bcc, risk_level))
    
    logger.info(f"Found {len(image_files)} images under 350MB")
    
    # Group images by class
    class_groups = {
        'high_risk_bcc': [],
        'low_risk_bcc': [],
        'non_bcc': []
    }
    
    for img_path, size, is_bcc, risk_level in image_files:
        if is_bcc:
            if risk_level == 1:  # High risk
                class_groups['high_risk_bcc'].append(img_path)
            else:  # Low risk
                class_groups['low_risk_bcc'].append(img_path)
        else:
            class_groups['non_bcc'].append(img_path)
    
    logger.info(f"High risk BCC: {len(class_groups['high_risk_bcc'])}")
    logger.info(f"Low risk BCC: {len(class_groups['low_risk_bcc'])}")
    logger.info(f"Non-BCC: {len(class_groups['non_bcc'])}")
    
    # Randomly select images for each set
    train_images = []
    val_images = []
    test_images = []
    
    # Select training images (7 BCC: 4 high risk, 3 low risk, 3 non-BCC)
    if len(class_groups['high_risk_bcc']) >= 4:
        train_images.extend(random.sample(class_groups['high_risk_bcc'], 4))
    else:
        train_images.extend(class_groups['high_risk_bcc'])
    
    if len(class_groups['low_risk_bcc']) >= 3:
        train_images.extend(random.sample(class_groups['low_risk_bcc'], 3))
    else:
        train_images.extend(class_groups['low_risk_bcc'])
    
    if len(class_groups['non_bcc']) >= 3:
        train_images.extend(random.sample(class_groups['non_bcc'], 3))
    else:
        train_images.extend(class_groups['non_bcc'])
    
    # Remove selected images from pools
    for img in train_images:
        if img in class_groups['high_risk_bcc']:
            class_groups['high_risk_bcc'].remove(img)
        elif img in class_groups['low_risk_bcc']:
            class_groups['low_risk_bcc'].remove(img)
        else:
            class_groups['non_bcc'].remove(img)
    
    # Select validation images (3 images)
    remaining_images = (class_groups['high_risk_bcc'] + 
                       class_groups['low_risk_bcc'] + 
                       class_groups['non_bcc'])
    if len(remaining_images) >= val_size:
        val_images = random.sample(remaining_images, val_size)
    else:
        val_images = remaining_images
    
    # Remove validation images from pools
    for img in val_images:
        if img in class_groups['high_risk_bcc']:
            class_groups['high_risk_bcc'].remove(img)
        elif img in class_groups['low_risk_bcc']:
            class_groups['low_risk_bcc'].remove(img)
        else:
            class_groups['non_bcc'].remove(img)
    
    # Select test images (2 images)
    remaining_images = (class_groups['high_risk_bcc'] + 
                       class_groups['low_risk_bcc'] + 
                       class_groups['non_bcc'])
    if len(remaining_images) >= test_size:
        test_images = random.sample(remaining_images, test_size)
    else:
        test_images = remaining_images
    
    # Copy files to respective directories
    for img_path in train_images:
        shutil.copy(img_path, os.path.join(output_dir, 'train'))
        logger.info(f"Copied {Path(img_path).name} to train set")
    
    for img_path in val_images:
        shutil.copy(img_path, os.path.join(output_dir, 'val'))
        logger.info(f"Copied {Path(img_path).name} to validation set")
    
    for img_path in test_images:
        shutil.copy(img_path, os.path.join(output_dir, 'test'))
        logger.info(f"Copied {Path(img_path).name} to test set")
    
    # Save split information
    split_info = {
        'train': [Path(img).name for img in train_images],
        'val': [Path(img).name for img in val_images],
        'test': [Path(img).name for img in test_images]
    }
    
    with open(os.path.join(output_dir, 'split_info.json'), 'w') as f:
        import json
        json.dump(split_info, f, indent=4)
    
    logger.info("Data split completed successfully")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split BCC dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input images')
    parser.add_argument('--labels_file', type=str, required=True,
                      help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save split data')
    args = parser.parse_args()
    
    split_data(args.input_dir, args.labels_file, args.output_dir) 