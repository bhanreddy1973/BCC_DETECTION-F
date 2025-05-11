import os
import shutil
import pandas as pd
from pathlib import Path
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_image_paths(source_dir):
    """Get all TIFF image paths from a directory."""
    return [f for f in os.listdir(source_dir) if f.endswith(('.tif', '.tiff'))]

def copy_images(source_dir, source_paths, dest_dir, num_images):
    """Copy specified number of random images to destination directory."""
    selected_paths = random.sample(source_paths, min(num_images, len(source_paths)))
    for src_path in selected_paths:
        src_full_path = os.path.join(source_dir, src_path)
        dest_path = os.path.join(dest_dir, src_path)
        shutil.copy2(src_full_path, dest_path)
        logger.info(f"Copied image: {src_path}")
    return selected_paths

def create_label_mapping(image_paths, label, split, labels_df):
    """Create a mapping of image paths to their labels."""
    mappings = []
    for path in image_paths:
        # Get the UUID from the filename
        uuid = os.path.splitext(path)[0]
        
        # Find matching row in labels_df
        if label != "non_malignant":
            row = labels_df[labels_df['uuid'] == uuid].iloc[0]
            mapping = {
                "uuid": uuid,
                "patient_id": row['patient_id'],
                "case": row['case'],
                "cancer": row['cancer'],
                "bcc_risk": row['bcc_risk'],
                "image_path": str(path),
                "label": label,
                "split": split
            }
        else:
            # For non-malignant cases, use default values
            mapping = {
                "uuid": uuid,
                "patient_id": "N/A",
                "case": "N/A",
                "cancer": 0,
                "bcc_risk": 0,
                "image_path": str(path),
                "label": label,
                "split": split
            }
        mappings.append(mapping)
    return mappings

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Read the labels file
    labels_df = pd.read_csv("../dataset/package/bcc/data/labels/bcc_cases.csv")
    
    # Define source directories
    bcc_source_dir = "../dataset/package/bcc/data/images"
    non_malignant_source_dir = "../dataset/package/non-malignant/data/images"
    
    # Get all image paths
    print("\nScanning BCC images...")
    bcc_images = get_image_paths(bcc_source_dir)
    print(f"Found {len(bcc_images)} BCC images")
    
    print("\nScanning non-malignant images...")
    non_malignant_images = get_image_paths(non_malignant_source_dir)
    print(f"Found {len(non_malignant_images)} non-malignant images")
    
    # Initialize list to store all mappings
    all_mappings = []
    
    # Process BCC images
    base_dir = "data/raw"
    
    # Training split
    # Select 3 high risk and 3 low risk BCC images
    high_risk_bcc = labels_df[labels_df['bcc_risk'] == 2]['uuid'].tolist()
    low_risk_bcc = labels_df[labels_df['bcc_risk'] == 1]['uuid'].tolist()
    
    # Convert UUIDs to image paths
    high_risk_paths = [p for p in bcc_images if os.path.splitext(p)[0] in high_risk_bcc]
    low_risk_paths = [p for p in bcc_images if os.path.splitext(p)[0] in low_risk_bcc]
    
    print("\nSelecting training images...")
    # Select random images for training
    train_high_risk = copy_images(bcc_source_dir, high_risk_paths, os.path.join(base_dir, "train/bcc_high_risk"), 3)
    train_low_risk = copy_images(bcc_source_dir, low_risk_paths, os.path.join(base_dir, "train/bcc_low_risk"), 3)
    train_non_malignant = copy_images(non_malignant_source_dir, non_malignant_images, os.path.join(base_dir, "train/non_malignant"), 4)
    
    all_mappings.extend(create_label_mapping(train_high_risk, "bcc_high_risk", "train", labels_df))
    all_mappings.extend(create_label_mapping(train_low_risk, "bcc_low_risk", "train", labels_df))
    all_mappings.extend(create_label_mapping(train_non_malignant, "non_malignant", "train", labels_df))
    
    # Remove selected images from available pool
    bcc_images = [p for p in bcc_images if p not in train_high_risk + train_low_risk]
    non_malignant_images = [p for p in non_malignant_images if p not in train_non_malignant]
    
    print("\nSelecting validation images...")
    # Validation split
    val_bcc = copy_images(bcc_source_dir, bcc_images, os.path.join(base_dir, "val/bcc_high_risk"), 3)
    val_non_malignant = copy_images(non_malignant_source_dir, non_malignant_images, os.path.join(base_dir, "val/non_malignant"), 1)
    
    all_mappings.extend(create_label_mapping(val_bcc, "bcc_high_risk", "val", labels_df))
    all_mappings.extend(create_label_mapping(val_non_malignant, "non_malignant", "val", labels_df))
    
    # Remove selected images from available pool
    bcc_images = [p for p in bcc_images if p not in val_bcc]
    non_malignant_images = [p for p in non_malignant_images if p not in val_non_malignant]
    
    print("\nSelecting test images...")
    # Test split
    test_bcc = copy_images(bcc_source_dir, bcc_images, os.path.join(base_dir, "test/bcc_high_risk"), 4)
    test_non_malignant = copy_images(non_malignant_source_dir, non_malignant_images, os.path.join(base_dir, "test/non_malignant"), 2)
    
    all_mappings.extend(create_label_mapping(test_bcc, "bcc_high_risk", "test", labels_df))
    all_mappings.extend(create_label_mapping(test_non_malignant, "non_malignant", "test", labels_df))
    
    # Create and save the mapping DataFrame
    df = pd.DataFrame(all_mappings)
    # Reorder columns to match original format plus new columns
    columns = ['uuid', 'patient_id', 'case', 'cancer', 'bcc_risk', 'image_path', 'label', 'split']
    df = df[columns]
    
    # Save mapping.csv
    mapping_path = os.path.join(base_dir, "labels/mapping.csv")
    df.to_csv(mapping_path, index=False)
    print(f"\nSaved dataset mapping to: {mapping_path}")
    
    # Create and save labels.csv
    labels_df = df[['uuid', 'patient_id', 'case', 'cancer', 'bcc_risk', 'label']].copy()
    labels_path = os.path.join(base_dir, "labels/labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved labels to: {labels_path}")
    
    # Print summary
    print("\nDataset Organization Summary:")
    print("\nTraining Set:")
    print(f"  BCC High Risk: {len(train_high_risk)} images")
    print(f"  BCC Low Risk: {len(train_low_risk)} images")
    print(f"  Non-malignant: {len(train_non_malignant)} images")
    print(f"  Total: {len(train_high_risk) + len(train_low_risk) + len(train_non_malignant)} images")
    
    print("\nValidation Set:")
    print(f"  BCC: {len(val_bcc)} images")
    print(f"  Non-malignant: {len(val_non_malignant)} images")
    print(f"  Total: {len(val_bcc) + len(val_non_malignant)} images")
    
    print("\nTest Set:")
    print(f"  BCC: {len(test_bcc)} images")
    print(f"  Non-malignant: {len(test_non_malignant)} images")
    print(f"  Total: {len(test_bcc) + len(test_non_malignant)} images")
    
    print("\nTotal dataset size:", len(df), "images")
    print("\nAll images have been copied to data/raw/")

if __name__ == "__main__":
    main()