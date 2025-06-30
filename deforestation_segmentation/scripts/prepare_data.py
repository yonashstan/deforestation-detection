import os
import shutil
import argparse
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box

def create_binary_mask(image_path, is_deforested):
    """Create a binary mask where 1 represents deforestation"""
    # Read image
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    # Create a full image mask of 0s (forest) or 1s (deforested)
    mask = np.ones((height, width)) if is_deforested else np.zeros((height, width))
    return mask

def process_image(image_path, output_dir, size=512):
    """Process a single image and create its mask if needed"""
    # Read and resize image
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (size, size))
    
    # Save processed image
    image_name = image_path.name
    output_path = output_dir / 'images' / image_name
    cv2.imwrite(str(output_path), image)
    
    # Create and save mask if it's a deforested image
    if 'deforested' in str(image_path).lower():
        mask_path = output_dir / 'masks' / image_name
        create_binary_mask(image_path, True)
    else:
        # Create empty mask for forest images (no deforestation)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask_path = output_dir / 'masks' / image_name
        cv2.imwrite(str(mask_path), mask)

def prepare_dataset(
    raw_forest_dir,
    raw_deforested_dir,
    output_dir,
    val_split=0.2,
    target_size=(256, 256)
):
    # Create output directories
    output_dir = Path(output_dir)
    train_img_dir = output_dir / "train" / "images"
    train_mask_dir = output_dir / "train" / "masks"
    val_img_dir = output_dir / "val" / "images"
    val_mask_dir = output_dir / "val" / "masks"
    
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get file lists (now looking for .png files)
    forest_files = list(Path(raw_forest_dir).glob("*.png"))
    deforested_files = list(Path(raw_deforested_dir).glob("*.png"))
    
    if len(forest_files) == 0:
        raise ValueError(f"No PNG files found in {raw_forest_dir}")
    if len(deforested_files) == 0:
        raise ValueError(f"No PNG files found in {raw_deforested_dir}")
    
    print(f"Found {len(forest_files)} forest images and {len(deforested_files)} deforested images")
    
    # Split both classes into train/val
    forest_train, forest_val = train_test_split(
        forest_files, test_size=val_split, random_state=42
    )
    deforested_train, deforested_val = train_test_split(
        deforested_files, test_size=val_split, random_state=42
    )
    
    print(f"Training set: {len(forest_train)} forest, {len(deforested_train)} deforested")
    print(f"Validation set: {len(forest_val)} forest, {len(deforested_val)} deforested")
    
    def process_images(image_files, is_deforested, img_dir, mask_dir):
        for idx, src_path in enumerate(tqdm(image_files, desc="Processing images")):
            # Create a numbered filename
            prefix = "deforested" if is_deforested else "forest"
            dst_name = f"{prefix}_{idx:04d}.png"
            
            # Read and resize image
            img = cv2.imread(str(src_path))
            if img is None:
                print(f"Warning: Could not read image {src_path}")
                continue
                
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Save resized image
            dst_img_path = img_dir / dst_name
            cv2.imwrite(str(dst_img_path), img_resized)
            
            # Create and save mask
            mask = create_binary_mask(src_path, is_deforested)
            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            mask_uint8 = (mask_resized * 255).astype(np.uint8)
            
            dst_mask_path = mask_dir / dst_name
            cv2.imwrite(str(dst_mask_path), mask_uint8)
    
    # Process training data
    process_images(forest_train, False, train_img_dir, train_mask_dir)
    process_images(deforested_train, True, train_img_dir, train_mask_dir)
    
    # Process validation data
    process_images(forest_val, False, val_img_dir, val_mask_dir)
    process_images(deforested_val, True, val_img_dir, val_mask_dir)
    
    print("Data preparation completed!")
    return len(forest_train), len(deforested_train), len(forest_val), len(deforested_val)

if __name__ == "__main__":
    # Define paths
    raw_forest_dir = Path("data/raw/forest")
    raw_deforested_dir = Path("data/raw/deforested")
    output_dir = Path("data")
    
    # Remove rename.py if it exists in deforested directory
    rename_script = Path(raw_deforested_dir) / "rename.py"
    if rename_script.exists():
        rename_script.unlink()
    
    # Prepare the dataset
    train_forest, train_deforested, val_forest, val_deforested = prepare_dataset(
        raw_forest_dir=raw_forest_dir,
        raw_deforested_dir=raw_deforested_dir,
        output_dir=output_dir,
        val_split=0.2,  # 20% validation split
        target_size=(256, 256)
    )
    
    print("\nDataset preparation summary:")
    print(f"Total images processed: {train_forest + train_deforested + val_forest + val_deforested}")
    print(f"Training set: {train_forest + train_deforested} images")
    print(f"Validation set: {val_forest + val_deforested} images") 