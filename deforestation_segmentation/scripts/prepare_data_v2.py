import os
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
from scipy import ndimage

def detect_clouds(image):
    """Detect clouds using color and brightness thresholds"""
    # Convert to HSV for better color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Clouds are typically bright and white/gray
    brightness = hsv[..., 2]
    saturation = hsv[..., 1]
    
    # Cloud mask where pixels are bright but not saturated
    cloud_mask = (brightness > 200) & (saturation < 30)
    
    # Clean up the mask using morphological operations
    cloud_mask = cloud_mask.astype(np.uint8) * 255
    kernel = np.ones((5,5), np.uint8)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
    cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
    
    return cloud_mask > 0

def enhance_edges(image):
    """Enhance edges to better detect deforestation boundaries"""
    # Convert to LAB color space for better color separation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract and enhance L channel
    l_channel = lab[..., 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l_channel)
    
    # Edge detection
    edges = cv2.Canny(enhanced_l, 50, 150)
    
    # Dilate edges slightly
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges

def create_binary_mask(image_path, is_deforested):
    """Create a binary mask where 1 represents deforestation"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    height, width = image.shape[:2]
    
    if is_deforested:
        # For deforested images, we need to be more careful
        # Detect clouds first
        cloud_mask = detect_clouds(image)
        
        # Enhance edges
        edges = enhance_edges(image)
        
        # Create initial mask
        mask = np.ones((height, width))
        
        # Remove clouds from mask
        mask[cloud_mask] = 0
        
        # Use edges to refine deforestation boundaries
        mask = ndimage.binary_dilation(mask & ~cloud_mask, structure=np.ones((3,3)))
    else:
        # For forest images, simply mark everything as forest (0)
        mask = np.zeros((height, width))
    
    return mask

def prepare_dataset(
    raw_forest_dir,
    raw_deforested_dir,
    output_dir,
    val_split=0.2,
    target_size=(256, 256)
):
    """Prepare dataset with improved cloud handling and edge detection"""
    # Create output directories
    output_dir = Path(output_dir)
    train_img_dir = output_dir / "train" / "images"
    train_mask_dir = output_dir / "train" / "masks"
    val_img_dir = output_dir / "val" / "images"
    val_mask_dir = output_dir / "val" / "masks"
    
    # Create visualization directory for debugging
    vis_dir = output_dir / "preprocessing_vis"
    for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, vis_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get file lists
    forest_files = list(Path(raw_forest_dir).glob("*.png"))
    deforested_files = list(Path(raw_deforested_dir).glob("*.png"))
    
    if len(forest_files) == 0:
        raise ValueError(f"No PNG files found in {raw_forest_dir}")
    if len(deforested_files) == 0:
        raise ValueError(f"No PNG files found in {raw_deforested_dir}")
    
    print(f"Found {len(forest_files)} forest images and {len(deforested_files)} deforested images")
    
    # Split data
    forest_train, forest_val = train_test_split(
        forest_files, test_size=val_split, random_state=42
    )
    deforested_train, deforested_val = train_test_split(
        deforested_files, test_size=val_split, random_state=42
    )
    
    def process_images(image_files, is_deforested, img_dir, mask_dir):
        for idx, src_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                # Create a numbered filename
                prefix = "deforested" if is_deforested else "forest"
                dst_name = f"{prefix}_{idx:04d}.png"
                
                # Read image
                img = cv2.imread(str(src_path))
                if img is None:
                    print(f"Warning: Could not read image {src_path}")
                    continue
                
                # Create visualization of preprocessing steps
                if is_deforested:
                    # Detect clouds
                    clouds = detect_clouds(img)
                    # Enhance edges
                    edges = enhance_edges(img)
                    
                    # Create visualization
                    vis_img = img.copy()
                    vis_img[clouds] = [0, 0, 255]  # Mark clouds in red
                    vis_img[edges > 0] = [0, 255, 0]  # Mark edges in green
                    
                    # Save visualization
                    vis_path = vis_dir / f"vis_{dst_name}"
                    cv2.imwrite(str(vis_path), vis_img)
                
                # Resize image
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
                
            except Exception as e:
                print(f"Error processing {src_path}: {str(e)}")
    
    # Process training data
    process_images(forest_train, False, train_img_dir, train_mask_dir)
    process_images(deforested_train, True, train_img_dir, train_mask_dir)
    
    # Process validation data
    process_images(forest_val, False, val_img_dir, val_mask_dir)
    process_images(deforested_val, True, val_img_dir, val_mask_dir)
    
    print("Data preparation completed!")
    print(f"\nPreprocessing visualizations saved in {vis_dir}")
    return len(forest_train), len(deforested_train), len(forest_val), len(deforested_val)

if __name__ == "__main__":
    # Define paths
    raw_forest_dir = Path("data/raw/forest")
    raw_deforested_dir = Path("data/raw/deforested")
    output_dir = Path("data")
    
    # Prepare the dataset
    train_forest, train_deforested, val_forest, val_deforested = prepare_dataset(
        raw_forest_dir=raw_forest_dir,
        raw_deforested_dir=raw_deforested_dir,
        output_dir=output_dir,
        val_split=0.2,
        target_size=(256, 256)
    )
    
    print("\nDataset preparation summary:")
    print(f"Total images processed: {train_forest + train_deforested + val_forest + val_deforested}")
    print(f"Training set: {train_forest + train_deforested} images")
    print(f"Validation set: {val_forest + val_deforested} images") 