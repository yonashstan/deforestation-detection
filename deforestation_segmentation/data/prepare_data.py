#!/usr/bin/env python3
"""
Data preparation script for Forest Loss Detection project.
Handles dataset organization, preprocessing, and splitting.
"""

import os
import sys
import logging
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import cv2
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
import json

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

class DataPreparator:
    def __init__(
        self,
        raw_data_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        augment_train: bool = True
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.augment_train = augment_train
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Create output directories
        self.create_output_dirs()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize augmentations
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """Setup augmentation pipelines"""
        # Basic augmentations for all splits
        self.basic_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Strong augmentations for training set
        self.train_transform = A.Compose([
            A.Resize(512, 512),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def create_output_dirs(self):
        """Create necessary output directories"""
        # Main data directory
        self.data_dir = self.output_dir / "data"
        
        # Split directories
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"
        
        # Create directories for each split
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "masks").mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"data_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def preprocess_image(self, image: np.ndarray, is_train: bool = False) -> np.ndarray:
        """Preprocess image for model input"""
        if is_train and self.augment_train:
            transformed = self.train_transform(image=image)
        else:
            transformed = self.basic_transform(image=image)
        
        return transformed["image"]
    
    def preprocess_mask(self, mask: np.ndarray, is_train: bool = False) -> np.ndarray:
        """Preprocess mask for model input"""
        if is_train and self.augment_train:
            transformed = self.train_transform(image=mask)
        else:
            transformed = self.basic_transform(image=mask)
        
        # Ensure binary mask
        mask = (transformed["image"] > 0).astype(np.float32)
        return mask
    
    def get_image_mask_pairs(self) -> List[Tuple[Path, Path]]:
        """Get pairs of image and mask files"""
        image_files = sorted(self.raw_data_dir.glob("**/*.jpg"))
        mask_files = sorted(self.raw_data_dir.glob("**/*_mask.png"))
        
        # Verify matching pairs
        pairs = []
        for img_path in image_files:
            mask_path = img_path.parent / f"{img_path.stem}_mask.png"
            if mask_path in mask_files:
                pairs.append((img_path, mask_path))
        
        return pairs
    
    def split_dataset(self, pairs: List[Tuple[Path, Path]]) -> Dict[str, List[Tuple[Path, Path]]]:
        """Split dataset into train, validation, and test sets"""
        # First split into train and temp
        train_pairs, temp_pairs = train_test_split(
            pairs,
            train_size=self.train_ratio,
            random_state=self.seed
        )
        
        # Split temp into validation and test
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_pairs, test_pairs = train_test_split(
            temp_pairs,
            train_size=val_ratio_adjusted,
            random_state=self.seed
        )
        
        return {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs
        }
    
    def copy_and_preprocess_files(self, pairs: List[Tuple[Path, Path]], split: str):
        """Copy and preprocess files for a specific split"""
        split_dir = getattr(self, f"{split}_dir")
        is_train = split == "train"
        
        for img_path, mask_path in tqdm(pairs, desc=f"Processing {split} set"):
            # Read files
            image = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Preprocess
            image = self.preprocess_image(image, is_train)
            mask = self.preprocess_mask(mask, is_train)
            
            # Save processed files
            img_output_path = split_dir / "images" / f"{img_path.stem}.npy"
            mask_output_path = split_dir / "masks" / f"{mask_path.stem}.npy"
            
            np.save(img_output_path, image)
            np.save(mask_output_path, mask)
            
            # For training set, create augmented versions
            if is_train and self.augment_train:
                for i in range(3):  # Create 3 augmented versions of each image
                    aug_image = self.preprocess_image(image, True)
                    aug_mask = self.preprocess_mask(mask, True)
                    
                    aug_img_output_path = split_dir / "images" / f"{img_path.stem}_aug{i}.npy"
                    aug_mask_output_path = split_dir / "masks" / f"{mask_path.stem}_aug{i}.npy"
                    
                    np.save(aug_img_output_path, aug_image)
                    np.save(aug_mask_output_path, aug_mask)
    
    def prepare_dataset(self):
        """Main function to prepare the dataset"""
        logging.info("Starting dataset preparation...")
        
        # Get image-mask pairs
        pairs = self.get_image_mask_pairs()
        logging.info(f"Found {len(pairs)} image-mask pairs")
        
        # Split dataset
        splits = self.split_dataset(pairs)
        for split, split_pairs in splits.items():
            logging.info(f"{split.capitalize()} set size: {len(split_pairs)}")
        
        # Process each split
        for split, split_pairs in splits.items():
            logging.info(f"Processing {split} set...")
            self.copy_and_preprocess_files(split_pairs, split)
        
        # Save dataset statistics
        self.save_dataset_stats(splits)
        
        logging.info("Dataset preparation completed!")
    
    def save_dataset_stats(self, splits: Dict[str, List[Tuple[Path, Path]]]):
        """Save dataset statistics"""
        stats = {
            "total_samples": sum(len(pairs) for pairs in splits.values()),
            "split_sizes": {
                split: len(pairs) for split, pairs in splits.items()
            },
            "split_ratios": {
                split: len(pairs) / len(splits["train"]) 
                for split, pairs in splits.items()
            }
        }
        
        # Save stats as JSON
        with open(self.data_dir / "dataset_stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        # Log stats
        logging.info("Dataset Statistics:")
        logging.info(f"Total samples: {stats['total_samples']}")
        for split, size in stats["split_sizes"].items():
            logging.info(f"{split.capitalize()} set: {size} samples")

def main():
    # Configuration
    config = {
        "raw_data_dir": "raw_data",
        "output_dir": "prepared_data",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 42,
        "augment_train": True
    }
    
    # Create data preparator
    preparator = DataPreparator(
        raw_data_dir=config["raw_data_dir"],
        output_dir=config["output_dir"],
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        seed=config["seed"],
        augment_train=config["augment_train"]
    )
    
    # Prepare dataset
    preparator.prepare_dataset()

if __name__ == "__main__":
    main() 