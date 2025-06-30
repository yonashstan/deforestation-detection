"""Dataset classes for deforestation detection."""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .augmentation import DeforestationAugmentations

class DeforestationDataset(Dataset):
    """Dataset for deforestation detection from satellite imagery."""
    
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        transform: A.Compose = None,
        img_size: int = 384
    ):
        """Initialize dataset.
        
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing target masks
            transform: Albumentations transform pipeline
            img_size: Size to resize images to
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Get sorted lists of files
        self.image_files = sorted(list(self.images_dir.glob("*.png")))
        self.mask_files = sorted(list(self.masks_dir.glob("*.png")))
        
        assert len(self.image_files) == len(self.mask_files), \
            f"Got {len(self.image_files)} images but {len(self.mask_files)} masks"
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset.
        
        Returns:
            dict: Contains 'image' and 'mask' tensors and their file paths
        """
        # Load image and mask
        image = cv2.imread(str(self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # Binary mask
        
        # Apply transforms before adding channel dimension
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]  # Will be [C, H, W] due to ToTensorV2
            mask = transformed["mask"]    # Will be [H, W]
            
            # Add channel dimension to mask after transforms
            if len(mask.shape) == 2:  # If mask is still [H, W]
                mask = mask.unsqueeze(0)  # Make it [1, H, W]
        
        return {
            "image": image,
            "mask": mask,
            "image_path": str(self.image_files[idx]),
            "mask_path": str(self.mask_files[idx])
        }

def get_transforms(phase):
    """Get transforms for training or validation"""
    if phase == "train":
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                rotate=(-45, 45),
                translate_percent=(-0.0625, 0.0625),
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])
    else:  # val/test
        return A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ]) 