import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from typing import Tuple

class ForestAugmentation:
    def __init__(self, img_size=224, strong_aug=True):
        self.img_size = img_size
        
        # Basic transformations
        self.basic_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
        # Strong augmentations for contrastive learning
        self.strong_transform = A.Compose([
            A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.OneOf([
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.5),
            A.RandomShadow(p=0.3),
            # Forest-specific augmentations
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            ], p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
        
    def __call__(self, image, strong=True):
        """
        Apply augmentation to image
        Args:
            image: numpy array of shape (H, W, C)
            strong: whether to apply strong augmentation
        Returns:
            transformed image as torch tensor
        """
        if strong:
            return self.strong_transform(image=image)['image']
        return self.basic_transform(image=image)['image']
    
    def get_contrastive_pair(self, image):
        """Generate two different views of the same image"""
        return (
            self.strong_transform(image=image)['image'],
            self.strong_transform(image=image)['image']
        )

class DeforestationAugmentations:
    """Enhanced augmentations specifically tuned for deforestation detection."""
    
    @staticmethod
    def get_train_aug(img_size: int) -> A.Compose:
        """Get training augmentation pipeline.
        
        Args:
            img_size: Target image size
            
        Returns:
            Albumentations Compose object with training transforms
        """
        return A.Compose([
            # Spatial augmentations
            A.RandomResizedCrop(
                height=img_size,
                width=img_size,
                scale=(0.5, 1.0),  # Scale relative to original size
                ratio=(0.75, 1.33),  # Aspect ratio range
                p=1.0
            ),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=45,
                p=0.7,
                border_mode=cv2.BORDER_REFLECT101
            ),
            
            # Deforestation-specific augmentations
            A.OneOf([
                A.RandomShadow(p=1.0),  # Simulate cloud shadows
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
            ], p=0.5),
            
            # Color augmentations for different seasons/conditions
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20,
                    p=1.0
                ),
            ], p=0.5),
            
            # Weather and atmospheric effects
            A.OneOf([
                A.RandomFog(p=1.0),  # Simulate atmospheric conditions
                A.RandomRain(p=1.0),  # Simulate partial cloud coverage
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.3),
            
            # Elastic deformations for natural variation
            A.OneOf([
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03,
                    p=1.0
                ),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.0),
            ], p=0.3),
            
            # Normalization
            A.Normalize(),
            ToTensorV2(),
        ])

    @staticmethod
    def get_val_aug(img_size: int) -> A.Compose:
        """Get validation augmentation pipeline.
        
        Args:
            img_size: Target image size
            
        Returns:
            Albumentations Compose object with validation transforms
        """
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(),
            ToTensorV2(),
        ]) 