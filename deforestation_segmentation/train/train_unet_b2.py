#!/usr/bin/env python
"""Train a UNet++/UNet model (EfficientNet-B2 encoder) for deforestation segmentation.

Key design choices
------------------
1. Supervised-only (no contrastive loss) – simpler & more stable on small datasets.
2. Hybrid Focal-Tversky+Lovasz loss for class-imbalance & boundary quality.
3. One-Cycle LR schedule with warm-up for fast convergence.
4. Extensive yet light augmentations tuned for aerial imagery.
5. Mixed-precision support on Apple-Silicon (MPS) & CUDA.
6. Works out-of-box on a MacBook Pro M1-Pro 16 GB (≈6–8 GB VRAM usage @ 352²).
7. Comprehensive performance monitoring and analysis.

Example
-------
python scripts/train_unet_b2.py \
    --data-dir data/synth_v5 \
    --out-dir models/effb2_352_focal \
    --epochs 25 \
    --batch-size 8 \
    --img-size 352
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json
import psutil
import gc
import math
import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import segmentation_models_pytorch as smp

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import DeforestationDataset
from src.data.augmentation import DeforestationAugmentations
from src.models.deforestation_model import DeforestationLoss, get_model
from src.utils.visualization import TrainingVisualizer
from src.utils.training import get_scheduler

class TrainingTimer:
    """Detailed timing analysis for training phases."""
    
    def __init__(self):
        self.timers = defaultdict(float)
        self.counts = defaultdict(int)
        self.start_times = {}
        self.epoch_start = None
        self.training_start = None
        
    def start_training(self):
        self.training_start = time.perf_counter()
        
    def start_epoch(self):
        self.epoch_start = time.perf_counter()
        
    def end_epoch(self) -> float:
        if self.epoch_start is not None:
            epoch_time = time.perf_counter() - self.epoch_start
            self.epoch_start = None
            return epoch_time
        return 0.0
        
    def start(self, name):
        self.start_times[name] = time.perf_counter()
        
    def stop(self, name):
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timers[name] += elapsed
            self.counts[name] += 1
            del self.start_times[name]
            
    def get_stats(self) -> Dict:
        stats = {}
        total_time = time.perf_counter() - self.training_start if self.training_start else 0
        
        for name in self.timers:
            total = self.timers[name]
            count = self.counts[name]
            stats[name] = {
                'total': total,
                'count': count,
                'avg': total/count if count else 0,
                'percent': (total/total_time * 100) if total_time else 0
            }
        
        stats['total_time'] = total_time
        return stats
    
    def save_stats(self, path: Path):
        stats = self.get_stats()
        with open(path / 'timing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

class MemoryTracker:
    """Track CPU and GPU memory usage during training."""
    
    def __init__(self):
        self.cpu_stats = []
        self.gpu_stats = []
        self.timestamps = []
        
    def update(self):
        timestamp = time.time()
        self.timestamps.append(timestamp)
        
        # CPU Memory
        cpu_mem = psutil.Process().memory_info().rss / 1024**3  # GB
        self.cpu_stats.append(cpu_mem)
        
        # GPU Memory
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            self.gpu_stats.append((gpu_mem_alloc, gpu_mem_reserved))
        
    def get_stats(self) -> Dict:
        stats = {
            'cpu_memory': {
                'current': self.cpu_stats[-1] if self.cpu_stats else 0,
                'peak': max(self.cpu_stats) if self.cpu_stats else 0,
                'mean': np.mean(self.cpu_stats) if self.cpu_stats else 0
            }
        }
        
        if torch.cuda.is_available() and self.gpu_stats:
            allocated = [x[0] for x in self.gpu_stats]
            reserved = [x[1] for x in self.gpu_stats]
            stats['gpu_memory'] = {
                'current_allocated': allocated[-1],
                'peak_allocated': max(allocated),
                'mean_allocated': np.mean(allocated),
                'current_reserved': reserved[-1],
                'peak_reserved': max(reserved),
                'mean_reserved': np.mean(reserved)
            }
            
        return stats

class BatchTimeTracker:
    """Track and analyze batch processing times."""
    
    def __init__(self, window_size: int = 100):
        self.times = []
        self.window_size = window_size
        self.last_time = None
        
    def start_batch(self):
        self.last_time = time.perf_counter()
        
    def end_batch(self):
        if self.last_time is not None:
            batch_time = time.perf_counter() - self.last_time
            self.times.append(batch_time)
            if len(self.times) > self.window_size:
                self.times.pop(0)
            self.last_time = None
            
    def get_stats(self) -> Dict:
        if not self.times:
            return {}
        
        times = np.array(self.times)
        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'recent': float(np.mean(times[-min(10, len(times)):]))  # Last 10 batches
        }

class PerformanceAnalyzer:
    """Analyze training performance and provide recommendations."""
    
    def __init__(self, device_type: str):
        self.device_type = device_type
        self.warnings = []
        self.recommendations = []
        
    def analyze_timing(self, timing_stats: Dict) -> List[str]:
        recommendations = []
        
        # Data transfer analysis
        if 'data_transfer' in timing_stats:
            data_pct = timing_stats['data_transfer']['percent']
            if data_pct > 10:
                recommendations.append(
                    f"High data transfer time ({data_pct:.1f}%) - Consider:"
                    "\n  • Increasing num_workers"
                    "\n  • Using pin_memory=True"
                    "\n  • Prefetching data"
                )
        
        # Compute analysis
        if 'forward' in timing_stats and 'backward' in timing_stats:
            compute_pct = (timing_stats['forward']['percent'] + 
                         timing_stats['backward']['percent'])
            if compute_pct > 85:
                recommendations.append(
                    f"High compute time ({compute_pct:.1f}%) - Consider:"
                    "\n  • Reducing batch size"
                    "\n  • Using gradient accumulation"
                    "\n  • Enabling mixed precision training"
                )
        
        # Device-specific recommendations
        if self.device_type == "cpu":
            recommendations.append(
                "Running on CPU - Consider using GPU if available for significant speedup"
            )
        elif self.device_type == "mps":
            recommendations.append(
                "Running on Apple Silicon - Note that some operations may fall back to CPU"
            )
            
        return recommendations
    
    def analyze_memory(self, memory_stats: Dict) -> List[str]:
        recommendations = []
        
        # CPU Memory
        if memory_stats.get('cpu_memory', {}).get('current', 0) > 32:  # >32GB
            recommendations.append(
                "High CPU memory usage - Consider:"
                "\n  • Reducing batch size"
                "\n  • Using fewer worker processes"
                "\n  • Implementing memory-efficient data loading"
            )
        
        # GPU Memory
        gpu_stats = memory_stats.get('gpu_memory', {})
        if gpu_stats:
            allocated = gpu_stats.get('current_allocated', 0)
            reserved = gpu_stats.get('current_reserved', 0)
            if allocated/reserved > 0.95:  # >95% utilization
                recommendations.append(
                    "High GPU memory utilization - Consider:"
                    "\n  • Reducing batch size"
                    "\n  • Using gradient checkpointing"
                    "\n  • Enabling memory-efficient attention"
                )
                
        return recommendations
    
    def analyze_batch_times(self, batch_stats: Dict) -> List[str]:
        recommendations = []
        
        if not batch_stats:
            return recommendations
            
        # Check for high variance
        if batch_stats['std'] > 0.5 * batch_stats['mean']:
            recommendations.append(
                "High batch time variance - Consider:"
                "\n  • Investigating system bottlenecks"
                "\n  • Checking for background processes"
                "\n  • Monitoring thermal throttling"
            )
            
        # Check for increasing trend
        if batch_stats['recent'] > 1.1 * batch_stats['mean']:
            recommendations.append(
                "Increasing batch times detected - Consider:"
                "\n  • Checking for memory leaks"
                "\n  • Monitoring system resources"
                "\n  • Reducing batch size if persistent"
            )
            
        return recommendations
    
    def get_all_recommendations(self, 
                              timing_stats: Dict,
                              memory_stats: Dict,
                              batch_stats: Dict) -> List[str]:
        all_recs = (
            self.analyze_timing(timing_stats) +
            self.analyze_memory(memory_stats) +
            self.analyze_batch_times(batch_stats)
        )
        
        # Deduplicate while preserving order
        seen = set()
        return [x for x in all_recs if not (x in seen or seen.add(x))]

class DeforestationAugmentations:
    """Augmentation pipelines for deforestation segmentation."""
    
    @staticmethod
    def get_train_aug(img_size: Tuple[int, int]):
        """Get training augmentations."""
        return A.Compose([
            # Spatial augmentations
            A.RandomResizedCrop(
                size=img_size,
                scale=(0.7, 1.0),
                ratio=(0.8, 1.2),
                p=1.0
            ),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent=0.1,
                scale=0.2,
                rotate=45,
                p=0.7,
                border_mode=cv2.BORDER_REFLECT101
            ),
            
            # Deforestation-specific augmentations
            A.OneOf([
                A.RandomShadow(p=1.0),
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
                A.RandomFog(p=1.0),
                A.RandomRain(p=1.0),
                A.GaussNoise(p=1.0),
            ], p=0.3),
            
            # Elastic deformations for natural variation
            A.OneOf([
                A.ElasticTransform(
                    alpha=120,
                    sigma=6,
                    p=1.0
                ),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(
                    distort_limit=2,
                    p=1.0
                ),
            ], p=0.3),
            
            # Normalization
            A.Normalize(),
            ToTensorV2(),
        ])

    @staticmethod
    def get_val_aug(img_size: Tuple[int, int]):
        """Get validation augmentations."""
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(),
            ToTensorV2(),
        ])

# ------------------------------------------------------------------------- #
# Losses

class DeforestationLoss(nn.Module):
    """Enhanced loss function specifically for deforestation detection."""
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        dice_beta: float = 0.7,
        edge_weight: float = 0.3,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.dice_beta = dice_beta
        self.edge_weight = edge_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def check_nan(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Check for NaN values and print debug info."""
        if torch.isnan(x).any():
            print(f"\nNaN detected in {name}:")
            print(f"Min: {x.min().item()}, Max: {x.max().item()}")
            print(f"Mean: {x.mean().item()}, Std: {x.std().item()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss with gradient clipping."""
        pred = self.check_nan(pred, "pred_before_focal")
        pred = torch.clamp(pred, min=-10, max=10)  # More aggressive clamping
        
        bce = self.bce(pred, target)
        bce = self.check_nan(bce, "bce")
        bce = torch.clamp(bce, min=-10, max=10)
        
        pt = torch.exp(-bce)
        pt = torch.clamp(pt, min=1e-7, max=1.0)
        pt = self.check_nan(pt, "pt")
        
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_weight = self.check_nan(focal_weight, "focal_weight")
        
        if self.focal_alpha is not None:
            alpha_weight = target * self.focal_alpha + (1 - target) * (1 - self.focal_alpha)
            focal_weight = focal_weight * alpha_weight
        
        loss = (focal_weight * bce).mean()
        return self.check_nan(loss, "focal_loss")
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss with beta weighting and gradient clipping."""
        pred = torch.sigmoid(torch.clamp(pred, min=-10, max=10))
        pred = torch.clamp(pred, min=self.smooth, max=1-self.smooth)
        pred = self.check_nan(pred, "pred_dice")
        
        # True Positives, False Positives, False Negatives with smoothing
        tp = (pred * target).sum(dim=(2, 3)) + self.smooth
        fp = (pred * (1 - target)).sum(dim=(2, 3)) + self.smooth
        fn = ((1 - pred) * target).sum(dim=(2, 3)) + self.smooth
        
        tp = self.check_nan(tp, "tp")
        fp = self.check_nan(fp, "fp")
        fn = self.check_nan(fn, "fn")
        
        # Beta-weighted Dice coefficient
        numerator = (1 + self.dice_beta**2) * tp
        denominator = (1 + self.dice_beta**2) * tp + self.dice_beta**2 * fn + fp
        
        dice_score = numerator / denominator
        dice_score = self.check_nan(dice_score, "dice_score")
        
        return 1 - dice_score.mean()
    
    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge-aware loss with stability checks."""
        pred = torch.clamp(pred, min=self.smooth, max=1-self.smooth)
        pred = self.check_nan(pred, "pred_edge")
        
        # Extract edges using Sobel with smaller kernels
        kernel_x = torch.FloatTensor([[-1, 0, 1]]).to(pred.device)
        kernel_y = torch.FloatTensor([[-1], [0], [1]]).to(pred.device)
        kernel_x = kernel_x.view(1, 1, 1, 3)
        kernel_y = kernel_y.view(1, 1, 3, 1)
        
        pred_edges_x = F.conv2d(pred, kernel_x, padding=(0, 1))
        pred_edges_y = F.conv2d(pred, kernel_y, padding=(1, 0))
        target_edges_x = F.conv2d(target, kernel_x, padding=(0, 1))
        target_edges_y = F.conv2d(target, kernel_y, padding=(1, 0))
        
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + self.smooth)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + self.smooth)
        
        pred_edges = self.check_nan(pred_edges, "pred_edges")
        target_edges = self.check_nan(target_edges, "target_edges")
        
        return F.mse_loss(pred_edges, target_edges)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, min=-10, max=10)
        pred = self.check_nan(pred, "pred_input")
        target = self.check_nan(target, "target_input")
        
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        edge = self.edge_loss(torch.sigmoid(pred), target)
        
        # More balanced loss combination
        loss = 0.5 * focal + 0.4 * dice + 0.1 * edge
        loss = torch.clamp(loss, min=0.0, max=10.0)  # Ensure positive, bounded loss
        
        return self.check_nan(loss, "final_loss")

# ------------------------------------------------------------------------- #
# Utils

def compute_iou(pred: np.ndarray, gt: np.ndarray, eps=1e-6) -> float:
    """Compute IoU score between prediction and ground truth masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((intersection + eps) / (union + eps))

def iou_score(pred: torch.Tensor, gt: torch.Tensor, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * gt).sum((1, 2, 3))
    union = pred.sum((1, 2, 3)) + gt.sum((1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def split_dataset(root: Path, train_ratio=0.8, seed=42):
    """Split dataset into train and validation sets."""
    print(f"\nSetting up dataset from {root}")
    
    train_d, val_d = root / "train", root / "val"
    
    # Check if already split
    if (train_d / "images").exists() and (train_d / "masks").exists() and \
       (val_d / "images").exists() and (val_d / "masks").exists():
        print("Found existing train/val split")
        return train_d, val_d

    print("Creating new train/val split")
    
    # Verify source directories exist
    src_imgs = root / "images"
    src_masks = root / "masks"
    if not src_imgs.exists() or not src_masks.exists():
        raise ValueError(
            f"Source directories not found:\n"
            f"Images dir ({src_imgs}): {'exists' if src_imgs.exists() else 'missing'}\n"
            f"Masks dir ({src_masks}): {'exists' if src_masks.exists() else 'missing'}"
        )

    # Create target directories
    for d in [train_d / "images", train_d / "masks", 
              val_d / "images", val_d / "masks"]:
        d.mkdir(parents=True, exist_ok=True)

    # Get and verify image/mask pairs
    imgs = sorted(src_imgs.glob("*.png"))
    if not imgs:
        raise ValueError(f"No .png images found in {src_imgs}")
        
    # Verify each image has a corresponding mask
    valid_pairs = []
    for img_p in imgs:
        mask_p = src_masks / img_p.name
        if mask_p.exists():
            valid_pairs.append((img_p, mask_p))
        else:
            print(f"Warning: No mask found for {img_p.name}")
    
    if not valid_pairs:
        raise ValueError("No valid image-mask pairs found")
    
    print(f"Found {len(valid_pairs)} valid image-mask pairs")
    
    # Split and copy files
    random.Random(seed).shuffle(valid_pairs)
    split = int(len(valid_pairs) * train_ratio)
    
    for subset, pairs in [
        (train_d, valid_pairs[:split]), 
        (val_d, valid_pairs[split:])
    ]:
        print(f"Copying {len(pairs)} pairs to {subset}")
        for img_p, mask_p in pairs:
            shutil.copy2(img_p, subset / "images" / img_p.name)
            shutil.copy2(mask_p, subset / "masks" / mask_p.name)
    
    return train_d, val_d

def black_ignore_weight(imgs: torch.Tensor, thr: float = 0.05) -> torch.Tensor:
    """Return 1 for pixels to *keep* in loss, 0 for nearly black background.

    imgs expected in [0,1] range (after Albumentations Normalize with default
    mean/std). thr of 0.05 works well for pure black = 0.
    """
    gray = imgs.mean(dim=1, keepdim=True)  # B,1,H,W
    return (gray > thr).float()

# ------------------------------------------------------------------------- #
# Main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="v5/synth_v5", help="Root with train/ & val/ subdirs containing images/ & masks/")
    p.add_argument("--out-dir", type=str, default="v5/models/deforestation_v2", help="Directory to save checkpoints & plots")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)  # Reduced batch size for stability
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)  # Reduced learning rate for stability
    p.add_argument("--seed", type=int, default=42)
    
    # Model architecture options
    p.add_argument("--model", type=str, default="unet",
                  choices=["unet", "unetplusplus", "deeplabv3plus"],
                  help="Segmentation model architecture")
    p.add_argument("--encoder", type=str, default="efficientnet-b0",
                  choices=["efficientnet-b0", "efficientnet-b2", "efficientnet-b4"],
                  help="Encoder backbone")
    
    # Loss function parameters
    p.add_argument("--focal-gamma", type=float, default=2.0,
                  help="Focal loss gamma parameter")
    p.add_argument("--dice-beta", type=float, default=0.7,
                  help="Dice loss beta parameter")
    p.add_argument("--edge-weight", type=float, default=0.3,  # Reduced edge weight
                  help="Edge loss weight")
    
    # Learning rate schedule
    p.add_argument("--warmup-pct", type=float, default=0.05,  # Shorter warmup
                  help="Percentage of steps for LR warmup")
    p.add_argument("--save-every", type=int, default=1,
                  help="Save model and logs every N epochs (default: 1)")
    return p.parse_args()

def get_model(args, device):
    """Get segmentation model based on arguments."""
    model_params = {
        "encoder_name": args.encoder,
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 1,
        "activation": None,  # We'll handle this in loss
    }
    
    if args.model == "unetplusplus":
        model = smp.UnetPlusPlus(
            **model_params,
            decoder_attention_type="scse"
        )
    elif args.model == "deeplabv3plus":
        model = smp.DeepLabV3Plus(**model_params)
    else:  # unet
        model = smp.Unet(
            **model_params,
            decoder_attention_type="scse"
        )
    
    return model.to(device)

class TrainingVisualizer:
    """Handles all training visualizations and analytics."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.plots_dir = out_dir / "training_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of plots
        self.metric_plots_dir = self.plots_dir / "metrics"
        self.pred_plots_dir = self.plots_dir / "predictions"
        self.dist_plots_dir = self.plots_dir / "distributions"
        self.grad_plots_dir = self.plots_dir / "gradients"
        
        for d in [self.metric_plots_dir, self.pred_plots_dir, 
                 self.dist_plots_dir, self.grad_plots_dir]:
            d.mkdir(exist_ok=True)
        
        # Initialize history with only the metrics we're tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'epoch': []
        }
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def update_history(self, metrics: dict, epoch: int):
        """Update training history with new metrics."""
        for k, v in metrics.items():
            self.history[k].append(v)
        self.history['epoch'].append(epoch)
    
    def plot_metrics(self, epoch: int):
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Training Metrics - Epoch {epoch}', fontsize=16)
        
        # Loss plot
        ax = axes[0]
        ax.plot(self.history['epoch'], self.history['train_loss'], label='Train')
        ax.plot(self.history['epoch'], self.history['val_loss'], label='Val')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # IoU plot
        ax = axes[1]
        ax.plot(self.history['epoch'], self.history['val_iou'], label='Val IoU')
        ax.set_title('Validation IoU')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.metric_plots_dir / f'metrics_epoch_{epoch}.png')
        plt.close()
    
    def plot_predictions(self, images: torch.Tensor, masks: torch.Tensor, 
                        preds: torch.Tensor, epoch: int, num_samples: int = 6):
        """Plot sample predictions vs ground truth."""
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        fig.suptitle(f'Predictions vs Ground Truth - Epoch {epoch}', fontsize=16)
        
        for i in range(num_samples):
            # Original image
            ax = axes[i, 0]
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            ax.imshow(img)
            ax.set_title('Original Image')
            ax.axis('off')
            
            # Ground truth mask
            ax = axes[i, 1]
            mask = masks[i].cpu().squeeze().numpy()  # Remove extra dimension
            ax.imshow(mask, cmap='RdYlBu')
            ax.set_title('Ground Truth')
            ax.axis('off')
            
            # Predicted mask
            ax = axes[i, 2]
            pred = preds[i].cpu().squeeze().numpy()  # Remove extra dimension
            ax.imshow(pred, cmap='RdYlBu')
            ax.set_title(f'Prediction (IoU: {compute_iou(pred > 0.5, mask):.3f})')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.pred_plots_dir / f'predictions_epoch_{epoch}.png')
        plt.close()
    
    def plot_prediction_distributions(self, preds: torch.Tensor, masks: torch.Tensor, epoch: int):
        """Plot distribution of prediction probabilities."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'Prediction Distributions - Epoch {epoch}', fontsize=16)
        
        # Overall prediction distribution
        ax = axes[0]
        probs = torch.sigmoid(preds).cpu().numpy().flatten()
        sns.histplot(probs, bins=50, ax=ax)
        ax.set_title('Overall Prediction Distribution')
        ax.set_xlabel('Prediction Probability')
        
        # Prediction distribution by true class
        ax = axes[1]
        true_pos = probs[masks.cpu().numpy().flatten() == 1]
        true_neg = probs[masks.cpu().numpy().flatten() == 0]
        sns.histplot(true_pos, bins=25, alpha=0.5, label='True Positive', ax=ax)
        sns.histplot(true_neg, bins=25, alpha=0.5, label='True Negative', ax=ax)
        ax.set_title('Prediction Distribution by True Class')
        ax.set_xlabel('Prediction Probability')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.dist_plots_dir / f'distributions_epoch_{epoch}.png')
        plt.close()
    
    def plot_gradient_flow(self, named_parameters, epoch: int):
        """Plot gradient flow through network layers."""
        ave_grads = []
        max_grads = []
        layers = []
        
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())
        
        plt.figure(figsize=(15, 5))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02)
        plt.xlabel("Layers")
        plt.ylabel("Average gradient")
        plt.title(f"Gradient flow - Epoch {epoch}")
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(self.grad_plots_dir / f'gradients_epoch_{epoch}.png')
        plt.close()
    
    def save_epoch_summary(self, epoch: int, metrics: dict):
        """Save epoch metrics to CSV."""
        df = pd.DataFrame([metrics])
        df['epoch'] = epoch
        
        csv_path = self.plots_dir / 'training_history.csv'
        if not csv_path.exists():
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)

def get_optimizer(model, lr=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

def get_scheduler(optimizer, num_epochs, num_steps_per_epoch):
    """Get cosine learning rate scheduler with warmup."""
    num_training_steps = num_epochs * num_steps_per_epoch
    num_warmup_steps = min(1000, num_training_steps // 10)  # 10% warmup
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, criterion, optimizer, scheduler, scaler, train_loader, device, epoch):
    model.train()
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    running_loss = 0.0
    
    def check_grad_nan(model, step):
        """Check for NaN gradients in model."""
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"\nNaN gradient detected in {name} at step {step}")
                print(f"Grad stats - Min: {param.grad.min().item()}, Max: {param.grad.max().item()}")
                has_nan = True
        return has_nan

    for step, batch in enumerate(pbar):
        try:
            # Move data to device
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with or without gradient scaling
            if device.type == 'mps':
                # MPS doesn't support autocast, use regular forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                if torch.isnan(loss):
                    print(f"\nNaN loss detected at step {step}!")
                    print(f"Image stats - Min: {images.min().item()}, Max: {images.max().item()}, Mean: {images.mean().item()}")
                    print(f"Mask stats - Min: {masks.min().item()}, Max: {masks.max().item()}, Mean: {masks.mean().item()}")
                    print(f"Output stats - Min: {outputs.min().item()}, Max: {outputs.max().item()}, Mean: {outputs.mean().item()}")
                    continue  # Skip this batch
                
                # Regular backward pass
                loss.backward()
                
                # Check for NaN gradients
                if check_grad_nan(model, step):
                    print("Skipping gradient update due to NaN gradients")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Regular optimizer step
                optimizer.step()
                
            else:  # CUDA or CPU
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    
                    if torch.isnan(loss):
                        print(f"\nNaN loss detected at step {step}!")
                        print(f"Image stats - Min: {images.min().item()}, Max: {images.max().item()}, Mean: {images.mean().item()}")
                        print(f"Mask stats - Min: {masks.min().item()}, Max: {masks.max().item()}, Mean: {masks.mean().item()}")
                        print(f"Output stats - Min: {outputs.min().item()}, Max: {outputs.max().item()}, Mean: {outputs.mean().item()}")
                        continue  # Skip this batch
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Check for NaN gradients
                if check_grad_nan(model, step):
                    print("Skipping gradient update due to NaN gradients")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            
            # Update scheduler
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                'batch': f'{step}/{len(train_loader)}'
            })
            
        except RuntimeError as e:
            print(f"\nError at step {step}: {str(e)}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            else:
                gc.collect()
            continue
    
    return running_loss / len(train_loader)

def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_iou = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # Forward pass without autocast for MPS
            if device.type == 'mps':
                outputs = model(images)
                loss = criterion(outputs, masks)
            else:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            
            val_loss += loss.item()
            
            # Calculate IoU
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (pred_masks * masks).sum((2, 3))
            union = (pred_masks + masks).clamp(0, 1).sum((2, 3))
            batch_iou = (intersection + 1e-6) / (union + 1e-6)
            val_iou += batch_iou.mean().item()
    
    return val_loss / num_batches, val_iou / num_batches

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, val_iou, filepath):
    """Save a checkpoint of the training state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_iou': val_iou
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the checkpoint
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Enable deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Initialize performance monitoring
    timer = TrainingTimer()
    memory_tracker = MemoryTracker()
    batch_tracker = BatchTimeTracker(window_size=100)
    
    # Set up device
    if torch.backends.mps.is_available():
        print("\nUsing Apple-Silicon MPS")
        device = torch.device("mps")
        # MPS doesn't support mixed precision yet, so we'll use regular training
        use_amp = False
        scaler = None
    elif torch.cuda.is_available():
        print("\nUsing CUDA GPU")
        device = torch.device("cuda")
        use_amp = True
        scaler = torch.cuda.amp.GradScaler()
    else:
        print("\nUsing CPU")
        device = torch.device("cpu")
        use_amp = False
        scaler = None
    
    analyzer = PerformanceAnalyzer(device.type)

    root = Path(args.data_dir)
    img_size = (args.img_size, args.img_size)
    
    timer.start('data_setup')
    train_dir, val_dir = split_dataset(root)

    # Enhanced augmentations
    train_aug = DeforestationAugmentations.get_train_aug(img_size)
    val_aug = DeforestationAugmentations.get_val_aug(img_size)

    # Datasets & loaders
    train_ds = DeforestationDataset(train_dir / "images", train_dir / "masks", transform=train_aug, img_size=img_size)
    val_ds = DeforestationDataset(val_dir / "images", val_dir / "masks", transform=val_aug, img_size=img_size)

    nw = 0 if device.type == "mps" else min(4, psutil.cpu_count() or 1)
    pin_memory = device.type != "mps"  # Disable pin_memory for MPS
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=nw, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=nw, 
        pin_memory=pin_memory
    )
    timer.stop('data_setup')

    timer.start('model_setup')
    model = get_model(args, device)
    criterion = DeforestationLoss(
        focal_gamma=args.focal_gamma,
        dice_beta=args.dice_beta,
        edge_weight=args.edge_weight
    )
    
    # Optimizer setup
    optimizer = get_optimizer(model, args.lr)
    scheduler = get_scheduler(optimizer, args.epochs, len(train_loader))

    # Initialize GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    timer.stop('model_setup')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_iou, history = 0.0, defaultdict(list)
    timer.start_training()
    
    # Initial memory snapshot
    memory_tracker.update()
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(Path(args.out_dir))

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, criterion, optimizer, scheduler, scaler, train_loader, device, epoch)
        
        # Validation phase
        val_loss, val_iou = validate(model, criterion, val_loader, device)
        
        # Log metrics
        print(f"\nEpoch {epoch} Statistics:")
        print(f"Train Loss: {tr_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | IoU: {val_iou:.4f}\n")
        
        # Update history and create visualizations
        metrics = {
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_iou": val_iou
        }
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        
        # Generate visualizations
        visualizer.update_history(metrics, epoch)
        visualizer.plot_metrics(epoch)
        
        # Get a batch of validation data for visualization
        val_batch = next(iter(val_loader))
        val_images = val_batch["image"].to(device)
        val_masks = val_batch["mask"].to(device)
        with torch.no_grad():
            val_preds = model(val_images)
            val_preds = torch.sigmoid(val_preds) > 0.5
        
        # Plot sample predictions
        visualizer.plot_predictions(
            val_images.cpu(),
            val_masks.cpu(),
            val_preds.cpu(),
            epoch,
            num_samples=min(6, len(val_images))
        )
        
        # Plot prediction distributions
        visualizer.plot_prediction_distributions(val_preds.cpu(), val_masks.cpu(), epoch)
        
        # Plot gradient flow
        visualizer.plot_gradient_flow(model.named_parameters(), epoch)
        
        # Save epoch summary to CSV
        visualizer.save_epoch_summary(epoch, metrics)
        
        # Save latest model checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            tr_loss, val_loss, val_iou,
            os.path.join(args.out_dir, "latest_model.pt")
        )
        
        # Save periodic checkpoint if requested
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                tr_loss, val_loss, val_iou,
                os.path.join(args.out_dir, f"checkpoint_epoch_{epoch}.pt")
            )
        
        # Save best model checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                tr_loss, val_loss, val_iou,
                os.path.join(args.out_dir, "best_model.pt")
            )
            print(f"New best model saved! (Val IoU: {val_iou:.4f})")
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs,
        tr_loss, val_loss, val_iou,
        os.path.join(args.out_dir, "final_model.pt")
    )
    
    print(f"\nTraining completed successfully! Best Val IoU: {best_iou:.4f}")
    print(f"Model checkpoints saved in: {args.out_dir}")
    print(f"Training visualizations saved in: {args.out_dir}/training_plots/")

if __name__ == "__main__":
    main() 