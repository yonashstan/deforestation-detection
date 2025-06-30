import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import logging
from datetime import datetime
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import DeforestationDataset

class FocalLovaszLoss(nn.Module):
    """Combined Focal and Lovasz loss for better handling of class imbalance and boundaries"""
    def __init__(self, alpha=0.5, gamma=2, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.lovasz = smp.losses.LovaszLoss(mode='binary')
        
    def forward(self, y_pred, y_true):
        # Move tensors to CPU for loss calculation
        if y_pred.device.type == 'mps':
            y_pred = y_pred.cpu()
            y_true = y_true.cpu()
            if self.pos_weight is not None:
                self.pos_weight = self.pos_weight.cpu()
        
        # Binary cross entropy with logits and class weights
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            y_pred, y_true, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Focal loss term
        prob = torch.sigmoid(y_pred)
        p_t = prob * y_true + (1 - prob) * (1 - y_true)
        focal_loss = ((1 - p_t) ** self.gamma * bce_loss).mean()
        
        # Lovasz loss term
        lovasz_loss = self.lovasz(y_pred, y_true)
        
        # L2 regularization term
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., device=y_pred.device)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        
        # Combine losses
        total_loss = self.alpha * focal_loss + (1 - self.alpha) * lovasz_loss + l2_lambda * l2_reg
        
        # Move result back to original device
        if y_pred.device.type == 'mps':
            total_loss = total_loss.to('mps')
        
        return total_loss

def get_training_augmentation():
    """Get training augmentations optimized for aerial imagery"""
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

def get_validation_augmentation():
    """Get validation augmentations"""
    return A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

def calculate_iou(pred_mask, true_mask):
    """Calculate IoU score for binary segmentation"""
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_class_weights(dataset):
    """Calculate class weights based on class distribution"""
    total_pixels = 0
    forest_pixels = 0
    
    for _, mask in tqdm(dataset, desc="Calculating class weights"):
        total_pixels += mask.numel()
        forest_pixels += (mask == 0).sum().item()
    
    deforested_pixels = total_pixels - forest_pixels
    
    # Calculate weights (inverse of class frequency)
    forest_weight = total_pixels / (2 * forest_pixels)
    deforested_weight = total_pixels / (2 * deforested_pixels)
    
    print(f"Class distribution:")
    print(f"Forest pixels: {forest_pixels} ({forest_pixels/total_pixels*100:.2f}%)")
    print(f"Deforested pixels: {deforested_pixels} ({deforested_pixels/total_pixels*100:.2f}%)")
    print(f"Weights - Forest: {forest_weight:.4f}, Deforested: {deforested_weight:.4f}")
    
    return torch.tensor([deforested_weight])

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    save_dir,
    patience=10
):
    """Train the model with early stopping and learning rate scheduling"""
    best_val_iou = 0
    patience_counter = 0
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_iou = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Step the scheduler every batch for OneCycleLR
            
            # Calculate IoU
            pred_masks = (torch.sigmoid(outputs) > 0.5).float()
            batch_iou = calculate_iou(pred_masks, masks)
            
            train_loss += loss.item()
            train_iou += batch_iou.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou.item():.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_iou = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                pred_masks = (torch.sigmoid(outputs) > 0.5).float()
                batch_iou = calculate_iou(pred_masks, masks)
                
                val_loss += loss.item()
                val_iou += batch_iou.item()
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            
            # Save model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'history': history,
                'metrics': {
                    'train_loss': train_loss,
                    'train_iou': train_iou,
                    'val_loss': val_loss,
                    'val_iou': val_iou
                }
            }
            
            torch.save(checkpoint, save_dir / 'deforestation_model_best.pth')
            print(f'Saved best model with Val IoU: {val_iou:.4f}')
            
            # Plot and save learning curves
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(history['train_iou'], label='Train IoU')
            plt.plot(history['val_iou'], label='Val IoU')
            plt.title('IoU History')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_dir / 'training_curves.png')
            plt.close()
        else:
            patience_counter += 1
        
        # Save latest model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'history': history,
            'metrics': {
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou
            }
        }
        torch.save(checkpoint, save_dir / 'deforestation_model_latest.pth')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    return history

def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS backend")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA backend")
    else:
        device = torch.device("cpu")
        print("Using CPU backend")
    
    # Data directories - using synthetic v2 dataset
    data_dir = Path("data/synth_v2")
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    # Split synthetic data if not already split
    if not (train_dir / "images").exists():
        print("\nSplitting synthetic dataset into train/val...")
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        (train_dir / "images").mkdir(exist_ok=True)
        (train_dir / "masks").mkdir(exist_ok=True)
        (val_dir / "images").mkdir(exist_ok=True)
        (val_dir / "masks").mkdir(exist_ok=True)
        
        # Get all images and split
        all_images = sorted(list((data_dir / "images").glob("*.png")))
        random.seed(42)
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.8)  # 80/20 split
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]
        
        # Copy files to respective directories
        for subset, images in [("train", train_images), ("val", val_images)]:
            target_dir = train_dir if subset == "train" else val_dir
            for img_path in images:
                mask_path = data_dir / "masks" / img_path.name
                shutil.copy(img_path, target_dir / "images" / img_path.name)
                shutil.copy(mask_path, target_dir / "masks" / mask_path.name)
        print(f"Split dataset: {len(train_images)} train, {len(val_images)} val")
    
    # Create datasets with optimized augmentations
    train_dataset = DeforestationDataset(
        images_dir=train_dir / "images",
        masks_dir=train_dir / "masks",
        transform=get_training_augmentation(),
        img_size=(256, 256)  # Further reduced size for much faster training
    )
    
    val_dataset = DeforestationDataset(
        images_dir=val_dir / "images",
        masks_dir=val_dir / "masks",
        transform=get_validation_augmentation(),
        img_size=(256, 256)  # Same size as training
    )
    
    # Calculate class weights for balanced training
    print("\nCalculating class weights...")
    pos_weight = calculate_class_weights(train_dataset)
    pos_weight = pos_weight.to(device)
    
    # Optimized batch size and workers for M1
    batch_size = 24 if device.type == "mps" else 48  # Balanced batch size for stability
    num_workers = 0 if device.type == "mps" else 4   # Disabled workers on MPS for better performance
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model with optimized architecture
    model = smp.Unet(
        encoder_name="efficientnet-b0",  # Lightweight but powerful
        encoder_weights="imagenet",      # Pre-trained weights
        in_channels=3,
        classes=1,
        decoder_attention_type=None,     # Removed attention for faster training
        decoder_channels=[256, 128, 64, 32, 16],  # Standard decoder structure
    )
    
    # Initialize weights and add dropout
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            # Add light dropout
            module.register_forward_hook(lambda m, _, output: nn.functional.dropout(
                output, p=0.05, training=m.training
            ))
    
    model = model.to(device)
    
    # Loss function with class weights
    criterion = FocalLovaszLoss(alpha=0.5, pos_weight=pos_weight)
    
    # Optimizer with gradient clipping and weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999)  # Default betas work well
    )
    
    # Learning rate scheduler with warmup
    num_epochs = 10  # Reduced epochs, focusing on early convergence
    warmup_epochs = 2  # Shorter warmup for fewer epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-3,  # Balanced learning rate for stability
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/num_epochs,
        div_factor=25,
        final_div_factor=1000
    )
    
    # Train model with early stopping
    save_dir = Path("models/deforestation_segmentation_synth")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_dir=save_dir,
        patience=7   # Reduced patience since we're using fewer epochs
    )

if __name__ == "__main__":
    main() 