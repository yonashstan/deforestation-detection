#!/usr/bin/env python3
"""
Training script for Forest Loss Detection using MCLC model.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datetime import datetime
import wandb
from typing import Dict, Any, Optional

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from scripts.mclc_segmentation import MCLCSegmentation
from data.dataset import get_dataloaders
from data.augmentation.deforestation_augmentation import DeforestationPattern
from utils.metrics import calculate_metrics
from utils.visualization import plot_training_curves, save_prediction_visualization

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Initialize optimizer and scheduler
        self.optimizer = Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # Initialize loss functions
        self.segmentation_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = nn.CrossEntropyLoss()
        
        # Create output directories
        self.output_dir = Path(config["output_dir"])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.visualization_dir = self.output_dir / "visualizations"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize wandb
        if config.get("use_wandb", False):
            wandb.init(
                project="forest-loss-detection",
                config=config,
                name=f"mclc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            wandb.watch(model)
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_seg_loss = 0
        total_cont_loss = 0
        total_cons_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch["image"].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            pred_masks = outputs["segmentation"]
            contrastive_features = outputs.get("contrastive_features")
            consistency_features = outputs.get("consistency_features")
            
            # Calculate losses
            seg_loss = self.segmentation_loss(pred_masks, batch["mask"].to(self.device))
            cont_loss = self.contrastive_loss(contrastive_features) if contrastive_features is not None else 0
            cons_loss = self.calculate_consistency_loss(consistency_features) if consistency_features is not None else 0
            
            # Combine losses
            loss = (
                self.config["segmentation_weight"] * seg_loss +
                self.config["contrastive_weight"] * cont_loss +
                self.config["consistency_weight"] * cons_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_clip_val"])
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_cont_loss += cont_loss.item() if isinstance(cont_loss, torch.Tensor) else cont_loss
            total_cons_loss += cons_loss.item() if isinstance(cons_loss, torch.Tensor) else cons_loss
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "seg_loss": f"{seg_loss.item():.4f}"
            })
        
        # Calculate average losses
        num_batches = len(self.train_loader)
        return {
            "train_loss": total_loss / num_batches,
            "train_seg_loss": total_seg_loss / num_batches,
            "train_cont_loss": total_cont_loss / num_batches,
            "train_cons_loss": total_cons_loss / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_seg_loss = 0
        metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                pred_masks = outputs["segmentation"]
                
                # Calculate loss
                seg_loss = self.segmentation_loss(pred_masks, masks)
                loss = seg_loss
                
                # Update metrics
                total_loss += loss.item()
                total_seg_loss += seg_loss.item()
                
                # Calculate IoU and other metrics
                pred_masks = torch.sigmoid(pred_masks) > 0.5
                batch_metrics = calculate_metrics(pred_masks, masks)
                metrics.append(batch_metrics)
        
        # Calculate average metrics
        num_batches = len(self.val_loader)
        avg_metrics = {
            "val_loss": total_loss / num_batches,
            "val_seg_loss": total_seg_loss / num_batches
        }
        
        # Add average of other metrics
        for metric_name in metrics[0].keys():
            avg_metrics[f"val_{metric_name}"] = np.mean([m[metric_name] for m in metrics])
        
        return avg_metrics
    
    def calculate_consistency_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate local consistency loss"""
        if features is None:
            return torch.tensor(0.0, device=self.device)
        
        # Calculate consistency between neighboring pixels
        consistency_loss = 0
        for i in range(features.size(1)):
            for j in range(features.size(2)):
                if i < features.size(1) - 1:
                    consistency_loss += torch.mean((features[:, i, j] - features[:, i+1, j])**2)
                if j < features.size(2) - 1:
                    consistency_loss += torch.mean((features[:, i, j] - features[:, i, j+1])**2)
        
        return consistency_loss / (features.size(1) * features.size(2))
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def train(self, num_epochs: int):
        """Train the model for specified number of epochs"""
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_metrics["val_loss"])
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            logging.info(f"Metrics: {metrics}")
            
            if self.config.get("use_wandb", False):
                wandb.log(metrics)
            
            # Save checkpoint
            is_best = val_metrics["val_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["val_loss"]
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Save visualization
            if epoch % self.config.get("visualization_interval", 5) == 0:
                self.save_visualization()
    
    def save_visualization(self):
        """Save prediction visualization"""
        self.model.eval()
        with torch.no_grad():
            batch = next(iter(self.val_loader))
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            outputs = self.model(images)
            pred_masks = torch.sigmoid(outputs["segmentation"]) > 0.5
            
            save_prediction_visualization(
                images[0].cpu(),
                masks[0].cpu(),
                pred_masks[0].cpu(),
                self.visualization_dir / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

def main():
    # Configuration
    config = {
        "data_dir": "data",
        "output_dir": "outputs",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "batch_size": 8,
        "num_epochs": 50,
        "gradient_clip_val": 1.0,
        "segmentation_weight": 1.0,
        "contrastive_weight": 0.5,
        "consistency_weight": 0.3,
        "use_wandb": True,
        "visualization_interval": 5
    }
    
    # Create deforestation pattern config
    pattern_config = DeforestationPattern(
        min_size=(50, 50),
        max_size=(200, 200),
        num_patterns=3
    )
    
    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        use_deforestation_aug=True,
        pattern_config=pattern_config
    )
    
    # Create model
    model = MCLCSegmentation(
        backbone="efficientnet-b0",
        in_channels=3,
        out_channels=1,
        features=[32, 64, 128, 256],
        use_attention=True,
        use_contrastive=True,
        use_consistency=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train model
    trainer.train(num_epochs=config["num_epochs"])

if __name__ == "__main__":
    main() 