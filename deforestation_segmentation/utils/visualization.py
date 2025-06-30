"""Visualization utilities for training monitoring."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd

class TrainingVisualizer:
    """Handles all training visualizations and analytics."""
    
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
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
        
        # Initialize history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'lr': [], 'epoch': []
        }
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def update_history(self, metrics: dict, epoch: int):
        """Update training history with new metrics."""
        for k, v in metrics.items():
            self.history[k].append(v)
        self.history['epoch'].append(epoch)
    
    def plot_metrics(self, epoch: int):
        """Plot training and validation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Training Metrics - Epoch {epoch}', fontsize=16)
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(self.history['epoch'], self.history['train_loss'], label='Train')
        ax.plot(self.history['epoch'], self.history['val_loss'], label='Val')
        ax.set_title('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        
        # Dice plot
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['train_dice'], label='Train')
        ax.plot(self.history['epoch'], self.history['val_dice'], label='Val')
        ax.set_title('Dice Score')
        ax.set_xlabel('Epoch')
        ax.legend()
        
        # IoU plot
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['train_iou'], label='Train')
        ax.plot(self.history['epoch'], self.history['val_iou'], label='Val')
        ax.set_title('IoU Score')
        ax.set_xlabel('Epoch')
        ax.legend()
        
        # Learning rate plot
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], self.history['lr'])
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        
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
            mask = masks[i].cpu().numpy()
            ax.imshow(mask, cmap='RdYlBu')
            ax.set_title('Ground Truth')
            ax.axis('off')
            
            # Predicted mask
            ax = axes[i, 2]
            pred = torch.sigmoid(preds[i]).cpu().numpy()
            ax.imshow(pred, cmap='RdYlBu')
            ax.set_title(f'Prediction (IoU: {self.compute_iou(pred > 0.5, mask):.3f})')
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
    
    @staticmethod
    def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
        """Compute IoU score between prediction and target masks."""
        intersection = np.logical_and(pred, target).sum()
        union = np.logical_or(pred, target).sum()
        return intersection / (union + 1e-6) 