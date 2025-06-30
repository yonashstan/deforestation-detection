import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.models.simclr import SimCLR
from src.data.dataset import ForestDataset, ForestPatchDataset

def train_simclr(
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 3e-4,
    temperature: float = 0.07,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str = 'cuda',
    log_wandb: bool = True,
    project_name: str = 'forest-loss-detection'
):
    """
    Train SimCLR model on forest imagery
    Args:
        data_dir: Directory containing forest images
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate
        temperature: Temperature parameter for contrastive loss
        weight_decay: Weight decay for optimizer
        num_workers: Number of workers for data loading
        device: Device to train on ('cuda' or 'cpu')
        log_wandb: Whether to log metrics to wandb
        project_name: Name of wandb project
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if log_wandb:
        wandb.init(
            project=project_name,
            config={
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'temperature': temperature,
                'weight_decay': weight_decay,
                'epochs': epochs
            }
        )
    
    # Create dataset and dataloader
    dataset = ForestPatchDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Initialize model
    model = SimCLR(temperature=temperature)
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
            for batch in pbar:
                view1, view2 = [v.to(device) for v in batch['views']]
                
                optimizer.zero_grad()
                
                # Mixed precision training
                with autocast():
                    loss = model(view1, view2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                if log_wandb:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss for epoch
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')
        
        if log_wandb:
            wandb.log({
                'epoch': epoch,
                'avg_loss': avg_loss
            })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, output_dir / 'best_model.pth')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pth')
    
    if log_wandb:
        wandb.finish()
    
    return model

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SimCLR model on forest imagery')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing forest images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature parameter for contrastive loss')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--project_name', type=str, default='forest-loss-detection', help='Name of wandb project')
    
    args = parser.parse_args()
    
    train_simclr(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        log_wandb=not args.no_wandb,
        project_name=args.project_name
    ) 