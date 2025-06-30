import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.models.simclr import SimCLR, ForestLossDetector
from src.data.dataset import ForestDataset

def train_detector(
    pretrained_path: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device: str = 'cuda',
    log_wandb: bool = True,
    project_name: str = 'forest-loss-detection'
):
    """
    Train forest loss detector using pre-trained SimCLR encoder
    Args:
        pretrained_path: Path to pre-trained SimCLR model
        data_dir: Directory containing labeled forest images
        output_dir: Directory to save model checkpoints
        batch_size: Batch size for training
        epochs: Number of epochs to train
        learning_rate: Learning rate
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
                'weight_decay': weight_decay,
                'epochs': epochs
            }
        )
    
    # Load pre-trained SimCLR model
    checkpoint = torch.load(pretrained_path)
    simclr = SimCLR()
    simclr.load_state_dict(checkpoint['model_state_dict'])
    
    # Create forest loss detector
    model = ForestLossDetector(simclr.encoder)
    model = model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = ForestDataset(Path(data_dir) / 'train')
    val_dataset = ForestDataset(Path(data_dir) / 'val')
    
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
    
    # Initialize optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Initialize gradient scaler
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]') as pbar:
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * train_correct / train_total
                })
                
                if log_wandb:
                    wandb.log({
                        'train_batch_loss': loss.item(),
                        'train_batch_acc': 100. * train_correct / train_total
                    })
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            with tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]') as pbar:
                for images, labels in pbar:
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': 100. * val_correct / val_total
                    })
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        if log_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss/len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss/len(val_loader),
                'val_acc': val_acc
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, output_dir / 'best_model.pth')
    
    if log_wandb:
        wandb.finish()
    
    return model

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train forest loss detector')
    parser.add_argument('--pretrained_path', type=str, required=True, help='Path to pre-trained SimCLR model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing labeled forest images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--project_name', type=str, default='forest-loss-detection', help='Name of wandb project')
    
    args = parser.parse_args()
    
    train_detector(
        pretrained_path=args.pretrained_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        log_wandb=not args.no_wandb,
        project_name=args.project_name
    ) 