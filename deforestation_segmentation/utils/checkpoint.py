import torch
from pathlib import Path
import json
import time
from datetime import datetime

class CheckpointHandler:
    """Handle model checkpoints and training state."""
    
    def __init__(self, model_dir, model_name="deforestation_model"):
        """
        Initialize checkpoint handler.
        
        Args:
            model_dir (str or Path): Directory to save checkpoints
            model_name (str): Name of the model
        """
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Path for best model
        self.best_model_path = self.model_dir / f"{model_name}_best.pth"
        # Path for latest model
        self.latest_model_path = self.model_dir / f"{model_name}_latest.pth"
        # Path for training state
        self.training_state_path = self.model_dir / "training_state.json"
        
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """
        Save model checkpoint and training state.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch (int): Current epoch
            metrics (dict): Dictionary of metrics
            is_best (bool): Whether this is the best model so far
        """
        # Save model state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.latest_model_path)
        
        # If this is the best model, save it separately
        if is_best:
            torch.save(checkpoint, self.best_model_path)
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.training_state_path, 'w') as f:
            json.dump(training_state, f, indent=4)
    
    def load_checkpoint(self, model, optimizer=None, device=None):
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer (optional): PyTorch optimizer
            device (optional): Device to load the model to
        
        Returns:
            dict: Checkpoint data including epoch and metrics
        """
        # Try to load the best model first, fall back to latest
        checkpoint_path = self.best_model_path if self.best_model_path.exists() else self.latest_model_path
        
        if not checkpoint_path.exists():
            return None
        
        # Load checkpoint
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', None)
        }
    
    def get_training_state(self):
        """Get the current training state."""
        if not self.training_state_path.exists():
            return None
        
        with open(self.training_state_path, 'r') as f:
            return json.load(f)
    
    def clean_old_checkpoints(self, keep_best=True, keep_latest=True):
        """
        Clean up old checkpoints.
        
        Args:
            keep_best (bool): Whether to keep the best model
            keep_latest (bool): Whether to keep the latest model
        """
        for checkpoint in self.model_dir.glob("*.pth"):
            if keep_best and checkpoint == self.best_model_path:
                continue
            if keep_latest and checkpoint == self.latest_model_path:
                continue
            checkpoint.unlink() 