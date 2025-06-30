#!/usr/bin/env python3
"""
Evaluation script for Forest Loss Detection model.
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from models.mclc_segmentation import MCLCSegmentation
from src.data.dataset import get_dataloaders
from utils.metrics import calculate_metrics
from utils.visualization import save_prediction_visualization

class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs/evaluation"
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.prediction_dir = self.output_dir / "predictions"
        self.metrics_dir = self.output_dir / "metrics"
        self.confusion_matrix_dir = self.output_dir / "confusion_matrices"
        
        for dir_path in [self.prediction_dir, self.metrics_dir, self.confusion_matrix_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        self.model.eval()
        all_metrics = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                pred_masks = torch.sigmoid(outputs["segmentation"]) > 0.5
                
                # Calculate metrics
                batch_metrics = calculate_metrics(pred_masks, masks)
                all_metrics.append(batch_metrics)
                
                # Store predictions and targets for confusion matrix
                all_preds.extend(pred_masks.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
                
                # Save visualizations for some samples
                if batch_idx < 10:  # Save first 10 batches
                    for i in range(min(4, len(images))):  # Save up to 4 images per batch
                        save_prediction_visualization(
                            images[i].cpu(),
                            masks[i].cpu(),
                            pred_masks[i].cpu(),
                            self.prediction_dir / f"prediction_batch{batch_idx}_sample{i}.png"
                        )
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name in all_metrics[0].keys():
            avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics])
        
        # Generate confusion matrix
        self.generate_confusion_matrix(all_preds, all_targets)
        
        # Save metrics
        self.save_metrics(avg_metrics)
        
        return avg_metrics
    
    def generate_confusion_matrix(self, preds: List[int], targets: List[int]):
        """Generate and save confusion matrix"""
        cm = confusion_matrix(targets, preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plt.savefig(self.confusion_matrix_dir / 'confusion_matrix.png')
        plt.close()
        
        # Save numerical values
        np.save(self.confusion_matrix_dir / 'confusion_matrix.npy', cm)
        
        # Generate classification report
        report = classification_report(targets, preds, output_dict=True)
        with open(self.confusion_matrix_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=4)
    
    def save_metrics(self, metrics: Dict[str, float]):
        """Save evaluation metrics"""
        # Save as JSON
        with open(self.metrics_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save as text
        with open(self.metrics_dir / 'metrics.txt', 'w') as f:
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        
        # Log metrics
        logging.info("Evaluation Metrics:")
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")

def main():
    # Configuration
    config = {
        "checkpoint_path": "outputs/checkpoints/best_model.pt",
        "data_dir": "prepared_data/data",
        "output_dir": "outputs/evaluation",
        "batch_size": 8,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
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
    
    # Load checkpoint
    checkpoint = torch.load(config["checkpoint_path"], map_location=config["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Get test dataloader
    _, _, test_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        use_deforestation_aug=False  # No augmentation for evaluation
    )
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=config["device"],
        output_dir=config["output_dir"]
    )
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print results
    print("\nEvaluation Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main() 