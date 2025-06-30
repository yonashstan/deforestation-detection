#!/usr/bin/env python3
"""
Visualization script for deforestation segmentation model predictions.
Tests the model on both deforested and forest images and creates comprehensive visualizations.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def get_validation_augmentation():
    """Get validation augmentations"""
    return A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ])

def load_model(model_path, device):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")
    
    # Initialize model (same as in training)
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,  # We'll load our trained weights
        in_channels=3,
        classes=1,
        decoder_attention_type=None,
        decoder_channels=[256, 128, 64, 32, 16],
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Checkpoint info: Epoch {checkpoint.get('epoch', 'N/A')}")
    best_val_iou = checkpoint.get('best_val_iou', None)
    if best_val_iou is not None:
        print(f"Best validation IoU: {best_val_iou:.4f}")
    else:
        print("Best validation IoU: N/A")
    
    return model

def preprocess_image(image_path, transforms):
    """Preprocess a single image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Apply transforms
    transformed = transforms(image=np.array(image))['image']
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(transformed).permute(2, 0, 1).unsqueeze(0).float()
    
    return image_tensor, original_size

def predict_mask(model, image_tensor, device):
    """Generate prediction mask for an image."""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # Apply sigmoid for binary classification
        probabilities = torch.sigmoid(output)
        
        # Convert to numpy
        prob_np = probabilities.squeeze().cpu().numpy()
        
        # Create binary mask (threshold at 0.5)
        mask = (prob_np > 0.5).astype(np.uint8)
        
        return prob_np, mask

def create_visualization(original_image, prediction_mask, confidence_map, title, save_path):
    """Create a comprehensive visualization of the prediction results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Binary prediction mask
    axes[1].imshow(prediction_mask, cmap='RdYlBu_r', alpha=0.8)
    axes[1].set_title('Prediction Mask\n(Red: Deforested, Blue: Forest)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Confidence map
    im = axes[2].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Confidence Map\n(Brighter = Higher Confidence)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Add colorbar for confidence map
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Confidence', rotation=270, labelpad=15)
    
    # Overall title
    fig.suptitle(f'Deforestation Segmentation Results: {title}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

def analyze_predictions(confidence_map, prediction_mask):
    """Analyze prediction statistics."""
    # Calculate statistics
    mean_confidence = np.mean(confidence_map)
    std_confidence = np.std(confidence_map)
    min_confidence = np.min(confidence_map)
    max_confidence = np.max(confidence_map)
    
    # Calculate percentage of deforested pixels
    deforested_pixels = np.sum(prediction_mask)
    total_pixels = prediction_mask.size
    percent_deforested = (deforested_pixels / total_pixels) * 100
    
    # Calculate percentage at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    threshold_stats = {}
    for threshold in thresholds:
        high_conf_pixels = np.sum(confidence_map > threshold)
        threshold_stats[threshold] = (high_conf_pixels / total_pixels) * 100
    
    return {
        'mean_confidence': mean_confidence,
        'std_confidence': std_confidence,
        'min_confidence': min_confidence,
        'max_confidence': max_confidence,
        'percent_deforested': percent_deforested,
        'threshold_stats': threshold_stats
    }

def main():
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    model_path = "models/deforestation_segmentation/deforestation_model_best.pth"
    data_dir = "data/processed"
    output_dir = "outputs/visualization_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(model_path, device)
    
    # Get transforms
    transforms = get_validation_augmentation()
    
    # Test images from both classes
    test_images = [
        ("data/raw/deforested/deforested_0000.png", "Deforested Area"),
        ("data/raw/deforested/deforested_0001.png", "Deforested Area"),
        ("data/raw/forest/forest_0000.png", "Forest Area"),
        ("data/raw/forest/forest_0001.png", "Forest Area"),
    ]
    
    print("\n" + "="*60)
    print("TESTING MODEL ON SAMPLE IMAGES")
    print("="*60)
    
    for image_path, image_type in test_images:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
            
        print(f"\nProcessing: {os.path.basename(image_path)} ({image_type})")
        
        # Preprocess image
        image_tensor, original_size = preprocess_image(image_path, transforms)
        
        # Load original image for visualization
        original_image = np.array(Image.open(image_path).convert('RGB'))
        
        # Generate predictions
        confidence_map, prediction_mask = predict_mask(model, image_tensor, device)
        
        # Analyze predictions
        stats = analyze_predictions(confidence_map, prediction_mask)
        
        # Print statistics
        print(f"  Mean confidence: {stats['mean_confidence']:.4f}")
        print(f"  Confidence std: {stats['std_confidence']:.4f}")
        print(f"  Confidence range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")
        print(f"  Percent deforested: {stats['percent_deforested']:.2f}%")
        
        for threshold, percentage in stats['threshold_stats'].items():
            print(f"  Pixels > {threshold}: {percentage:.2f}%")
        
        # Create visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"prediction_{base_name}.png")
        
        create_visualization(
            original_image, 
            prediction_mask, 
            confidence_map, 
            f"{image_type} - {base_name}",
            save_path
        )
    
    print(f"\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Results saved in: {output_dir}")
    print("Files created:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")

if __name__ == "__main__":
    main() 