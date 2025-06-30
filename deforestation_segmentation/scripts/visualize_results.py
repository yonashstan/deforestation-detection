import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay a colored mask on an image"""
    mask = mask.astype(np.bool_)
    overlay = image.copy()
    overlay[mask] = overlay[mask] * (1 - alpha) + np.array(color) * alpha
    return overlay

def visualize_prediction(image_path, mask_path, pred_path, output_path):
    """Visualize original image, ground truth, prediction, and overlay"""
    # Read images
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = mask > 127
    
    pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    pred = pred > 127
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 4, figure=fig)
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Ground truth mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Ground Truth')
    ax2.axis('off')
    
    # Predicted mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(pred, cmap='gray')
    ax3.set_title('Prediction')
    ax3.axis('off')
    
    # Overlay prediction on image
    ax4 = fig.add_subplot(gs[0, 3])
    overlay = overlay_mask(image, pred, color=(255, 0, 0))  # Red for predictions
    overlay = overlay_mask(overlay, mask, color=(0, 255, 0), alpha=0.3)  # Green for ground truth
    ax4.imshow(overlay)
    ax4.set_title('Overlay (Green: GT, Red: Pred)')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Paths
    data_dir = Path("data/val")
    results_dir = Path("outputs/test_results")
    vis_dir = results_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Get a few sample images
    image_paths = sorted(list((data_dir / "images").glob("*.png")))[:5]  # First 5 images
    
    for image_path in image_paths:
        # Get corresponding mask and prediction paths
        mask_path = data_dir / "masks" / image_path.name
        pred_path = results_dir / "predictions" / f"pred_{image_path.name}"
        
        if not all(p.exists() for p in [image_path, mask_path, pred_path]):
            print(f"Skipping {image_path.name} - missing files")
            continue
        
        output_path = vis_dir / f"vis_{image_path.stem}.png"
        print(f"Creating visualization for {image_path.name}")
        
        visualize_prediction(image_path, mask_path, pred_path, output_path)
    
    print(f"\nVisualizations saved in {vis_dir}")

if __name__ == "__main__":
    main() 