import torch
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def get_inference_transforms():
    """Transforms for inference: just normalization and tensor conversion."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def load_model(model_path, device):
    """Loads the pre-trained U-Net model."""
    print(f"Loading model from: {model_path}")
    model = smp.Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def generate_prediction_files(model, image_paths, output_dir, device, threshold=0.5):
    """Runs inference on images and saves the predicted masks as files."""
    transforms = get_inference_transforms()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in tqdm(image_paths, desc="Generating Predictions"):
        image_raw = Image.open(image_path).convert("RGB")
        image_np = np.array(image_raw)
        
        augmented = transforms(image=image_np)
        image_tensor = augmented['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.sigmoid(output)
            
        probs_np = probs.cpu().squeeze().numpy()
        mask = (probs_np > threshold).astype(np.uint8) * 255 # Scale to 0-255 for saving
        
        # Save the mask
        mask_image = Image.fromarray(mask, mode='L') # 'L' for grayscale
        output_file = output_dir / f"pred_{image_path.name}"
        mask_image.save(output_file)

def main():
    # --- Configuration ---
    MODEL_PATH = "models/deforestation_segmentation/deforestation_model_best.pth"
    VAL_IMAGE_DIR = "data/val/images"
    OUTPUT_DIR = "outputs/test_results/predictions"
    PREDICTION_THRESHOLD = 0.98 # Using higher threshold based on previous analysis
    NUM_IMAGES = 5

    # --- Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Get Image Paths ---
    val_image_paths = sorted(list(Path(VAL_IMAGE_DIR).glob("*.png")))[:NUM_IMAGES]
    if not val_image_paths:
        print(f"Error: No images found in {VAL_IMAGE_DIR}")
        return
        
    print(f"Found {len(val_image_paths)} images to process.")

    # --- Generate Predictions ---
    model = load_model(MODEL_PATH, device)
    generate_prediction_files(
        model, 
        val_image_paths, 
        Path(OUTPUT_DIR), 
        device,
        threshold=PREDICTION_THRESHOLD
    )
    
    print(f"\\nPrediction masks saved in: {OUTPUT_DIR}")
    print("You can now run 'scripts/visualize_results.py'")

if __name__ == "__main__":
    main() 