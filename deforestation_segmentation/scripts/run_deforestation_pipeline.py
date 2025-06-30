from pathlib import Path
import sys
import torch
sys.path.append('.')  # Add project root to path

from scripts.prepare_data import prepare_dataset
from training.deforestation_trainer import DeforestationTrainer

def main():
    # Check available memory and set parameters accordingly
    batch_size = 4  # Conservative batch size for 16GB RAM
    num_workers = 2  # Reduced workers for M1
    
    # 1. Data Preparation
    print("Step 1: Preparing Dataset...")
    raw_forest_dir = Path("data/raw/forest")
    raw_deforested_dir = Path("data/raw/deforested")
    output_dir = Path("data")
    
    train_forest, train_deforested, val_forest, val_deforested = prepare_dataset(
        raw_forest_dir=raw_forest_dir,
        raw_deforested_dir=raw_deforested_dir,
        output_dir=output_dir,
        val_split=0.2,  # 20% validation split
        target_size=(256, 256)
    )
    
    print(f"\nDataset Statistics:")
    print(f"Training set: {train_forest} forest images, {train_deforested} deforested images")
    print(f"Validation set: {val_forest} forest images, {val_deforested} deforested images")
    
    # Print device information
    if torch.backends.mps.is_available():
        device_name = "MPS (Apple Silicon)"
    elif torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_name = "CPU"
    print(f"\nUsing device: {device_name}")
    
    # 2. Model Training
    print("\nStep 2: Training Model...")
    trainer = DeforestationTrainer(
        train_images_dir=output_dir / "train" / "images",
        train_masks_dir=output_dir / "train" / "masks",
        val_images_dir=output_dir / "val" / "images",
        val_masks_dir=output_dir / "val" / "masks",
        model_save_dir=Path("models") / "deforestation_segmentation",
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Start training with adjusted parameters for M1
    trainer.train(
        epochs=15,
        initial_lr=5e-4  # Slightly lower learning rate for stability
    )
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 