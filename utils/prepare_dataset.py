
import os
import shutil
import random
from pathlib import Path

def prepare_dataset(base_dir):
    """
    Organize dataset:
    1. Move generated labels from preds/labels to labels/train
    2. Split into train/val (80/20)
    """
    base_path = Path(base_dir)
    # Updated path to where YOLOv8 saves predictions
    # pattern: runs/pose/{project}/{name}/labels
    preds_labels_dir = Path("runs/pose/dataset/preds/labels")
    train_labels_dir = base_path / "labels" / "train"
    train_images_dir = base_path / "images" / "train"
    
    val_labels_dir = base_path / "labels" / "val"
    val_images_dir = base_path / "images" / "val"
    
    # Ensure directories exist
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Move predicted labels to training folder
    print("Moving labels...")
    if preds_labels_dir.exists():
        for label_file in preds_labels_dir.glob("*.txt"):
            shutil.move(str(label_file), str(train_labels_dir / label_file.name))
        print(f"Moved labels from {preds_labels_dir} to {train_labels_dir}")
    else:
        print(f"Warning: No labels found in {preds_labels_dir}")
        return

    # Create validation split
    print("Creating validation split...")
    image_files = list(train_images_dir.glob("*.jpg"))
    
    # Filter only images that have labels
    labeled_images = []
    for img_path in image_files:
        label_path = train_labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            labeled_images.append(img_path)
    
    if not labeled_images:
        print("No labeled images found to split.")
        return
        
    # Shuffle and split
    random.shuffle(labeled_images)
    split_idx = int(len(labeled_images) * 0.2)
    val_files = labeled_images[:split_idx]
    
    print(f"Moving {len(val_files)} files to validation set...")
    
    for img_path in val_files:
        label_path = train_labels_dir / (img_path.stem + ".txt")
        
        # Move image
        shutil.move(str(img_path), str(val_images_dir / img_path.name))
        
        # Move label
        shutil.move(str(label_path), str(val_labels_dir / label_path.name))
        
    print("Dataset preparation complete!")
    print(f"Train: {len(list(train_images_dir.glob('*.jpg')))} images")
    print(f"Val: {len(list(val_images_dir.glob('*.jpg')))} images")

if __name__ == "__main__":
    prepare_dataset("dataset")
