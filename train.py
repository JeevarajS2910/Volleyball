"""
Training script for YOLOv8 volleyball detection model.
Trains a standard object detection model with 3 classes:
  0: volleyball
  1: player_team1
  2: player_team2
"""

from ultralytics import YOLO
import argparse
import torch
import os


def train_model(data_yaml, epochs=100, img_size=640, model_path="yolo26n.pt"):
    """Train YOLOv8 detection model."""
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU (CUDA)' if device == 0 else 'CPU'}")

    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration file not found at {data_yaml}")
        return

    print(f"Starting training with dataset: {data_yaml}")
    print(f"Training for {epochs} epochs at image size {img_size}")

    try:
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            device=device,
            patience=50,
            batch=8 if device == 0 else 4,
            workers=0,
            save=True,
            cache=False,
            # Augmentation settings optimized for small ball detection
            mosaic=1.0,
            mixup=0.1,
            scale=0.5,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            name="volleyball_v2",
        )
        print("\nTraining complete!")
        print("Best weights saved to: runs/detect/volleyball_v2/weights/best.pt")
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nPossible causes:")
        print("1. Dataset not found or empty")
        print("2. Invalid data.yaml configuration")
        print("3. GPU out of memory - try reducing batch size")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Volleyball Detection Model")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--model", default="yolo26n.pt", help="Base model (default: yolo26n.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()

    train_model(args.data, args.epochs, args.imgsz, model_path=args.model)


if __name__ == "__main__":
    main()
