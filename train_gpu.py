"""
Training script for YOLO volleyball detection model.
Trains with GPU and generates all metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95)
with charts/plots saved automatically.

Classes:
  0: volleyball
  1: player_team1
  2: player_team2
"""

from ultralytics import YOLO
import torch
import os
import sys


def main():
    # Enable synchronous CUDA errors for better debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # ─── Configuration ───────────────────────────────────────────
    MODEL_PATH = r"C:\volley ball\yolo26n.pt"
    DATA_YAML = r"C:\volley ball\dataset\data.yaml"
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 4  # Reduced for RTX 3050 (4GB VRAM); use -1 for auto-batch
    PROJECT_DIR = r"C:\volley ball\runs\detect"
    RUN_NAME = "volleyball_gpu_train"

    # ─── GPU Check ───────────────────────────────────────────────
    print("=" * 60)
    print("YOLO Volleyball Detection - GPU Training")
    print("=" * 60)

    if torch.cuda.is_available():
        device = 0
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU detected: {gpu_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("❌ No GPU detected! Training will be very slow on CPU.")
        print("   Make sure CUDA-enabled PyTorch is installed.")
        device = 'cpu'

    # ─── Validate paths ──────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(DATA_YAML):
        print(f"❌ Dataset config not found: {DATA_YAML}")
        sys.exit(1)

    print(f"\n📦 Model:   {MODEL_PATH}")
    print(f"📂 Dataset: {DATA_YAML}")
    print(f"🔄 Epochs:  {EPOCHS}")
    print(f"📐 ImgSize: {IMG_SIZE}")
    print(f"📊 Batch:   {BATCH_SIZE}")
    print(f"💾 Output:  {PROJECT_DIR}/{RUN_NAME}")
    print("=" * 60)

    # ─── Load model ──────────────────────────────────────────────
    print("\nLoading model...")
    model = YOLO(MODEL_PATH)

    # ─── Train ───────────────────────────────────────────────────
    print("Starting training...\n")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
        batch=BATCH_SIZE,
        workers=0,
        patience=50,

        # Save settings
        save=True,
        save_period=10,          # Save checkpoint every 10 epochs
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # ── Plots & Metrics (generates all charts) ──
        plots=True,              # Generate all training plots/charts

        # Augmentation settings for volleyball detection
        mosaic=1.0,
        mixup=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Don't cache (save RAM)
        cache=False,

        # Verbose output
        verbose=True,
    )

    # ─── Print Final Metrics ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - FINAL METRICS")
    print("=" * 60)

    # The results object contains all metrics
    if results and hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n{'Metric':<30} {'Value':>10}")
        print("-" * 42)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:<30} {value:>10.4f}")
            else:
                print(f"{key:<30} {str(value):>10}")

    output_dir = os.path.join(PROJECT_DIR, RUN_NAME)
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"\n📊 Charts generated (check the output folder):")
    print(f"   • results.png        - Precision, Recall, mAP@0.5, mAP@0.5:0.95 curves over epochs")
    print(f"   • confusion_matrix.png")
    print(f"   • F1_curve.png")
    print(f"   • P_curve.png        - Precision curve")
    print(f"   • R_curve.png        - Recall curve")
    print(f"   • PR_curve.png       - Precision-Recall curve")
    print(f"   • labels.jpg         - Label distribution")
    print(f"   • labels_correlogram.jpg")
    print(f"\n🏆 Best weights: {output_dir}/weights/best.pt")
    print(f"   Last weights: {output_dir}/weights/last.pt")


if __name__ == "__main__":
    main()