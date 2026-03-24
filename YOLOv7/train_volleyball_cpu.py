import os
import subprocess
import sys
import requests

# ────────────────────────── CONFIGURATION ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_DIR   = SCRIPT_DIR
DATA_YAML  = os.path.join(YOLO_DIR, "data", "volleyball.yaml")
WEIGHTS    = os.path.join(YOLO_DIR, "yolov7.pt")
CONFIG     = os.path.join(YOLO_DIR, "cfg", "training", "yolov7.yaml")

# Hyperparameters (CPU-friendly)
HYP        = os.path.join(YOLO_DIR, "data", "hyp.scratch.p5.yaml")
BATCH_SIZE = 2  # Small batch for CPU
EPOCHS     = 100
IMG_SIZE   = 640
# ───────────────────────────────────────────────────────────────────

def download_weights():
    if not os.path.exists(WEIGHTS):
        print("Downloading official YOLOv7 pre-trained weights...")
        url = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
        response = requests.get(url, stream=True)
        with open(WEIGHTS, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Weights downloaded successfully.")

def run_training():
    print("Starting Official YOLOv7 Training (CPU Mode)...")
    
    # Construct command (forced CPU)
    cmd = [
        sys.executable, "train.py",
        "--workers", "0",
        "--device", "cpu",
        "--batch-size", str(BATCH_SIZE),
        "--data", DATA_YAML,
        "--img", str(IMG_SIZE), str(IMG_SIZE),
        "--cfg", CONFIG,
        "--weights", WEIGHTS,
        "--name", "yolov7-volleyball-cpu",
        "--hyp", HYP,
        "--epochs", str(EPOCHS)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=YOLO_DIR)

if __name__ == "__main__":
    download_weights()
    run_training()
