"""
YOLOv8 Volleyball Detection — GPU Training with Auto-Logging & Auto-Save
=========================================================================
• Trains YOLOv8 on the volleyball dataset using GPU
• Saves full training log to a timestamped .log file
• Auto-saves checkpoints every 10 epochs + best/last weights
• Auto-saves all result charts (P/R curves, confusion matrix, etc.)
• Prints a summary of final metrics at the end

Classes:
  0: volleyball
  1: player_team1
  2: player_team2
"""

import os
import sys
import logging
from datetime import datetime

import torch
from ultralytics import YOLO


# ────────────────────────── CONFIGURATION ──────────────────────────
# Detect project root dynamically (parent of YOLOv8 directory)
CURRENT_FILE_PATH = os.path.abspath(__file__)
SCRIPT_DIR        = os.path.dirname(CURRENT_FILE_PATH)
BASE_DIR          = os.path.dirname(SCRIPT_DIR)

MODEL_PATH        = os.path.join(BASE_DIR, "yolo26n.pt")
DATA_YAML         = os.path.join(BASE_DIR, "dataset", "data.yaml")

EPOCHS            = 100
IMG_SIZE          = 640
BATCH_SIZE        = 4          # safe for RTX 3050 4 GB VRAM
PATIENCE          = 50         # early-stopping patience
SAVE_PERIOD       = 10         # checkpoint every N epochs

PROJECT_DIR       = os.path.join(SCRIPT_DIR, "runs")
RUN_NAME          = "volleyball_train"
LOG_DIR           = os.path.join(SCRIPT_DIR, "logs")
# ───────────────────────────────────────────────────────────────────


def setup_logger():
    """Create a logger that writes to both console AND a .log file."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"training_{timestamp}.log")

    logger = logging.getLogger("yolo_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    logger.info(f"Log file → {log_file}")
    return logger, log_file


def check_gpu(logger):
    """Detect GPU and return device id."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"✅ GPU detected: {name}  ({mem:.1f} GB VRAM)")
        return 0
    else:
        logger.info("❌ No GPU found — falling back to CPU (training will be slow).")
        return "cpu"


def validate_paths(logger):
    """Make sure model weights and data.yaml exist."""
    ok = True
    if not os.path.isfile(MODEL_PATH):
        logger.info(f"❌ Model not found: {MODEL_PATH}")
        ok = False
    if not os.path.isfile(DATA_YAML):
        logger.info(f"❌ Dataset config not found: {DATA_YAML}")
        ok = False
    return ok


def update_data_yaml(data_yaml_path, dataset_dir, logger):
    """Update data.yaml with absolute path to dataset directory."""
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Update the path line with absolute dataset directory
        updated_lines = []
        for line in lines:
            if line.strip().startswith('path:'):
                updated_lines.append(f'path: {dataset_dir}\n')
            else:
                updated_lines.append(line)
        
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        
        logger.info(f"✅ Updated data.yaml with path: {dataset_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Could not update data.yaml: {e}")



def train(logger):
    """Run the full training pipeline."""
    # ── GPU ──
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = check_gpu(logger)

    # ── Paths ──
    if not validate_paths(logger):
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  YOLO Volleyball Detection — GPU Training")
    logger.info("=" * 60)
    logger.info(f"  Model      : {MODEL_PATH}")
    logger.info(f"  Dataset    : {DATA_YAML}")
    logger.info(f"  Epochs     : {EPOCHS}")
    logger.info(f"  Image size : {IMG_SIZE}")
    logger.info(f"  Batch size : {BATCH_SIZE}")
    logger.info(f"  Patience   : {PATIENCE}")
    logger.info(f"  Output     : {PROJECT_DIR}/{RUN_NAME}")
    logger.info("=" * 60)

    # ── Update data.yaml with absolute path ──
    dataset_dir = os.path.dirname(DATA_YAML)
    update_data_yaml(DATA_YAML, dataset_dir, logger)

    # ── Load model ──
    logger.info("Loading model …")
    model = YOLO(MODEL_PATH)

    # ── Train ──
    logger.info("Starting training …\n")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
        batch=BATCH_SIZE,
        workers=0,
        patience=PATIENCE,

        # Auto-save settings
        save=True,
        save_period=SAVE_PERIOD,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # Generate all plots / charts automatically
        plots=True,

        # Augmentation (tuned for volleyball + players)
        mosaic=1.0,
        mixup=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        cache=False,
        verbose=True,
    )

    # ── Final metrics ──
    output_dir = os.path.join(PROJECT_DIR, RUN_NAME)
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE — FINAL METRICS")
    logger.info("=" * 60)

    if results and hasattr(results, "results_dict"):
        metrics = results.results_dict
        logger.info(f"\n{'Metric':<35} {'Value':>10}")
        logger.info("-" * 47)
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"{key:<35} {value:>10.4f}")
            else:
                logger.info(f"{key:<35} {str(value):>10}")

    logger.info(f"\n📁 Results saved to : {output_dir}")
    logger.info("📊 Auto-saved charts:")
    for chart in [
        "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "F1_curve.png", "P_curve.png", "R_curve.png", "PR_curve.png",
        "labels.jpg", "labels_correlogram.jpg",
    ]:
        path = os.path.join(output_dir, chart)
        status = "✅" if os.path.isfile(path) else "—"
        logger.info(f"   {status} {chart}")

    logger.info(f"\n🏆 Best weights : {output_dir}/weights/best.pt")
    logger.info(f"   Last weights : {output_dir}/weights/last.pt")
    logger.info("=" * 60)

    return results


# ──────────────────────────────── MAIN ────────────────────────────────
if __name__ == "__main__":
    logger, log_file = setup_logger()
    try:
        train(logger)
    except Exception as exc:
        logger.exception(f"Training failed: {exc}")
        sys.exit(1)
    finally:
        logger.info(f"\n📝 Full training log saved to: {log_file}")
