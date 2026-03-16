import os
import sys
from ultralytics import YOLO

# ────────────────────────── CONFIGURATION ──────────────────────────
# Detect paths dynamically
CURRENT_FILE_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(CURRENT_FILE_PATH)
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Path to the trained YOLOv26 model weights
MODEL_PATH = os.path.join(SCRIPT_DIR, "runs", "volleyball_train", "weights", "best.pt")

# Path to the dataset configuration file (shared)
DATA_YAML_PATH = os.path.join(BASE_DIR, "dataset", "data.yaml")
# ───────────────────────────────────────────────────────────────────


def get_mean_precision_recall(metrics_box):
    """Safely extract mean Precision and Recall from a metrics.box object."""
    try:
        if hasattr(metrics_box, "mp") and hasattr(metrics_box, "mr"):
            return float(metrics_box.mp), float(metrics_box.mr)
        if hasattr(metrics_box, "mean_results"):
            res = metrics_box.mean_results()
            return float(res[0]), float(res[1])
        p_arr = metrics_box.p
        r_arr = metrics_box.r
        p = float(sum(p_arr) / len(p_arr)) if len(p_arr) > 0 else 0.0
        r = float(sum(r_arr) / len(r_arr)) if len(r_arr) > 0 else 0.0
        return p, r
    except Exception:
        return 0.0, 0.0


def main():
    print(f"✅ Loading YOLOv26 model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model weights not found at {MODEL_PATH}")
        print("   Please train the YOLOv26 model first using train.py")
        sys.exit(1)

    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ Error: Dataset configuration not found at {DATA_YAML_PATH}")
        sys.exit(1)

    # Initialize YOLO Model
    model = YOLO(MODEL_PATH)

    val_project_dir = os.path.join(SCRIPT_DIR, "runs", "val")
    val_name = "volleyball_val"

    print(f"🚀 Starting Validation for YOLOv26 on dataset: {DATA_YAML_PATH}")

    # Run validation
    metrics = model.val(
        data=DATA_YAML_PATH,     # dataset configuration
        project=val_project_dir, # directory to save results
        name=val_name,           # specific experiment name
        save_json=True,          # Save COCO format JSON results
        save_txt=True,           # Save prediction texts
        save_conf=True,          # Save confidences in prediction texts
        plots=True,              # Generate PR curves, F1 curves, confusion matrix, etc.
        verbose=True             # Print detailed stats per class
    )

    # Retrieve mAP metrics
    try:
        map50 = float(metrics.box.map50)
        map95 = float(metrics.box.map)
    except Exception:
        map50, map95 = 0.0, 0.0

    # Retrieve Precision & Recall
    p, r = get_mean_precision_recall(metrics.box)

    # ── Results summary ──────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("📈 YOLOv26 VALIDATION METRICS OVERVIEW")
    print("━" * 50)
    print(f"Precision:   {p:.4f}")
    print(f"Recall:      {r:.4f}")
    print(f"mAP@50:      {map50:.4f}")
    print(f"mAP@50-95:   {map95:.4f}")
    print("━" * 50)
    print(f"📂 Detailed validation results are saved in: ")
    print(f"   {os.path.join(val_project_dir, val_name)}")
    print("━" * 50)


if __name__ == "__main__":
    main()
