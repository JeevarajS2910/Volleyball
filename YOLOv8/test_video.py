import os
import sys
from ultralytics import YOLO

# ────────────────────────── CONFIGURATION ──────────────────────────
# Detect paths dynamically
CURRENT_FILE_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(CURRENT_FILE_PATH)
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Path to the trained YOLOv8 model
MODEL_PATH = os.path.join(SCRIPT_DIR, "runs", "volleyball_train", "weights", "best.pt")

# Path to the dataset configuration file
DATA_YAML_PATH = os.path.join(BASE_DIR, "dataset", "data.yaml")
# ───────────────────────────────────────────────────────────────────

def main():
    print(f"✅ Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model weights not found at {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ Error: Dataset configuration not found at {DATA_YAML_PATH}")
        sys.exit(1)

    # Initialize YOLO Model
    model = YOLO(MODEL_PATH)
    
    # Optional: If you want to run inference on a specific directory of images and save predictions
    # you can use model.predict(source="path/to/images", save=True)
    # However, to compute mAP, Precision, and Recall, we use the validation mode on an annotated dataset.
    
    val_project_dir = os.path.join(SCRIPT_DIR, "runs", "val")
    val_name = "volleyball_test"

    print(f"🚀 Starting validation (testing) and metrics calculation on dataset: {DATA_YAML_PATH}")
    # Evaluate the model on the validation set
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
    
    # Retrieve metrics (Ultralytics v8 dynamic attributes)
    map50 = metrics.box.map50
    map95 = metrics.box.map
    
    # By default, metrics.box.p and metrics.box.r are arrays mapping to each class.
    # We can get the mean values if available:
    try:
        if hasattr(metrics.box, 'mean_results'):
            # internal structure can sometimes have mean_results (precision, recall, mAP50, mAP50-95)
            # Typically 0: p, 1: r, 2: mAP50, 3: mAP
            p = metrics.box.mean_results()[0]
            r = metrics.box.mean_results()[1]
        elif hasattr(metrics.box, 'mp') and hasattr(metrics.box, 'mr'):
            # Using specific mean precision/recall properties
            p = metrics.box.mp
            r = metrics.box.mr
        else:
            p = sum(metrics.box.p) / len(metrics.box.p) if len(metrics.box.p) > 0 else 0
            r = sum(metrics.box.r) / len(metrics.box.r) if len(metrics.box.r) > 0 else 0
    except Exception:
        p = 0
        r = 0

    print("\n" + "━"*50)
    print("📈 VALIDATION METRICS OVERVIEW")
    print("━"*50)
    print(f"Precision:   {p:.4f}")
    print(f"Recall:      {r:.4f}")
    print(f"mAP@50:      {map50:.4f}")
    print(f"mAP@50-95:   {map95:.4f}")
    print("━"*50)
    print(f"📂 Detailed test results, images, matrices, and PR curves are saved in: ")
    print(f"   {os.path.join(val_project_dir, val_name)}")
    print("━"*50)

if __name__ == "__main__":
    main()
