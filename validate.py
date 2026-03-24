"""
validate.py – Unified validation script for YOLOv7, YOLOv8, YOLOv11, YOLOv26.

For each model it:
  1. Runs model.val() on the shared dataset
  2. Collects Precision, Recall, mAP@50, mAP@50-95
  3. Saves per-model results & plots inside each model's runs/val/ folder
  4. Writes a combined summary to  validate_results/summary.csv
  5. Saves a side-by-side comparison bar chart to validate_results/comparison.png
"""

import os
import csv
import sys
import traceback

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────── CONFIGURATION ────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # c:\volley ball

# Shared dataset config
DATA_YAML = os.path.join(BASE_DIR, "dataset", "data.yaml")

# Model definitions  {label: path-to-best.pt}
MODELS = {
    "YOLOv7":  os.path.join(BASE_DIR, "YOLOv7",  "runs", "volleyball_train", "weights", "best.pt"),
    "YOLOv8":  os.path.join(BASE_DIR, "YOLOv8",  "runs", "volleyball_train", "weights", "best.pt"),
    "YOLOv11": os.path.join(BASE_DIR, "YOLOv11", "runs", "volleyball_train", "weights", "best.pt"),
    "YOLOv26": os.path.join(BASE_DIR, "YOLOv26", "runs", "volleyball_train", "weights", "best.pt"),
}

# Where to save the per-model val results (inside each model's folder)
# and a combined output folder
COMBINED_OUT = os.path.join(BASE_DIR, "validate_results")
# ──────────────────────────────────────────────────────────────────────


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


def validate_model(name, model_path):
    """Run validation for one model and return a dict of metrics."""
    print(f"\n{'═'*60}")
    print(f"  Validating: {name}")
    print(f"  Weights   : {model_path}")
    print(f"{'═'*60}")

    result = {
        "model":     name,
        "precision": 0.0,
        "recall":    0.0,
        "map50":     0.0,
        "map95":     0.0,
        "status":    "failed",
        "output_dir": "",
    }

    if not os.path.exists(model_path):
        print(f"  ❌ Weights not found – skipping.")
        result["status"] = "weights_not_found"
        return result

    if not os.path.exists(DATA_YAML):
        print(f"  ❌ data.yaml not found at {DATA_YAML} – aborting.")
        result["status"] = "data_yaml_not_found"
        return result

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)

        model_root   = os.path.dirname(os.path.dirname(model_path))   # …/<Model>/runs
        val_project  = os.path.join(model_root, "val")
        val_name     = "volleyball_validate"

        metrics = model.val(
            data=DATA_YAML,
            project=val_project,
            name=val_name,
            save_json=True,
            save_txt=True,
            save_conf=True,
            plots=True,
            verbose=True,
        )

        map50 = float(metrics.box.map50)
        map95 = float(metrics.box.map)
        p, r  = get_mean_precision_recall(metrics.box)

        actual_out = os.path.join(val_project, val_name)
        result.update({
            "precision": p,
            "recall":    r,
            "map50":     map50,
            "map95":     map95,
            "status":    "ok",
            "output_dir": actual_out,
        })

        print(f"\n  📈 {name} Results:")
        print(f"     Precision  : {p:.4f}")
        print(f"     Recall     : {r:.4f}")
        print(f"     mAP@50     : {map50:.4f}")
        print(f"     mAP@50-95  : {map95:.4f}")
        print(f"     Saved to   : {actual_out}")

    except Exception as e:
        print(f"  ❌ Validation failed for {name}: {e}")
        traceback.print_exc()
        result["status"] = f"error: {e}"

    return result


def save_csv(results, out_dir):
    """Write summary CSV."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")
    fieldnames = ["model", "precision", "recall", "map50", "map95", "status", "output_dir"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  📄 CSV summary saved: {csv_path}")
    return csv_path


def save_comparison_chart(results, out_dir):
    """Save a grouped bar chart comparing all four models across all metrics."""
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        print("  ⚠️  No successful validations – skipping chart.")
        return

    labels   = [r["model"] for r in ok_results]
    metrics  = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    keys     = ["precision", "recall", "map50", "map95"]
    colors   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    x      = np.arange(len(labels))
    n      = len(metrics)
    width  = 0.18
    offset = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    for i, (key, metric, color) in enumerate(zip(keys, metrics, colors)):
        vals = [r[key] for r in ok_results]
        bars = ax.bar(x + offset[i], vals, width, label=metric, color=color,
                      alpha=0.87, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, color="white", fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title("YOLO Model Comparison – Volleyball Detection Validation",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    legend = ax.legend(
        handles=[mpatches.Patch(color=c, label=m) for c, m in zip(colors, metrics)],
        loc="upper right", framealpha=0.25, facecolor="#0f3460",
        edgecolor="white", fontsize=9, labelcolor="white",
    )

    ax.yaxis.grid(True, color="#334466", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

    chart_path = os.path.join(out_dir, "comparison.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊 Comparison chart saved: {chart_path}")
    return chart_path


def print_summary_table(results):
    """Print a pretty console table."""
    header = f"\n{'Model':<12} {'Precision':>10} {'Recall':>10} {'mAP@50':>10} {'mAP@50-95':>11}  Status"
    print("\n" + "━" * 68)
    print("  📋  FINAL COMPARISON SUMMARY")
    print("━" * 68)
    print(header)
    print("─" * 68)
    for r in results:
        if r["status"] == "ok":
            print(
                f"  {r['model']:<10} {r['precision']:>10.4f} {r['recall']:>10.4f}"
                f" {r['map50']:>10.4f} {r['map95']:>11.4f}  ✅ ok"
            )
        else:
            print(f"  {r['model']:<10} {'—':>10} {'—':>10} {'—':>10} {'—':>11}  ❌ {r['status']}")
    print("━" * 68)


def main():
    print("=" * 68)
    print("  YOLO Multi-Model Validation")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Output  : {COMBINED_OUT}")
    print("=" * 68)

    if not os.path.exists(DATA_YAML):
        print(f"\n❌ Shared data.yaml not found at:\n  {DATA_YAML}")
        print("Please check the dataset path and try again.")
        sys.exit(1)

    all_results = []
    for name, model_path in MODELS.items():
        result = validate_model(name, model_path)
        all_results.append(result)

    print_summary_table(all_results)
    save_csv(all_results, COMBINED_OUT)
    save_comparison_chart(all_results, COMBINED_OUT)

    print(f"\n✅ All done!  Combined results are in: {COMBINED_OUT}\n")


if __name__ == "__main__":
    main()
