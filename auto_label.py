"""
Auto-labeling script for volleyball dataset.
Uses pretrained YOLOv8 to generate labels for 3 classes:
  0: volleyball (sports ball)
  1: player_team1 (upper half of frame)
  2: player_team2 (lower half of frame)

Writes standard YOLO detection format (class x_center y_center width height).
"""

import os
import glob
import cv2
from ultralytics import YOLO
from pathlib import Path


# ─── Config ───────────────────────────────────────────────────────────────────
DETECTION_MODEL = "yolov8n.pt"       # pretrained COCO model
DATASET_ROOT    = "dataset"
IMAGE_DIRS      = ["images/train", "images/val"]
LABEL_DIRS      = ["labels/train", "labels/val"]
PERSON_CLASS    = 0                  # COCO person class
BALL_CLASS      = 32                 # COCO sports ball class
CONF_PERSON     = 0.35               # lower threshold to catch more players
CONF_BALL       = 0.15               # very low for small volleyball
TEAM_SPLIT      = 0.50               # fraction of frame height to split teams
# ──────────────────────────────────────────────────────────────────────────────

# Our custom class mapping
CLASS_VOLLEYBALL   = 0
CLASS_PLAYER_TEAM1 = 1
CLASS_PLAYER_TEAM2 = 2


def auto_label():
    print(f"[✔] Loading detection model: {DETECTION_MODEL}")
    model = YOLO(DETECTION_MODEL)

    total_images = 0
    total_balls = 0
    total_team1 = 0
    total_team2 = 0

    for img_dir, lbl_dir in zip(IMAGE_DIRS, LABEL_DIRS):
        img_path = os.path.join(DATASET_ROOT, img_dir)
        lbl_path = os.path.join(DATASET_ROOT, lbl_dir)

        if not os.path.exists(img_path):
            print(f"[!] Image directory not found: {img_path}")
            continue

        os.makedirs(lbl_path, exist_ok=True)

        # Find all images
        images = sorted(
            glob.glob(os.path.join(img_path, "*.jpg")) +
            glob.glob(os.path.join(img_path, "*.png")) +
            glob.glob(os.path.join(img_path, "*.jpeg"))
        )

        print(f"\n[→] Processing {len(images)} images in {img_dir}")

        for idx, img_file in enumerate(images):
            img = cv2.imread(img_file)
            if img is None:
                print(f"  [!] Cannot read: {img_file}")
                continue

            h, w = img.shape[:2]
            split_y = h * TEAM_SPLIT  # team boundary line

            # Run detection for persons and balls
            results = model.predict(
                img,
                conf=min(CONF_PERSON, CONF_BALL),
                classes=[PERSON_CLASS, BALL_CLASS],
                verbose=False,
                device="cuda"
            )

            labels = []
            frame_balls = 0
            frame_t1 = 0
            frame_t2 = 0

            if results and len(results) > 0:
                res = results[0]
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Convert to YOLO normalized format
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    if cls_id == BALL_CLASS and conf >= CONF_BALL:
                        # Volleyball
                        labels.append(f"{CLASS_VOLLEYBALL} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
                        frame_balls += 1

                    elif cls_id == PERSON_CLASS and conf >= CONF_PERSON:
                        # Determine team based on vertical position
                        center_y = (y1 + y2) / 2
                        if center_y < split_y:
                            team_class = CLASS_PLAYER_TEAM1
                            frame_t1 += 1
                        else:
                            team_class = CLASS_PLAYER_TEAM2
                            frame_t2 += 1

                        labels.append(f"{team_class} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

            # Write label file
            label_file = os.path.join(
                lbl_path,
                Path(img_file).stem + ".txt"
            )
            with open(label_file, "w") as f:
                f.write("\n".join(labels))

            total_images += 1
            total_balls += frame_balls
            total_team1 += frame_t1
            total_team2 += frame_t2

            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(images)}] processed")

    print(f"\n{'─' * 55}")
    print(f"AUTO-LABELING COMPLETE")
    print(f"{'─' * 55}")
    print(f"  Total images labeled : {total_images}")
    print(f"  Total ball detections: {total_balls}")
    print(f"  Total team1 players  : {total_team1}")
    print(f"  Total team2 players  : {total_team2}")
    print(f"{'─' * 55}")


if __name__ == "__main__":
    auto_label()
