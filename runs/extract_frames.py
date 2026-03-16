import cv2
import os
import sys

# Path to the input video
VIDEO_PATH = r"c:\volley ball\dataset\videoplayback.1770659521545.publer.com.mp4"
OUTPUT_DIR = r"c:\volley ball\runs\analysis_frames"

if not os.path.exists(VIDEO_PATH):
    print(f"Video not found at {VIDEO_PATH}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening video")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps}, Total Frames: {total_frames}")

# Extract frame 10, frame 500, frame 1000 to get a sense of the camera
frames_to_capture = [10, int(total_frames * 0.25), int(total_frames * 0.75)]

for f_idx in frames_to_capture:
    if f_idx >= total_frames:
        continue
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    
    if ret:
        out_path = os.path.join(OUTPUT_DIR, f"frame_{f_idx}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")

cap.release()
print("Done extracting sample frames.")
