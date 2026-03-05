"""
Script to extract frames from video for dataset creation.
"""

import cv2
import os
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir, interval=10):
    """
    Extract frames from video at specified interval.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        interval: Save every Nth frame
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create output directory structure
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames. Extracting every {interval}th frame...")
    
    count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0:
            frame_name = f"frame_{count:06d}.jpg"
            save_path = os.path.join(images_dir, frame_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            print(f"\rSaved {saved_count} frames...", end="")
            
        count += 1
        
    cap.release()
    print(f"\nDone! Extracted {saved_count} frames to {images_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--output", default="dataset", help="Output directory")
    parser.add_argument("--interval", type=int, default=30, help="Frame extraction interval")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.interval)


if __name__ == "__main__":
    main()
