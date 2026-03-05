"""
Volleyball Tracking System
Main entry point for video processing with custom YOLO detection,
optional pose estimation, tracking, and action classification.
"""

import cv2
import argparse
from pathlib import Path
from typing import Optional

from tracker.yolo_tracker import YOLOTracker
from classifier.action_classifier import ActionClassifier, PlayerState
from utils.visualization import Visualizer


def process_video(
    input_path: str,
    output_path: Optional[str] = None,
    detection_model: str = "runs/detect/volleyball_v2/weights/best.pt",
    pose_model: str = "yolo26n.pt",
    show_preview: bool = True,
    conf_threshold: float = 0.3
) -> None:
    """
    Process a video file or webcam stream.

    Args:
        input_path: Path to video file or '0' for webcam
        output_path: Path to save output video (optional)
        detection_model: Path to custom volleyball detection model
        pose_model: Path to YOLO pose model (optional)
        show_preview: Whether to show live preview
        conf_threshold: Confidence threshold for detections
    """
    # Initialize components
    print("Initializing tracker...")
    tracker = YOLOTracker(
        detection_model_path=detection_model,
        pose_model_path=pose_model,
        conf_threshold=conf_threshold
    )

    print("Initializing classifier...")
    classifier = ActionClassifier()

    print("Initializing visualizer...")
    visualizer = Visualizer(
        show_keypoints=True,
        show_skeleton=True,
        show_trails=True,
        show_states=True
    )

    # Open video source
    if input_path == '0' or input_path == 0:
        cap = cv2.VideoCapture(0)
        print("Using webcam as input...")
    else:
        cap = cv2.VideoCapture(input_path)
        print(f"Processing video: {input_path}")

    if not cap.isOpened():
        print(f"Error: Could not open video source: {input_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Initialize video writer if output path specified
    writer = None
    if output_path:
        ext = Path(output_path).suffix.lower()
        if ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")

    # Process frames
    frame_count = 0
    print("\nProcessing... Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run detection and tracking
            results = tracker.detect_and_track(frame)

            # Classify player states
            states = {}
            state_colors = {}

            for player in results.get('players', []):
                track_id = player['track_id']
                velocity = tracker.calculate_velocity(track_id)

                state = classifier.classify(
                    track_id=track_id,
                    keypoints=player.get('keypoints'),
                    bbox=player['bbox'],
                    velocity=velocity
                )

                states[track_id] = state.value
                state_colors[track_id] = classifier.get_state_color(state)

            # Draw visualizations
            annotated_frame = visualizer.draw_frame(
                frame=frame,
                results=results,
                track_history=tracker.track_history,
                states=states,
                state_colors=state_colors,
                ball_history=tracker.get_ball_history()
            )

            # Show progress
            if total_frames > 0:
                progress = (frame_count / total_frames) * 100
                print(f"\rFrame {frame_count}/{total_frames} ({progress:.1f}%)", end="")
            else:
                print(f"\rFrame {frame_count}", end="")

            # Write to output
            if writer:
                writer.write(annotated_frame)

            # Show preview
            if show_preview:
                cv2.imshow("Volleyball Tracker", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nUser interrupted.")
                    break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    finally:
        print(f"\n\nProcessed {frame_count} frames.")
        cap.release()
        if writer:
            writer.release()
            print(f"Output saved to: {output_path}")
        cv2.destroyAllWindows()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Volleyball Tracking System with YOLO"
    )

    parser.add_argument(
        "input", type=str, nargs="?", default="0",
        help="Input video path or '0' for webcam (default: webcam)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output video path (optional)"
    )
    parser.add_argument(
        "--model", type=str,
        default="runs/detect/volleyball_v23/weights/best.pt",
        help="Path to custom volleyball detection model"
    )
    parser.add_argument(
        "--pose-model", type=str, default="yolov8n-pose.pt",
        help="Path to YOLO pose model (default: yolov8n-pose.pt)"
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Disable live preview window"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Confidence threshold (default: 0.3)"
    )

    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        detection_model=args.model,
        pose_model=args.pose_model,
        show_preview=not args.no_preview,
        conf_threshold=args.conf
    )


if __name__ == "__main__":
    main()
