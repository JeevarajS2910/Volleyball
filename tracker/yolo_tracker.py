"""
YOLO Tracker for volleyball detection and tracking.
Uses a single custom-trained model for all classes:
  0: volleyball
  1: player_team1
  2: player_team2
Optionally uses a pose model for action classification.
"""

from ultralytics import YOLO
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2


class YOLOTracker:
    """Unified tracker using custom volleyball detection model."""

    # Custom class IDs (from our training)
    VOLLEYBALL_CLASS = 0
    TEAM1_CLASS = 1
    TEAM2_CLASS = 2

    # Keypoint indices (COCO format) for pose model
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    def __init__(self,
                 detection_model_path: str = "runs/detect/volleyball_v23/weights/best.pt",
                 pose_model_path: Optional[str] = "yolo26n.pt",
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.7):
        """
        Initialize the tracker.

        Args:
            detection_model_path: Path to custom trained volleyball model
            pose_model_path: Path to YOLO pose model (optional, for action classification)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        print(f"  Loading detection model: {detection_model_path}")
        self.detection_model = YOLO(detection_model_path)

        self.pose_model = None
        if pose_model_path:
            try:
                print(f"  Loading pose model: {pose_model_path}")
                self.pose_model = YOLO(pose_model_path)
            except Exception as e:
                print(f"  [!] Pose model not loaded: {e}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Track history for velocity calculation and trails
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.ball_history: List[Tuple[float, float]] = []

    def detect_and_track(self, frame: np.ndarray) -> Dict:
        """
        Run detection and tracking on a frame.

        Returns dict with keys:
            - players: list of player dicts (with team info)
            - ball: ball detection dict or None
            - referee: None (referees handled differently now)
        """
        results = {
            'players': [],
            'ball': None,
            'referee': None,
            'frame': frame
        }

        # Run custom detection model with tracking
        det_results = self.detection_model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        # Optionally run pose model for keypoints
        pose_keypoints = {}
        if self.pose_model:
            try:
                pose_results = self.pose_model.predict(
                    frame,
                    conf=0.4,
                    verbose=False
                )
                if pose_results and len(pose_results) > 0:
                    pr = pose_results[0]
                    if pr.boxes is not None and pr.keypoints is not None:
                        for i, box in enumerate(pr.boxes):
                            bbox = box.xyxy[0].cpu().numpy()
                            kpts = pr.keypoints[i].data[0].cpu().numpy()
                            # Store keypoints keyed by bbox center for matching
                            cx = (bbox[0] + bbox[2]) / 2
                            cy = (bbox[1] + bbox[3]) / 2
                            pose_keypoints[(int(cx), int(cy))] = kpts
            except Exception:
                pass

        # Process detection results
        if det_results and len(det_results) > 0:
            det = det_results[0]
            if det.boxes is not None and len(det.boxes) > 0:
                for box in det.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0]) if box.id is not None else -1

                    if cls_id == self.VOLLEYBALL_CLASS:
                        # Ball detection
                        ball_det = {
                            'track_id': track_id,
                            'bbox': bbox,
                            'confidence': conf
                        }
                        # Keep best ball
                        if results['ball'] is None or conf > results['ball']['confidence']:
                            results['ball'] = ball_det

                        # Track ball position
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        self.ball_history.append((cx, cy))
                        self.ball_history = self.ball_history[-60:]

                    elif cls_id in (self.TEAM1_CLASS, self.TEAM2_CLASS):
                        # Player detection
                        cx = int((bbox[0] + bbox[2]) / 2)
                        cy = int((bbox[1] + bbox[3]) / 2)

                        # Try to match with pose keypoints
                        kpts = self._match_pose(cx, cy, pose_keypoints)

                        team = "team1" if cls_id == self.TEAM1_CLASS else "team2"

                        detection = {
                            'track_id': track_id,
                            'bbox': bbox,
                            'confidence': conf,
                            'keypoints': kpts,
                            'team': team,
                            'class_id': cls_id,
                            'is_referee': False
                        }

                        results['players'].append(detection)

                        # Update track history
                        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append(center)
                        self.track_history[track_id] = self.track_history[track_id][-30:]

        return results

    def _match_pose(self, cx: int, cy: int,
                    pose_keypoints: Dict) -> Optional[np.ndarray]:
        """Match a detection to the nearest pose keypoint set."""
        if not pose_keypoints:
            return None

        best_kpts = None
        best_dist = float('inf')
        for (px, py), kpts in pose_keypoints.items():
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist < best_dist and dist < 100:  # max 100px match distance
                best_dist = dist
                best_kpts = kpts

        return best_kpts

    def get_track_history(self, track_id: int) -> List[Tuple[float, float]]:
        """Get the position history for a tracked object."""
        return self.track_history.get(track_id, [])

    def calculate_velocity(self, track_id: int) -> Tuple[float, float]:
        """Calculate velocity for a tracked object."""
        history = self.get_track_history(track_id)
        if len(history) < 2:
            return (0.0, 0.0)

        recent = history[-5:]
        if len(recent) < 2:
            return (0.0, 0.0)

        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        dt = len(recent) - 1

        return (dx / dt, dy / dt)

    def get_ball_history(self) -> List[Tuple[float, float]]:
        """Get ball position history for trail drawing."""
        return self.ball_history
