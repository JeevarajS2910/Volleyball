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
from tracker.court_tracker import CourtTracker
from utils.kalman_filter import KalmanFilter
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

        # Court Mask (Polygonal) - Default to a standard perspective for volleyball
        # This will be used to filter out audience/referees
        self.court_mask = None 
        self.filter_by_court = True

        # Track history for velocity calculation and trails
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.ball_history: List[Tuple[float, float]] = []
        
        # Rally and Pass logic
        self.rally_active = False
        self.current_touches = [] # List of (time, team, player_id)
        self.touch_coords = []    # List of (x, y) for heatmap
        self.player_participation: Dict[int, int] = {} # track_id -> total_touches
        self.last_ball_pos = None
        self.serve_speed = 0.0
        
        # Court Tracker
        self.court_tracker = CourtTracker()
        self.pixel_to_meter = 0.0
        
        # Ball Tracker (Kalman)
        self.ball_kf = KalmanFilter(process_noise=0.01, measurement_noise=0.05)

    def detect_and_track(self, frame: np.ndarray) -> Dict:
        """
        Run detection and tracking on a frame.

        Returns dict with keys:
            - players: list of player dicts (with team info)
            - ball: ball detection dict or None
            - court_mask: the polygon used for filtering
        """
        if self.court_mask is None:
            # Initialize a default court mask if not provided
            # This is a rough estimation for a standard wide-angle broadcast view
            h, w = frame.shape[:2]
            # More restrictive trapezoid for court masking
            self.court_mask = np.array([
                [int(w*0.15), int(h*0.95)],
                [int(w*0.85), int(h*0.95)],
                [int(w*0.75), int(h*0.35)],
                [int(w*0.25), int(h*0.35)]
            ], np.int32)

        # Update Court Tracking
        court_corners = self.court_tracker.update(frame)
        if court_corners is not None:
            self.court_mask = court_corners.astype(np.int32)
            self.pixel_to_meter = (self.court_tracker.pixel_to_meter_x + self.court_tracker.pixel_to_meter_y) / 2

        results = {
            'players': [],
            'ball': None,
            'court_mask': self.court_mask,
            'court_dims': (self.court_tracker.pixel_to_meter_x, self.court_tracker.pixel_to_meter_y),
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
                        # Smooth ball position with Kalman Filter
                        cx = (bbox[0] + bbox[2]) / 2
                        cy = (bbox[1] + bbox[3]) / 2
                        smoothed_pos = self.ball_kf.update((cx, cy))
                        cx, cy = smoothed_pos
                        
                        # Update ball det with smoothed pos
                        ball_det['bbox'] = np.array([
                            cx - (bbox[2]-bbox[0])/2,
                            cy - (bbox[3]-bbox[1])/2,
                            cx + (bbox[2]-bbox[0])/2,
                            cy + (bbox[3]-bbox[1])/2
                        ])

                        # Keep best ball
                        if results['ball'] is None or conf > results['ball']['confidence']:
                            results['ball'] = ball_det

                        # Track ball position
                        self.ball_history.append((cx, cy))
                        self.ball_history = self.ball_history[-60:]

                        # Add advanced features
                        ball_det['net_dist'] = self.court_tracker.get_distance_from_net((cx, cy))
                        ball_vel = self.calculate_velocity(track_id)
                        ball_det['landing_point'] = self.court_tracker.predict_landing_point((cx, cy), ball_vel)

                    elif cls_id in (self.TEAM1_CLASS, self.TEAM2_CLASS):
                        cx = int((bbox[0] + bbox[2]) / 2)
                        cy = int((bbox[1] + bbox[3]) / 2)

                        # Advanced Filtering
                        if self.filter_by_court and self.court_tracker.homography is not None:
                            rw_x, rw_y = self.court_tracker.pixel_to_meter((cx, cy))
                            
                            # 1. Identify Referee
                            # Referee is usually near the net (y=9m) and just outside court (x<0 or x>9)
                            is_near_net = abs(rw_y - 9.0) < 2.0
                            is_just_outside = (-3.0 <= rw_x < -0.2) or (9.2 < rw_x <= 12.0)
                            
                            if is_near_net and is_just_outside:
                                results['referee'] = {
                                    'track_id': track_id,
                                    'bbox': bbox,
                                    'confidence': conf,
                                    'position': (cx, cy),
                                    'team': 'referee'
                                }
                                continue

                            # 2. Filter Crowd (far away from court)
                            # We allow a 2m buffer for players
                            if not self.court_tracker.is_on_court((cx, cy), buffer_m=2.0):
                                continue
                        
                        # Fallback to polygonal mask if homography fails
                        elif self.filter_by_court:
                            is_inside = cv2.pointPolygonTest(self.court_mask, (float(cx), float(cy)), False) >= 0
                            if not is_inside:
                                continue

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
                            'position': (cx, cy)
                        }

                        results['players'].append(detection)

                        # Update track history
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append((cx, cy))
                        self.track_history[track_id] = self.track_history[track_id][-30:]

        self._update_rally_logic(results)
        return results

    def _update_rally_logic(self, resultsDict: Dict):
        """Update rally and touch statistics."""
        ball = resultsDict.get('ball')
        if not ball:
            self.last_ball_pos = None
            return

        curr_ball_pos = ((ball['bbox'][0] + ball['bbox'][2])/2, 
                         (ball['bbox'][1] + ball['bbox'][3])/2)
        
        # Check for touches
        for player in resultsDict.get('players', []):
            px, py = player['position']
            dist = np.sqrt((px - curr_ball_pos[0])**2 + (py - curr_ball_pos[1])**2)
            
            # If ball is very close to a player, consider it a touch
            if dist < 50: # distance threshold
                if not self.rally_active:
                    self.rally_active = True
                    print("\n[Rally Started]")
                
                # Record touch if it's a new player/team touch
                if not self.current_touches or self.current_touches[-1][2] != player['track_id']:
                    self.current_touches.append((None, player['team'], player['track_id']))
                    self.touch_coords.append((curr_ball_pos[0], curr_ball_pos[1]))
                    # Increment participation count
                    tid = player['track_id']
                    self.player_participation[tid] = self.player_participation.get(tid, 0) + 1
                    print(f"Touch by {player['team']} (Player {player['track_id']})")

        self.last_ball_pos = curr_ball_pos

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
