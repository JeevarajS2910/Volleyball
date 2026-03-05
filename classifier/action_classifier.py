"""
Action classifier for volleyball player states.
Uses pose keypoints to classify player actions.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from enum import Enum


class PlayerState(Enum):
    """Enum for player action states."""
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"
    JUMPING = "jumping"
    DIVING = "diving"
    SERVING = "serving"
    ATTACKING = "attacking"
    BLOCKING = "blocking"
    SETTING = "setting"
    RECEIVING = "receiving"
    UNKNOWN = "unknown"


class ActionClassifier:
    """
    Heuristic-based action classifier using pose keypoints.
    
    Keypoint indices (COCO format):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    
    # Keypoint indices
    NOSE = 0
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    def __init__(self, 
                 jump_threshold: float = 0.15,
                 arm_raise_threshold: float = 0.3,
                 dive_angle_threshold: float = 45):
        """
        Initialize the action classifier.
        
        Args:
            jump_threshold: Relative height threshold for jump detection
            arm_raise_threshold: Relative height threshold for raised arms
            dive_angle_threshold: Body angle threshold for dive detection (degrees)
        """
        self.jump_threshold = jump_threshold
        self.arm_raise_threshold = arm_raise_threshold
        self.dive_angle_threshold = dive_angle_threshold
        
        # History for temporal analysis
        self.state_history: Dict[int, List[PlayerState]] = {}
        self.position_history: Dict[int, List[Tuple[float, float]]] = {}
        
    def classify(self, 
                 track_id: int,
                 keypoints: Optional[np.ndarray],
                 bbox: np.ndarray,
                 velocity: Tuple[float, float] = (0, 0)) -> PlayerState:
        """
        Classify the player's action state.
        
        Args:
            track_id: Unique tracking ID for the player
            keypoints: Array of shape (17, 3) with (x, y, confidence)
            bbox: Bounding box [x1, y1, x2, y2]
            velocity: Velocity tuple (vx, vy) from tracking
            
        Returns:
            PlayerState enum value
        """
        if keypoints is None or len(keypoints) < 17:
            return PlayerState.UNKNOWN
        
        # Extract key measurements
        measurements = self._extract_measurements(keypoints, bbox)
        
        if measurements is None:
            return PlayerState.UNKNOWN
        
        # Determine state based on measurements and velocity
        state = self._determine_state(measurements, velocity)
        
        # Update history
        if track_id not in self.state_history:
            self.state_history[track_id] = []
        self.state_history[track_id].append(state)
        self.state_history[track_id] = self.state_history[track_id][-30:]
        
        # Apply temporal smoothing
        state = self._smooth_state(track_id, state)
        
        return state
    
    def _extract_measurements(self, keypoints: np.ndarray, 
                              bbox: np.ndarray) -> Optional[Dict]:
        """Extract relevant measurements from keypoints."""
        # Get keypoint coordinates and confidences
        def get_point(idx):
            if idx < len(keypoints) and keypoints[idx][2] > 0.3:
                return keypoints[idx][:2]
            return None
        
        # Key points
        nose = get_point(self.NOSE)
        l_shoulder = get_point(self.LEFT_SHOULDER)
        r_shoulder = get_point(self.RIGHT_SHOULDER)
        l_hip = get_point(self.LEFT_HIP)
        r_hip = get_point(self.RIGHT_HIP)
        l_wrist = get_point(self.LEFT_WRIST)
        r_wrist = get_point(self.RIGHT_WRIST)
        l_ankle = get_point(self.LEFT_ANKLE)
        r_ankle = get_point(self.RIGHT_ANKLE)
        l_knee = get_point(self.LEFT_KNEE)
        r_knee = get_point(self.RIGHT_KNEE)
        
        # Calculate measurements
        bbox_height = bbox[3] - bbox[1]
        bbox_width = bbox[2] - bbox[0]
        
        if bbox_height <= 0:
            return None
        
        measurements = {
            'bbox_height': bbox_height,
            'bbox_width': bbox_width,
            'bbox_aspect_ratio': bbox_width / bbox_height if bbox_height > 0 else 1
        }
        
        # Shoulder midpoint
        if l_shoulder is not None and r_shoulder is not None:
            measurements['shoulder_mid'] = (l_shoulder + r_shoulder) / 2
        
        # Hip midpoint
        if l_hip is not None and r_hip is not None:
            measurements['hip_mid'] = (l_hip + r_hip) / 2
        
        # Body angle (for diving detection)
        if 'shoulder_mid' in measurements and 'hip_mid' in measurements:
            dx = measurements['shoulder_mid'][0] - measurements['hip_mid'][0]
            dy = measurements['shoulder_mid'][1] - measurements['hip_mid'][1]
            measurements['body_angle'] = np.degrees(np.arctan2(abs(dx), abs(dy)))
        
        # Arms raised (relative to shoulders)
        if l_wrist is not None and l_shoulder is not None:
            measurements['left_arm_raised'] = (l_shoulder[1] - l_wrist[1]) / bbox_height
        if r_wrist is not None and r_shoulder is not None:
            measurements['right_arm_raised'] = (r_shoulder[1] - r_wrist[1]) / bbox_height
        
        # Knee bend (for jumping detection)
        if l_knee is not None and l_hip is not None and l_ankle is not None:
            knee_angle = self._calculate_angle(l_hip, l_knee, l_ankle)
            measurements['left_knee_angle'] = knee_angle
        if r_knee is not None and r_hip is not None and r_ankle is not None:
            knee_angle = self._calculate_angle(r_hip, r_knee, r_ankle)
            measurements['right_knee_angle'] = knee_angle
        
        # Ankle to hip ratio (for jump detection)
        if l_ankle is not None and l_hip is not None:
            measurements['left_leg_ratio'] = (l_hip[1] - l_ankle[1]) / bbox_height
        if r_ankle is not None and r_hip is not None:
            measurements['right_leg_ratio'] = (r_hip[1] - r_ankle[1]) / bbox_height
        
        return measurements
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, 
                         p3: np.ndarray) -> float:
        """Calculate angle at p2 between p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        return angle
    
    def _determine_state(self, measurements: Dict, 
                         velocity: Tuple[float, float]) -> PlayerState:
        """Determine player state from measurements."""
        vx, vy = velocity
        
        # Check for diving (body nearly horizontal)
        if 'body_angle' in measurements:
            if measurements['body_angle'] > self.dive_angle_threshold:
                return PlayerState.DIVING
        
        # Check for jumping (both arms raised high and upward velocity)
        left_arm_raised = measurements.get('left_arm_raised', 0)
        right_arm_raised = measurements.get('right_arm_raised', 0)
        both_arms_raised = (left_arm_raised > self.arm_raise_threshold and 
                           right_arm_raised > self.arm_raise_threshold)
        
        # Negative vy means moving upward in image coordinates
        is_moving_up = vy < -5
        
        if both_arms_raised and is_moving_up:
            return PlayerState.JUMPING
        
        # Check for blocking (arms raised, near net position)
        if both_arms_raised and abs(vy) < 3:
            return PlayerState.BLOCKING
        
        # Check for attacking (one arm raised high, jumping)
        one_arm_very_high = (left_arm_raised > 0.4 or right_arm_raised > 0.4)
        if one_arm_very_high and is_moving_up:
            return PlayerState.ATTACKING
        
        # Check for serving (one arm raised, relatively stationary)
        if one_arm_very_high and abs(vx) < 3 and abs(vy) < 3:
            return PlayerState.SERVING
        
        # Check for setting (arms raised at medium height)
        arms_medium = (0.15 < left_arm_raised < 0.35 and 
                      0.15 < right_arm_raised < 0.35)
        if arms_medium:
            return PlayerState.SETTING
        
        # Check for receiving (low stance, arms together)
        if 'left_knee_angle' in measurements and 'right_knee_angle' in measurements:
            avg_knee_angle = (measurements['left_knee_angle'] + 
                            measurements['right_knee_angle']) / 2
            if avg_knee_angle < 150:  # Bent knees
                return PlayerState.RECEIVING
        
        # Check for running
        if abs(vx) > 10:
            return PlayerState.RUNNING
        
        # Check for walking
        if abs(vx) > 3:
            return PlayerState.WALKING
        
        return PlayerState.STANDING
    
    def _smooth_state(self, track_id: int, current_state: PlayerState) -> PlayerState:
        """Apply temporal smoothing to reduce flickering."""
        history = self.state_history.get(track_id, [])
        
        if len(history) < 3:
            return current_state
        
        # Count recent states
        recent = history[-5:]
        state_counts = {}
        for s in recent:
            state_counts[s] = state_counts.get(s, 0) + 1
        
        # If current state appears multiple times, keep it
        if state_counts.get(current_state, 0) >= 2:
            return current_state
        
        # Otherwise, return the most common recent state
        most_common = max(state_counts, key=state_counts.get)
        return most_common
    
    def get_state_color(self, state: PlayerState) -> Tuple[int, int, int]:
        """Get display color for a state (BGR format)."""
        colors = {
            PlayerState.STANDING: (128, 128, 128),    # Gray
            PlayerState.WALKING: (255, 255, 0),       # Cyan
            PlayerState.RUNNING: (0, 255, 255),       # Yellow
            PlayerState.JUMPING: (0, 255, 0),         # Green
            PlayerState.DIVING: (255, 0, 255),        # Magenta
            PlayerState.SERVING: (0, 165, 255),       # Orange
            PlayerState.ATTACKING: (0, 0, 255),       # Red
            PlayerState.BLOCKING: (255, 0, 0),        # Blue
            PlayerState.SETTING: (255, 255, 255),     # White
            PlayerState.RECEIVING: (0, 128, 255),     # Orange-ish
            PlayerState.UNKNOWN: (100, 100, 100)      # Dark gray
        }
        return colors.get(state, (100, 100, 100))
