"""
Visualization utilities for volleyball tracking.
Draws team-colored bounding boxes, ball markers, keypoints,
tracking trails, and state labels.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


# COCO skeleton connections for pose visualization
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),                # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)     # Legs
]

# Team colors (BGR)
TEAM1_COLOR     = (255, 150, 0)     # Blue-ish
TEAM2_COLOR     = (0, 80, 255)      # Red-ish
BALL_COLOR      = (0, 255, 255)     # Yellow
REFEREE_COLOR   = (0, 255, 255)     # Yellow
KEYPOINT_COLOR  = (255, 0, 255)     # Magenta
SKELETON_COLOR  = (255, 255, 0)     # Cyan


class Visualizer:
    """Visualization class for drawing detections on frames."""

    def __init__(self,
                 show_keypoints: bool = True,
                 show_skeleton: bool = True,
                 show_trails: bool = True,
                 show_states: bool = True,
                 trail_length: int = 30):
        self.show_keypoints = show_keypoints
        self.show_skeleton = show_skeleton
        self.show_trails = show_trails
        self.show_states = show_states
        self.trail_length = trail_length

    def draw_frame(self,
                   frame: np.ndarray,
                   results: Dict,
                   track_history: Dict[int, List[Tuple[float, float]]],
                   states: Optional[Dict[int, str]] = None,
                   state_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
                   ball_history: Optional[List[Tuple[float, float]]] = None
                   ) -> np.ndarray:
        """Draw all detections on a frame."""
        output = frame.copy()

        # Draw players with team colors
        for player in results.get('players', []):
            team = player.get('team', 'team1')
            color = TEAM1_COLOR if team == 'team1' else TEAM2_COLOR
            output = self._draw_player(
                output, player, color, team,
                track_history, states, state_colors
            )

        # Draw referee
        if results.get('referee'):
            output = self._draw_player(
                output, results['referee'], REFEREE_COLOR, "referee",
                track_history, states, state_colors,
                label_prefix="REF"
            )

        # Draw ball
        if results.get('ball'):
            output = self._draw_ball(output, results['ball'], ball_history)

        # Draw info panel
        output = self._draw_info_panel(output, results)

        return output

    def _draw_player(self,
                     frame: np.ndarray,
                     detection: Dict,
                     color: Tuple[int, int, int],
                     team: str,
                     track_history: Dict,
                     states: Optional[Dict],
                     state_colors: Optional[Dict],
                     label_prefix: str = "") -> np.ndarray:
        """Draw a single player detection with team label."""
        bbox = detection['bbox']
        track_id = detection['track_id']
        keypoints = detection.get('keypoints')

        x1, y1, x2, y2 = map(int, bbox)

        # Use state color if available, otherwise team color
        display_color = color
        if state_colors and track_id in state_colors:
            display_color = state_colors[track_id]

        # Draw bounding box with thick border
        cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)

        # Build label
        label_parts = []
        if label_prefix:
            label_parts.append(label_prefix)
        else:
            team_label = "T1" if team == "team1" else "T2"
            label_parts.append(f"{team_label}-P{track_id}")

        if states and track_id in states:
            label_parts.append(states[track_id])

        conf = detection.get('confidence', 0)
        label_parts.append(f"{conf:.0%}")

        label = " | ".join(label_parts)

        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(frame, (x1, y1 - label_h - 10),
                     (x1 + label_w + 10, y1), display_color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Draw keypoints and skeleton
        if keypoints is not None:
            if self.show_skeleton:
                frame = self._draw_skeleton(frame, keypoints)
            if self.show_keypoints:
                frame = self._draw_keypoints(frame, keypoints)

        # Draw trail
        if self.show_trails and track_id in track_history:
            frame = self._draw_trail(frame, track_history[track_id], display_color)

        return frame

    def _draw_keypoints(self, frame: np.ndarray,
                        keypoints: np.ndarray) -> np.ndarray:
        """Draw keypoints on frame."""
        for i, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, KEYPOINT_COLOR, -1)
        return frame

    def _draw_skeleton(self, frame: np.ndarray,
                       keypoints: np.ndarray) -> np.ndarray:
        """Draw skeleton connections on frame."""
        for connection in SKELETON_CONNECTIONS:
            idx1, idx2 = connection
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                kpt1, kpt2 = keypoints[idx1], keypoints[idx2]
                if kpt1[2] > 0.3 and kpt2[2] > 0.3:
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)
        return frame

    def _draw_trail(self, frame: np.ndarray,
                    trail: List[Tuple[float, float]],
                    color: Tuple[int, int, int]) -> np.ndarray:
        """Draw tracking trail."""
        trail = trail[-self.trail_length:]
        for i in range(1, len(trail)):
            alpha = i / len(trail)
            thickness = int(1 + alpha * 3)
            pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            cv2.line(frame, pt1, pt2, color, thickness)
        return frame

    def _draw_ball(self, frame: np.ndarray,
                   ball: Dict,
                   ball_history: Optional[List[Tuple[float, float]]] = None
                   ) -> np.ndarray:
        """Draw ball detection with prominent marker."""
        bbox = ball['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = max((x2 - x1) // 2, (y2 - y1) // 2, 8)

        # Draw outer glow
        cv2.circle(frame, (center_x, center_y), radius + 4, BALL_COLOR, 2)
        # Draw inner circle
        cv2.circle(frame, (center_x, center_y), radius, BALL_COLOR, -1)
        # Draw crosshair
        cv2.line(frame, (center_x - radius - 6, center_y),
                (center_x + radius + 6, center_y), BALL_COLOR, 1)
        cv2.line(frame, (center_x, center_y - radius - 6),
                (center_x, center_y + radius + 6), BALL_COLOR, 1)

        # Label
        conf = ball.get('confidence', 0)
        label = f"BALL {conf:.0%}"
        cv2.putText(frame, label, (x1 - 10, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, BALL_COLOR, 2)

        # Draw ball trail
        if ball_history and self.show_trails:
            frame = self._draw_trail(frame, ball_history, BALL_COLOR)

        return frame

    def _draw_info_panel(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw information panel at top of frame."""
        h, w = frame.shape[:2]

        # Create semi-transparent panel
        panel_height = 45
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Count teams
        team1_count = sum(1 for p in results.get('players', []) if p.get('team') == 'team1')
        team2_count = sum(1 for p in results.get('players', []) if p.get('team') == 'team2')
        has_ball = results.get('ball') is not None

        # Draw team info with colors
        x_pos = 10
        cv2.putText(frame, f"Team 1: {team1_count}", (x_pos, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEAM1_COLOR, 2)
        x_pos += 160
        cv2.putText(frame, f"Team 2: {team2_count}", (x_pos, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEAM2_COLOR, 2)
        x_pos += 160

        ball_text = "Ball: TRACKED" if has_ball else "Ball: ---"
        ball_color = BALL_COLOR if has_ball else (100, 100, 100)
        cv2.putText(frame, ball_text, (x_pos, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, ball_color, 2)

        return frame
