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
        self.heatmap_accum = None
        self.ball_touch_heatmap = None
        self.heatmap_alpha = 0.5
        self.show_court_mask = True

    def draw_frame(self,
                    frame: np.ndarray,
                    results: Dict,
                    track_history: Dict[int, List[Tuple[float, float]]],
                    states: Optional[Dict[int, str]] = None,
                    state_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
                    ball_history: Optional[List[Tuple[float, float]]] = None,
                    rally_info: Optional[Dict] = None,
                    touch_coords: Optional[List[Tuple[float, float]]] = None
                    ) -> np.ndarray:
        """Draw all detections on a frame."""
        output = frame.copy()
        h, w = output.shape[:2]

        # Initialize heatmaps if needed (handle potential size changes)
        if self.heatmap_accum is None or self.heatmap_accum.shape != (h, w):
            self.heatmap_accum = np.zeros((h, w), dtype=np.float32)
        if self.ball_touch_heatmap is None or self.ball_touch_heatmap.shape != (h, w):
            self.ball_touch_heatmap = np.zeros((h, w), dtype=np.float32)

        # Update Touch Heatmap
        if touch_coords:
            for tx, ty in touch_coords:
                cv2.circle(self.ball_touch_heatmap, (int(tx), int(ty)), 40, 1, -1)

        # Draw Court Mask
        if self.show_court_mask and 'court_mask' in results:
            mask_overlay = output.copy()
            cv2.polylines(mask_overlay, [results['court_mask']], True, (0, 255, 0), 2)
            cv2.fillPoly(mask_overlay, [results['court_mask']], (0, 255, 0))
            cv2.addWeighted(mask_overlay, 0.15, output, 0.85, 0, output)
            
            # Label corners
            for i, pt in enumerate(results['court_mask']):
                cv2.circle(output, tuple(pt), 5, (0, 255, 0), -1)
                cv2.putText(output, f"C{i+1}", (pt[0]+10, pt[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw players with team colors
        for player in results.get('players', []):
            team = player.get('team', 'team1')
            color = TEAM1_COLOR if team == 'team1' else TEAM2_COLOR
            output = self._draw_player(
                output, player, color, team,
                track_history, states, state_colors
            )
            # Accumulate for heatmap
            px, py = player['position']
            cv2.circle(self.heatmap_accum, (int(px), int(py)), 20, 1, -1)

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

        # Draw Mini-Map (Tactical View)
        output = self._draw_mini_map(output, results)

        # Draw info panel
        output = self._draw_info_panel(output, results, rally_info)

        # Overlay Heatmaps (Subtle)
        if self.heatmap_accum is not None and np.any(self.heatmap_accum > 0):
            heatmap_norm = cv2.normalize(self.heatmap_accum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            if heatmap_color.shape[:2] == output.shape[:2]:
                cv2.addWeighted(output, 1.0, heatmap_color, 0.2, 0, output)
            
        if self.ball_touch_heatmap is not None and np.any(self.ball_touch_heatmap > 0):
            ball_heatmap_norm = cv2.normalize(self.ball_touch_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            ball_heatmap_color = cv2.applyColorMap(ball_heatmap_norm, cv2.COLORMAP_HOT)
            if ball_heatmap_color.shape[:2] == output.shape[:2]:
                cv2.addWeighted(output, 1.0, ball_heatmap_color, 0.4, 0, output)

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
        
        # Label with Confidence and Distance from Net
        label = f"BALL {ball.get('confidence', 0):.0%}"
        if 'net_dist' in ball:
            label += f" | {ball['net_dist']:.1f}m from NET"
            
        cv2.putText(frame, label, (x1 - 10, y1 - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, BALL_COLOR, 2)

        # Draw Predicted Landing Point
        if 'landing_point' in ball and ball['landing_point'] is not None:
            lx, ly = map(int, ball['landing_point'])
            cv2.drawMarker(frame, (lx, ly), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
            cv2.putText(frame, "LANDING", (lx + 10, ly), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # Draw line to landing point
            cv2.line(frame, (center_x, center_y), (lx, ly), (0, 0, 255), 1, cv2.LINE_AA)

        # Draw ball trail
        if ball_history and self.show_trails:
            frame = self._draw_trail(frame, ball_history, BALL_COLOR)

        return frame

    def _draw_info_panel(self, frame: np.ndarray, results: Dict, rally_info: Optional[Dict] = None) -> np.ndarray:
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
        cv2.putText(frame, f"T1: {team1_count}", (x_pos, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEAM1_COLOR, 2)
        x_pos += 100
        cv2.putText(frame, f"T2: {team2_count}", (x_pos, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEAM2_COLOR, 2)
        x_pos += 100

        ball_text = "BALL" if has_ball else "---"
        ball_color = BALL_COLOR if has_ball else (100, 100, 100)
        cv2.putText(frame, ball_text, (x_pos, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 2)
        x_pos += 100

        if rally_info:
            rally_status = "Rally: ACTIVE" if rally_info.get('active') else "Rally: IDLE"
            cv2.putText(frame, rally_status, (x_pos, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            x_pos += 180
            
            touches = rally_info.get('touches', 0)
            cv2.putText(frame, f"Touches: {touches}", (x_pos, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            x_pos += 150
            
            speed = rally_info.get('serve_speed', 0.0)
            cv2.putText(frame, f"Speed: {speed:.1f} km/h", (x_pos, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            x_pos += 200

            # Show Court Dimensions if available
            if 'court_dims' in results:
                mx, my = results['court_dims']
                cv2.putText(frame, f"Scale: {mx*100:.1f}x{my*100:.1f} cm/px", (x_pos, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Draw Participation Stats (Dynamic Table)
            participation = rally_info.get('participation', {})
            if participation:
                total_touches = sum(participation.values())
                y_off = 70
                cv2.putText(frame, "Individual Participation:", (10, y_off),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                for tid, count in sorted(participation.items(), key=lambda x: x[1], reverse=True)[:5]:
                    y_off += 20
                    pct = (count / total_touches * 100) if total_touches > 0 else 0
                    cv2.putText(frame, f"P{tid}: {pct:.1f}% ({count} touches)", (20, y_off),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def _draw_mini_map(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw a bird's-eye view tactical map."""
        h, w = frame.shape[:2]
        
        # Mini-map dimensions (1:2 aspect ratio for volleyball court)
        mw = 150
        mh = 300
        padding = 10
        mx, my = w - mw - padding, padding + 50 # Below info panel
        
        # Create mini-map canvas
        canvas = np.zeros((mh + 20, mw + 20, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (10, 10), (mw + 10, mh + 10), (100, 100, 100), 1) # Boundary
        cv2.line(canvas, (10, mh // 2 + 10), (mw + 10, mh // 2 + 10), (255, 255, 255), 1) # Net
        
        # Draw Attack Lines (3m from net)
        attack_y1 = int((mh / 18.0) * (9 - 3)) + 10
        attack_y2 = int((mh / 18.0) * (9 + 3)) + 10
        cv2.line(canvas, (10, attack_y1), (mw + 10, attack_y1), (200, 200, 200), 1)
        cv2.line(canvas, (10, attack_y2), (mw + 10, attack_y2), (200, 200, 200), 1)
        
        # Draw Players
        for player in results.get('players', []):
            team = player.get('team', 'team1')
            color = TEAM1_COLOR if team == 'team1' else TEAM2_COLOR
            
            # Map screen to court coords (0-9, 0-18)
            cx, cy = player['position']
            # This requires access to the homography or a court mapping
            # We'll use a hacky relative position if court_mask is available
            # or better: rely on court_dims if present
            court_pos = self._get_court_coords(cx, cy, results)
            if court_pos:
                rx, ry = court_pos
                px = int((rx / 9.0) * mw) + 10
                py = int((ry / 18.0) * mh) + 10
                cv2.circle(canvas, (px, py), 4, color, -1)
                cv2.putText(canvas, f"{player['track_id']}", (px+5, py-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Draw Ball
        if results.get('ball'):
            bx, by = (results['ball']['bbox'][0] + results['ball']['bbox'][2])/2, \
                     (results['ball']['bbox'][1] + results['ball']['bbox'][3])/2
            court_pos = self._get_court_coords(bx, by, results)
            if court_pos:
                rx, ry = court_pos
                px = int((rx / 9.0) * mw) + 10
                py = int((ry / 18.0) * mh) + 10
                cv2.circle(canvas, (px, py), 5, BALL_COLOR, -1)

        # Overlay on frame
        alpha = 0.7
        roi = frame[my:my+canvas.shape[0], mx:mx+canvas.shape[1]]
        if roi.shape[0] == canvas.shape[0] and roi.shape[1] == canvas.shape[1]:
            cv2.addWeighted(canvas, alpha, roi, 1 - alpha, 0, roi)
            cv2.putText(frame, "TACTICAL MAP", (mx, my - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

    def _get_court_coords(self, px, py, results) -> Optional[Tuple[float, float]]:
        """Helper to get 0-9, 0-18 coordinates from pixel position."""
        if 'court_mask' not in results:
            return None
            
        # Simplified: Use the bbox of the court mask to get relative position
        mask = results['court_mask']
        min_x = np.min(mask[:, 0])
        max_x = np.max(mask[:, 0])
        min_y = np.min(mask[:, 1])
        max_y = np.max(mask[:, 1])
        
        # Check if inside
        if px < min_x or px > max_x or py < min_y or py > max_y:
            return None
            
        rel_x = (px - min_x) / (max_x - min_x)
        rel_y = (py - min_y) / (max_y - min_y)
        
        # Map to 9m x 18m
        return (rel_x * 9.0, rel_y * 18.0)
