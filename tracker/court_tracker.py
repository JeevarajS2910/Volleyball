import cv2
import numpy as np
from typing import List, Tuple, Optional
from utils.kalman_filter import KalmanFilter

class CourtTracker:
    """
    Detects and tracks volleyball court lines and corners.
    Uses Kalman Filter for smoothing.
    """
    def __init__(self):
        # 4 corners of the court: top-left, top-right, bottom-right, bottom-left
        self.corners_kf = [KalmanFilter(0.01, 0.1) for _ in range(4)]
        self.corners = None
        
        # Standard volleyball court dimensions (meters)
        self.COURT_WIDTH = 9.0
        self.COURT_LENGTH = 18.0
        
        self.pixel_to_meter_x = 0.0
        self.pixel_to_meter_y = 0.0
        self.homography = None

    def detect_court(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect court corners using line detection.
        Returns 4 corners or None.
        """
        # Convert to HSV and filter for white lines
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjust these thresholds based on the video
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Hough Line Detection
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
        
        if lines is None:
            return None

        # Simplified corner detection: use the mask hull
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # Find the largest contour which should be the court
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < (frame.shape[0] * frame.shape[1] * 0.05):
            return None
        
        # Approximate the contour to a polygon
        epsilon = 0.05 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            # Sort corners: top-left, top-right, bottom-right, bottom-left
            corners = self._sort_corners(approx.reshape(4, 2))
            
            # Additional validation
            if self.validate_court(corners, frame.shape):
                return corners
            
        return None

    def validate_court(self, corners: np.ndarray, frame_shape: Tuple[int, int]) -> bool:
        """Validate if the detected quad looks like a volleyball court."""
        h, w = frame_shape[:2]
        area = cv2.contourArea(corners)
        
        # Area must be significant (at least 5% of frame)
        if area < (w * h * 0.05):
            return False
            
        # Aspect ratio check (court is 18m x 9m, so 2:1)
        # In perspective, this can vary, but shouldn't be extreme
        side1 = np.linalg.norm(corners[0] - corners[1])
        side2 = np.linalg.norm(corners[1] - corners[2])
        if side2 == 0: return False
        aspect = side1 / side2
        if aspect < 0.2 or aspect > 5.0:
            return False
            
        return True

    def is_on_court(self, pixel_coord: Tuple[float, float], buffer_m: float = 1.0) -> bool:
        """Check if real-world coordinates are on or near the court."""
        if self.homography is None:
            return True # If no court, we don't know, so don't filter (fallback)
            
        rw_x, rw_y = self.pixel_to_meter(pixel_coord)
        
        # Court is [0, 9] in X and [0, 18] in Y
        # We allow a buffer for players out of bounds
        if (-buffer_m <= rw_x <= self.COURT_WIDTH + buffer_m) and \
           (-buffer_m <= rw_y <= self.COURT_LENGTH + buffer_m):
            return True
            
        return False

    def update(self, frame: np.ndarray):
        """Update tracked corners and calculate court size."""
        detected_corners = self.detect_court(frame)
        
        smoothed_corners = []
        for i in range(4):
            # Always predict
            self.corners_kf[i].predict()
            if detected_corners is not None:
                # Update if we have a new measurement
                smoothed = self.corners_kf[i].update(detected_corners[i])
            else:
                # Use the prediction if no measurement
                # Note: KalmanFilter.update in my implementation returns statePost
                # I should add a way to get the current state without an update
                state = self.corners_kf[i].kf.statePost[:2].flatten()
                smoothed = state
            smoothed_corners.append(smoothed)
            
        self.corners = np.array(smoothed_corners, dtype=np.float32)
        self._update_homography()
        
        return self.corners

    def _sort_corners(self, pts):
        """Sort corners in order: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _update_homography(self):
        """Calculate homography and pixels-to-meters ratio."""
        if self.corners is None:
            return
            
        # Target coordinates in meters
        dst_pts = np.array([
            [0, 0],
            [self.COURT_WIDTH, 0],
            [self.COURT_WIDTH, self.COURT_LENGTH],
            [0, self.COURT_LENGTH]
        ], dtype=np.float32)
        
        self.homography, _ = cv2.findHomography(self.corners, dst_pts)
        
        # Calculate approximate scale (simple average for now)
        width_px = np.linalg.norm(self.corners[0] - self.corners[1])
        length_px = np.linalg.norm(self.corners[0] - self.corners[3])
        
        if width_px > 0 and length_px > 0:
            self.pixel_to_meter_x = self.COURT_WIDTH / width_px
            self.pixel_to_meter_y = self.COURT_LENGTH / length_px

    def pixel_to_meter(self, pixel_coord: Tuple[float, float]) -> Tuple[float, float]:
        """Convert pixel coordinates to meters using homography."""
        if self.homography is None:
            return (0.0, 0.0)
            
        pt = np.array([[[pixel_coord[0], pixel_coord[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography)
        return (transformed[0][0][0], transformed[0][0][1])

    def get_court_size_pixels(self) -> float:
        """Calculate court area in pixels."""
        if self.corners is None:
            return 0.0
        return cv2.contourArea(self.corners)

    def get_distance_from_net(self, pixel_coord: Tuple[float, float]) -> float:
        """Calculate distance from the net in meters."""
        if self.homography is None:
            return 0.0
        
        # Get real-world coordinates
        rw_pos = self.pixel_to_meter(pixel_coord)
        # Net is at Length/2 (9.0m)
        return abs(rw_pos[1] - self.COURT_LENGTH / 2)

    def predict_landing_point(self, pos: Tuple[float, float], vel: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Predict where the ball will land based on current velocity (pixels)."""
        if vel[1] <= 0: # Moving up or still
            return None
            
        # Very basic linear extrapolation to the bottom of the court
        # In a real system, we'd use a parabolic fit or the Kalman state
        if self.corners is None:
            return None
            
        ground_y = np.mean([self.corners[2][1], self.corners[3][1]])
        if pos[1] >= ground_y:
            return None
            
        time_to_land = (ground_y - pos[1]) / vel[1]
        landing_x = pos[0] + vel[0] * time_to_land
        return (landing_x, ground_y)
