import numpy as np
import cv2

class KalmanFilter:
    """
    A simple Kalman Filter for tracking points in 2D space.
    """
    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        # Initial state: [x, y, dx, dy]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Measurement matrix H (mapping state to measurement)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)
        
        # Transition matrix F (state evolution)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)
        
        # Process noise covariance Q
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance R
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance matrix P
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        self.initialized = False

    def predict(self):
        """Predict the next state."""
        return self.kf.predict()

    def update(self, coord):
        """Update the state with a new measurement."""
        if not self.initialized:
            self.kf.statePost = np.array([[coord[0]], [coord[1]], [0], [0]], np.float32)
            self.initialized = True
            return np.array([coord[0], coord[1]], np.float32)
            
        measurement = np.array([[np.float32(coord[0])], [np.float32(coord[1])]])
        self.kf.correct(measurement)
        prediction = self.kf.statePost
        return prediction[:2].flatten()
