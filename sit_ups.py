import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------ Helper Functions ------------------
def angle_between(a, b, c):
    """Calculates the angle between three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# ------------------ Thresholds and Constants ------------------
HIP_DOWN_ANGLE_TH = 160     # Angle when lying down
HIP_UP_ANGLE_TH = 80        # Angle at the top of the sit-up
SMOOTHING_WINDOW = 5

# ------------------ Situp Counter Class ------------------
class SitupCounter:
    def __init__(self):
        self.state = "down"
        self.reps = 0
        self.valid_reps = 0
        self.hip_angles = deque(maxlen=SMOOTHING_WINDOW)
        self.min_hip_in_rep = 180

    def update(self, hip_angle):
        self.hip_angles.append(hip_angle)
        h_ang = np.mean(self.hip_angles)
        self.min_hip_in_rep = min(self.min_hip_in_rep, h_ang)

        if self.state == "down":
            if h_ang <= HIP_UP_ANGLE_TH:  # Torso lifted to sit-up position
                self.state = "up"
                self.min_hip_in_rep = h_ang
        elif self.state == "up":
            if h_ang >= HIP_DOWN_ANGLE_TH:  # Returned to floor
                self.reps += 1
                if self.min_hip_in_rep <= HIP_UP_ANGLE_TH + 10: # Check for full range of motion
                    self.valid_reps += 1
                
                self.state = "down"
                self.min_hip_in_rep = 180

# ------------------ Main Processing Function ------------------
def situp_counter(input_path, output_path="output_situps.mp4"):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(3)), int(cap.get(4))))
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    counter = SitupCounter()
    font = cv2.FONT_HERSHEY_SIMPLEX

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def xy(i): return np.array([lm[i].x * w, lm[i].y * h])

            # hip angle = shoulder-hip-knee
            left_hip_ang = angle_between(xy(mp_pose.PoseLandmark.LEFT_SHOULDER),
                                          xy(mp_pose.PoseLandmark.LEFT_HIP),
                                          xy(mp_pose.PoseLandmark.LEFT_KNEE))
            right_hip_ang = angle_between(xy(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                                           xy(mp_pose.PoseLandmark.RIGHT_HIP),
                                           xy(mp_pose.PoseLandmark.RIGHT_KNEE))
            hip_angle = (left_hip_ang + right_hip_ang) / 2.0
            
            counter.update(hip_angle)

            # Draw landmarks and info
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Reps and status overlay
            cv2.rectangle(frame, (0, 0), (320, 120), (0, 0, 0), -1)
            cv2.putText(frame, "TOTAL REPS: {}".format(counter.reps), (10, 40), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, "GOOD REPS: {}".format(counter.valid_reps), (10, 80), font, 1, (0, 255, 0), 2)
            cv2.putText(frame, "BAD REPS: {}".format(counter.reps - counter.valid_reps), (10, 120), font, 1, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    
    return counter.valid_reps, counter.reps - counter.valid_reps, output_path
