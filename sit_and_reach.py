import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def sit_and_reach_tracker(input_path, output_path="sit_and_reach_output.mp4"):
    """
    Processes a video file to calculate the maximum sit-and-reach distance.
    Returns the max reach in cm and the path to the output video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(3)), int(cap.get(4))))

    max_reach_px = 0
    reach_origin_px = None
    scale_cm_per_px = None

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get landmark positions
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                # Convert to pixel coordinates
                foot_x = int(((left_ankle.x + right_ankle.x) / 2) * w)
                hand_x = int(((left_wrist.x + right_wrist.x) / 2) * w)
                hip_x = int(((left_hip.x + right_hip.x) / 2) * w)
                
                # Use a reference point on the hip to mark the start of the reach
                if reach_origin_px is None:
                    reach_origin_px = hip_x

                # Distance: how far hands reached compared to the hip's starting point
                # This assumes the hip remains relatively stationary.
                current_reach_px = hand_x - reach_origin_px

                # Use a reference length to establish a cm/pixel ratio
                # The length from hip to ankle is a good proxy for body scale.
                if scale_cm_per_px is None:
                    hip_y = int(((left_hip.y + right_hip.y) / 2) * h)
                    ankle_y = int(((left_ankle.y + right_ankle.y) / 2) * h)
                    hip_to_ankle_px = abs(hip_y - ankle_y)
                    # Assume an average hip-to-ankle length of ~60cm for a general scale
                    if hip_to_ankle_px > 0:
                        scale_cm_per_px = 60 / hip_to_ankle_px

                if current_reach_px > max_reach_px:
                    max_reach_px = current_reach_px
                
                # Draw landmarks and connections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display stats on the video
                max_reach_cm = max_reach_px * scale_cm_per_px if scale_cm_per_px else 0
                cv2.putText(image, f"Max Reach: {max_reach_cm:.1f} cm", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            out.write(image)

    cap.release()
    out.release()

    max_reach_cm = max_reach_px * scale_cm_per_px if scale_cm_per_px else 0
    return max_reach_cm, output_path
