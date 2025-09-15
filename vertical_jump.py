import cv2
import mediapipe as mp
import numpy as np

def detect_jumps_autoheight(input_path, output_path="output_jumps.mp4",
                            landmark_to_track="MID_HIP"):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(3)), int(cap.get(4))))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    y_positions = []
    jump_heights = []
    state = "ground"
    estimated_height_cm = None  # will calculate

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # --- Estimate person's height in video ---
            nose_y = lm[mp_pose.PoseLandmark.NOSE].y * h
            l_ankle_y = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h
            r_ankle_y = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h
            ankle_y = max(l_ankle_y, r_ankle_y)  # lowest ankle
            person_height_px = ankle_y - nose_y

            # Assume avg real-world human height ~ 170 cm (scaling factor)
            if estimated_height_cm is None and person_height_px > 0:
                scale_cm_per_px = 170 / person_height_px
                estimated_height_cm = 170  # store for display
            else:
                scale_cm_per_px = 170 / person_height_px if person_height_px > 0 else 1

            # --- Track landmark (hip or else) ---
            if landmark_to_track == "MID_HIP":
                y = (lm[mp_pose.PoseLandmark.LEFT_HIP].y +
                     lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2 * h
            else:
                y = lm[getattr(mp_pose.PoseLandmark, landmark_to_track)].y * h

            y_positions.append(y)

            # --- Jump detection logic ---
            if len(y_positions) > 5:
                baseline = np.percentile(y_positions, 90)  # standing height
                min_y = min(y_positions[-10:])
                diff = baseline - min_y

                if diff > 20 and state == "ground":
                    state = "air"
                    jump_heights.append(diff * scale_cm_per_px)
                elif diff < 10:
                    state = "ground"

            # --- Draw info on frame ---
            cv2.putText(frame, f"Jumps: {len(jump_heights)}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            if jump_heights:
                cv2.putText(frame, f"Last jump: {jump_heights[-1]:.1f} cm", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if estimated_height_cm:
                cv2.putText(frame, f"Est. Height: {estimated_height_cm:.0f} cm", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()

    print(f"✅ Saved: {output_path}")
    print(f"Total jumps: {len(jump_heights)}")
    print(f"Jump heights (cm): {jump_heights}")

    return output_path, jump_heights
