import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
import time

warnings.filterwarnings('ignore')

class GazeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # --- Landmark indices ---
        self.LEFT_IRIS = [473, 474, 475, 476, 477]
        self.RIGHT_IRIS = [468, 469, 470, 471, 472]
        self.LEFT_EYE_TOP_BOTTOM = [386, 374]
        self.LEFT_EYE_LEFT_RIGHT = [362, 263]
        
        # New landmarks for blink detection
        self.LEFT_EYELID = [386, 382, 381, 380, 373, 374]
        self.RIGHT_EYELID = [159, 158, 157, 173, 145, 144]
        
        # --- Calibration and Prediction Model ---
        self.calibration_data = []
        self.screen_coordinates = []
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_calibrated = False
        self.current_gaze_vector = None
        
        # --- Smoothing and State ---
        self.last_gaze_point = None
        self.smoothing_alpha = 0.3
        self.is_blinking = False

    def get_landmark_position(self, landmarks, indices, frame_shape):
        if not landmarks: return None
        h, w, _ = frame_shape
        points = []
        for idx in indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                points.append([lm.x * w, lm.y * h])
        return np.array(points) if points else None

    def detect_blink(self, landmarks, frame_shape):
        left_eyelid_pts = self.get_landmark_position(landmarks, self.LEFT_EYELID, frame_shape)
        right_eyelid_pts = self.get_landmark_position(landmarks, self.RIGHT_EYELID, frame_shape)
        
        if left_eyelid_pts is None or right_eyelid_pts is None:
            return True # Assume blink if eye landmarks are not visible

        # Calculate vertical distance for left eye
        left_v_dist = np.linalg.norm(left_eyelid_pts[0] - left_eyelid_pts[5])
        # Calculate horizontal distance for reference
        left_h_dist = np.linalg.norm(left_eyelid_pts[0] - left_eyelid_pts[3])

        try:
            eye_aspect_ratio = left_v_dist / (left_h_dist + 1e-6)
        except ZeroDivisionError:
            return True

        # If EAR is below a threshold, a blink is detected
        blink_threshold = 0.15 
        return eye_aspect_ratio < blink_threshold

    def calculate_gaze_and_head_pose(self, face_landmarks, frame):
        h, w, _ = frame.shape
        landmarks_3d = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark])

        # Gaze Vector Calculation
        left_iris_2d = self.get_landmark_position(face_landmarks, self.LEFT_IRIS, frame.shape).mean(axis=0)
        left_eye_horizontal_2d = self.get_landmark_position(face_landmarks, self.LEFT_EYE_LEFT_RIGHT, frame.shape)
        left_eye_vertical_2d = self.get_landmark_position(face_landmarks, self.LEFT_EYE_TOP_BOTTOM, frame.shape)
        
        try:
            h_ratio_left = (left_iris_2d[0] - left_eye_horizontal_2d[0][0]) / (left_eye_horizontal_2d[1][0] - left_eye_horizontal_2d[0][0])
            v_ratio_left = (left_iris_2d[1] - left_eye_vertical_2d[0][1]) / (left_eye_vertical_2d[1][1] - left_eye_vertical_2d[0][1])
            gaze_vector = np.array([h_ratio_left, v_ratio_left])
        except (ZeroDivisionError, IndexError, TypeError):
            return None 

        # 3D Head Pose Estimation
        face_2d = np.array([landmarks_3d[i, :2] for i in [1, 199, 33, 263, 61, 291]], dtype=np.float64)
        face_3d = np.array([landmarks_3d[i] for i in [1, 199, 33, 263, 61, 291]], dtype=np.float64)
        
        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]])
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4,1)))

        return np.concatenate((gaze_vector, rot_vec.flatten())) if success else None

    def calibrate(self, gaze_vector, screen_x, screen_y):
    
        if gaze_vector is not None:
            self.calibration_data.append(gaze_vector)
            self.screen_coordinates.append([screen_x, screen_y])
            print(f"Calibration point {len(self.calibration_data)} added.")
            if len(self.calibration_data) >= 9:
                self.model.fit(np.array(self.calibration_data), np.array(self.screen_coordinates))
                self.is_calibrated = True
                print("Calibration completed! Model is trained.")
    
    def predict_gaze_position(self, gaze_vector, frame_width, frame_height):

        if not self.is_calibrated or gaze_vector is None:
            return self.last_gaze_point # Return last known point if not calibrated
        
        try:
            prediction = self.model.predict([gaze_vector])[0]
            gaze_point = np.array([
                np.clip(prediction[0] * frame_width, 0, frame_width),
                np.clip(prediction[1] * frame_height, 0, frame_height)
            ], dtype=int)
            
            if self.last_gaze_point is None: self.last_gaze_point = gaze_point
            
            smoothed_point = self.smoothing_alpha * gaze_point + (1 - self.smoothing_alpha) * self.last_gaze_point
            self.last_gaze_point = smoothed_point
            return smoothed_point.astype(int)
        except Exception:
            return self.last_gaze_point
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        gaze_point = self.last_gaze_point # Default to last known point

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            self.is_blinking = self.detect_blink(face_landmarks, frame.shape)
            
            if not self.is_blinking:
                gaze_vector = self.calculate_gaze_and_head_pose(face_landmarks, frame)
                if gaze_vector is not None:
                    self.current_gaze_vector = gaze_vector
                    gaze_point = self.predict_gaze_position(gaze_vector, frame.shape[1], frame.shape[0])
        
        return frame, gaze_point


def main():
    gaze_tracker = GazeTracker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = 'Advanced Driver Gaze Monitoring v2'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    calibration_targets = [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1), (0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]
    current_target = 0
    
    # For calibration feedback
    show_feedback = 0
    feedback_pos = (0, 0)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to grab frame."); time.sleep(1); break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        processed_frame, gaze_point = gaze_tracker.process_frame(frame)
        
        # --- UI Drawing ---
        road_region = (int(0.25*w), int(0.1*h), int(0.75*w), int(0.6*h))
        info_region = (int(0.6*w), int(0.65*h), int(0.95*w), int(0.95*h))

        # Draw regions
        # ... [UI code for regions remains the same] ...
        
        gaze_location_text = "Distracted"
        gaze_location_color = (0, 0, 255)
        if gaze_point is not None:
            gaze_x, gaze_y = gaze_point
            if road_region[0] < gaze_x < road_region[2] and road_region[1] < gaze_y < road_region[3]:
                gaze_location_text = "Road"
                gaze_location_color = (0, 255, 0)
                cv2.rectangle(overlay, (road_region[0], road_region[1]), (road_region[2], road_region[3]), (0, 255, 0), -1)
            elif info_region[0] < gaze_x < info_region[2] and info_region[1] < gaze_y < info_region[3]:
                gaze_location_text = "Infotainment"
                gaze_location_color = (0, 255, 255)
                cv2.rectangle(overlay, (info_region[0], info_region[1]), (info_region[2], info_region[3]), (0, 255, 255), -1)

        processed_frame = cv2.addWeighted(overlay, 0.3, processed_frame, 0.7, 0)
        cv2.rectangle(processed_frame, (0, 0), (int(0.23 * w), h), (20, 20, 20), -1)
        
        # Status Panel
        status_text = "CALIBRATED" if gaze_tracker.is_calibrated else f"CALIBRATING ({len(gaze_tracker.calibration_data)}/9)"
        status_color = (0, 255, 0) if gaze_tracker.is_calibrated else (0, 165, 255)
        cv2.putText(processed_frame, "STATUS", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_frame, status_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Blink Status
        blink_text = "Blinking" if gaze_tracker.is_blinking else "Eyes Open"
        blink_color = (0, 165, 255) if gaze_tracker.is_blinking else (0, 255, 0)
        cv2.putText(processed_frame, "EYE STATUS", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_frame, blink_text, (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blink_color, 2)
        
        # Gaze Location
        cv2.putText(processed_frame, "GAZE LOCATION", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_frame, gaze_location_text, (15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gaze_location_color, 2)
        
        # Controls
        cv2.putText(processed_frame, "CONTROLS", (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # ... [UI code for controls remains the same] ...

        # Gaze Dot
        if gaze_point is not None and gaze_tracker.is_calibrated:
            dot_overlay = processed_frame.copy()
            cv2.circle(dot_overlay, tuple(gaze_point), 30, (255, 255, 255), -1)
            processed_frame = cv2.addWeighted(dot_overlay, 0.4, processed_frame, 0.6, 0)
        
        # Calibration Target Drawing
        if not gaze_tracker.is_calibrated:
            target_x, target_y = int(calibration_targets[current_target][0] * w), int(calibration_targets[current_target][1] * h)
            pulse = int(np.sin(time.time() * 5) * 5 + 20);
            cv2.circle(processed_frame, (target_x, target_y), pulse, (0, 255, 255), 2);
            cv2.circle(processed_frame, (target_x, target_y), 8, (0, 255, 255), -1);
        
        # Draw calibration feedback
        if show_feedback > 0:
            cv2.circle(processed_frame, feedback_pos, 25, (0, 255, 0), 4)
            show_feedback -= 1

        cv2.imshow(window_name, processed_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): break
        elif key == ord('c') and gaze_tracker.current_gaze_vector is not None:
            tx, ty = calibration_targets[current_target][0], calibration_targets[current_target][1]
            gaze_tracker.calibrate(gaze_tracker.current_gaze_vector, tx, ty)
            
            # Set feedback animation
            feedback_pos = (int(tx*w), int(ty*h))
            show_feedback = 10 # show for 10 frames
            
            current_target = (current_target + 1) % len(calibration_targets)
        elif key == ord('r'):
            gaze_tracker.__init__()
            current_target = 0
            print("Calibration reset!")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()