import pickle
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
from collections import defaultdict

class EyeContact:
    def __init__(self, model_path='Eye.pkl', video_source=0):
        self.class_counts = defaultdict(int)
        self.total_frames = 0
        self.cap = cv2.VideoCapture(video_source)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.results = []

    def print_summary(self):
        summary = {}
        for class_name, count in self.class_counts.items():
            percentage = (count / self.total_frames) * 100
            summary[class_name] = percentage
        return summary

    def process_frame(self, frame):
        self.total_frames += 1
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw face landmarks
            self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                          self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

            # Draw right hand landmarks
            self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

            # Draw left hand landmarks
            self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Export coordinates and make predictions
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten().tolist()

                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten().tolist()

                # Concatenate rows
                row = pose_row + face_row

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = self.model.predict(X)[0]
                body_language_prob = self.model.predict_proba(X)[0]

                # Update class counts
                self.class_counts[body_language_class] += 1
                self.results.append(body_language_class)

                # Grab ear coords for annotation
                coords = np.multiply(
                    (results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].x,
                    results.pose_landmarks.landmark[self.mp_holistic.PoseLandmark.LEFT_EAR].y),
                    [640, 480]).astype(int)

                # Draw rectangle and text for class prediction
                cv2.rectangle(image, (coords[0], coords[1] + 5), (coords[0] + len(body_language_class) * 20, coords[1] - 30), (245, 117, 16), -1)
                cv2.putText(image, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f'Error: {e}')
                pass

            # Display the resulting frame
            cv2.imshow('Raw Webcam Feed', image)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.process_frame(frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        return self.print_summary(), self.results
