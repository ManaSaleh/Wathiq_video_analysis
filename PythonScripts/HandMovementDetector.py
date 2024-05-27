import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from datetime import datetime

class HandMovementDetector:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.movement_data = []
        self.total_movement_session = 0.0
        self.prev_landmarks = None
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.frame_count = 0
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def detect_movement(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = self.hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

                    if self.prev_landmarks is not None:
                        movement = np.linalg.norm(landmarks - self.prev_landmarks, axis=1)
                        total_movement = np.sum(movement)
                        self.total_movement_session += total_movement
                        timestamp = datetime.now().strftime("%H:%M:%S")

                        self.movement_data.append((timestamp, total_movement))
                        print(f"Hand movement at {timestamp}: {total_movement:.2f}")

                    self.prev_landmarks = landmarks

            cv2.imshow('Hand Movement', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        return self.total_movement_session

    def generate_report(self):
        df = pd.DataFrame(self.movement_data, columns=['Timestamp', 'Movement'])
        df['Percentage of Total Movement'] = (df['Movement'] / self.total_movement_session) * 100
        df.to_csv('hand_movement_data.csv', index=False)
        average_movement =self.total_movement_session / self.frame_count if self.frame_count > 0 else 0
        print(f"Total hand movement during session: {average_movement}")
        return average_movement
