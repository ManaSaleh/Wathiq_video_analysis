import os
from ultralytics import YOLO
import cv2
import numpy as np

class ObjectTracker:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.movement_distances = []
        self.frame_count = 0
        self.prev_centroid = None

    def process_video(self):
        ret = True
        while ret:
            ret, frame = self.cap.read()
            self.frame_count += 1

            if ret:
                # Detect and track objects
                results = self.model.track(frame, persist=True)
                bboxes = results[0].boxes  # Assuming first result corresponds to the person

                # Calculate centroid of bounding box
                if len(bboxes) > 0:
                    bbox = bboxes[0].xywh[0]  # xywh format
                    cx, cy = bbox[0].item(), bbox[1].item()
                    centroid = (cx, cy)

                    # Calculate movement distance
                    if self.prev_centroid is not None:
                        distance = np.linalg.norm(np.array(centroid) - np.array(self.prev_centroid))
                        self.movement_distances.append(distance)

                    self.prev_centroid = centroid

                    # Draw bounding box and centroid
                    frame = results[0].plot()
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)

                # Visualize
                cv2.imshow('frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    def assess_movement(self):
        total_movement = sum(self.movement_distances)
        average_movement = total_movement / self.frame_count if self.frame_count > 0 else 0

        if average_movement < 5:
            assessment = "Presenter is relatively stationary. Consider using more gestures and movements to engage the audience."
        elif 5 <= average_movement < 15:
            assessment = "Presenter has a moderate level of movement, which is good for engaging the audience."
        else:
            assessment = "Presenter moves a lot. Ensure the movements are purposeful and not distracting."

        return {
            "total_movement": total_movement,
            "average_movement": average_movement,
            "assessment": assessment
        }

    def print_assessment(self):
        assessment = self.assess_movement()
        return f"Total Movement Distance: {assessment['total_movement']:.2f}\n" \
               f"\nAverage Movement per Frame: {assessment['average_movement']:.2f}\n" \
               f"\nAssessment: {assessment['assessment']}"