from deepface import DeepFace
import cv2
from collections import Counter

class EmotionDetector:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.emotions = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def analyze_emotion(self, face):
        try:
            results = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            if isinstance(results, dict) and 'dominant_emotion' in results:
                return results['dominant_emotion']
            elif isinstance(results, list) and len(results) > 0 and 'dominant_emotion' in results[0]:
                return results[0]['dominant_emotion']
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
        return None

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            faces = self.detect_faces(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                emotion = self.analyze_emotion(face)
                if emotion:
                    self.emotions.append(emotion)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            cv2.imshow('Real-time Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return self.generate_report()

    def generate_report(self):
        if self.emotions:
            emotion_counts = Counter(self.emotions)
            total_predictions = len(self.emotions)
            emotion_percentages = {emotion: count / total_predictions * 100 for emotion, count in emotion_counts.items()}
            return emotion_percentages
        else:
            return "No emotion predictions were made."