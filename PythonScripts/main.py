from BodyLanguageRecognizer import BodyLanguageRecognizer
from AudioProcessor import AudioProcessor
from Eyecontact import EyeContact
from EmotionDetector import EmotionDetector
from HandMovementDetector import HandMovementDetector
from object_tracker import ObjectTracker


video_path = "./Test_video.mp4"

# Body Language Recognition
recognizer = BodyLanguageRecognizer(video_path)
recognizer.process_video()
body_language_summary = recognizer.print_summary()

# Audio Processing
audio_processor = AudioProcessor(video_path)
classify_audio = audio_processor.classify_audio()
full_text = audio_processor.transcribe_audio()
text_summary = audio_processor.summarize_text(full_text)

# Eye Contact Detection
eye_contact_detector = EyeContact(video_source=video_path)
eye_contact_results, _ = eye_contact_detector.run()  # Modify to get only percentages

# Emotion Detection
emotion_detector = EmotionDetector(video_source=video_path)
emotion_results = emotion_detector.run()

# Hand Movement Detection
hand_movement_detector = HandMovementDetector(video_source=video_path)
hand_movement_results = hand_movement_detector.detect_movement()
movement_report = hand_movement_detector.generate_report()

# Object Tracking
model_path = 'yolov8n.pt'
object_tracker = ObjectTracker(model_path, video_path)
object_tracker.process_video()
object_tracker_assessment = object_tracker.print_assessment()

# Printing combined analysis results
print("\n---\Analysis Results\n")

# Body Language Summary
print("- Body Language Summary:", body_language_summary)

# Audio Classification
print("\n- Audio Classification:")
for item in classify_audio:
    print("  - {}: {:.2f}%".format(item['label'].capitalize(), item['score']*100))

# Transcribed Text
print("\n- Transcribed Text:")
print("  \"{}\"".format(full_text))

# Text Summary
print("\n- Text Summary:")
print("  \"{}\"".format(text_summary[0]["summary_text"]))

# Eye Contact Analysis
print("\n- Eye Contact Analysis:")
for label, percentage in eye_contact_results.items():
    print("  - {}: {:.2f}%".format(label.capitalize(), percentage))

# Emotion Recognition
print("\n- Emotion Recognition:")
for emotion, percentage in emotion_results.items():
    print("  - {}: {:.2f}%".format(emotion.capitalize(), percentage))

# Movement Report
print("\n- Movement Hands Report: {:.2f}%".format(movement_report))

# Object Tracking Assessment
print("\n- Object Tracking Assessment:")
print(object_tracker_assessment)


