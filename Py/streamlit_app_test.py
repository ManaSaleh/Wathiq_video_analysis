import streamlit as st
from BodyLanguageRecognizer import BodyLanguageRecognizer
from AudioProcessor import AudioProcessor
from Eyecontact import EyeContact
from EmotionDetector import EmotionDetector
from HandMovementDetector import HandMovementDetector
from object_tracker import ObjectTracker

def main():
    st.title("Multimodal Video Analysis App")

    st.sidebar.title("Upload Video")
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        video_path = f"./{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success("File uploaded successfully!")

        if st.sidebar.button("Analyze Video"):
            st.subheader("Processing video...")
            
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
            eye_contact_results, _ = eye_contact_detector.run()

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

            # Displaying results
            st.subheader("Analysis Results")

            # Body Language Summary
            st.markdown("### Body Language Summary")
            st.write(body_language_summary)

            # Audio Classification
            st.markdown("### Audio Classification")
            for item in classify_audio:
                st.write(f"- {item['label'].capitalize()}: {item['score']*100:.2f}%")

            # Transcribed Text
            st.markdown("### Transcribed Text")
            st.write(f"\"{full_text}\"")

            # Text Summary
            st.markdown("### Text Summary")
            st.write(f"\"{text_summary[0]['summary_text']}\"")

            # Eye Contact Analysis
            st.markdown("### Eye Contact Analysis")
            for label, percentage in eye_contact_results.items():
                st.write(f"- {label.capitalize()}: {percentage:.2f}%")

            # Emotion Recognition
            st.markdown("### Emotion Recognition")
            for emotion, percentage in emotion_results.items():
                st.write(f"- {emotion.capitalize()}: {percentage:.2f}%")

            # Movement Report
            st.markdown("### Hand Movement Report")
            st.write(f"{movement_report:.2f}%")

            # Object Tracking Assessment
            st.markdown("### Object Tracking Assessment")
            st.write(object_tracker_assessment)

if __name__ == "__main__":
    main()
