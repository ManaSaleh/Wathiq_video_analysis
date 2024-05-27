import streamlit as st
from BodyLanguageRecognizer import BodyLanguageRecognizer
from AudioProcessor import AudioProcessor
from EyeContact import EyeContact
from EmotionDetector import EmotionDetector
from HandMovementDetector import HandMovementDetector
from object_tracker import ObjectTracker
import os

def load_videos(directory):
    videos = []
    for filename in os.listdir(directory):
        if filename.endswith((".mp4", ".avi", ".mov")):  # Add more video formats if needed
            videos.append(os.path.join(directory, filename))
    return videos

def main():
    st.set_page_config(page_title="Interview/Presentation Analysis App", page_icon=":video_camera:")
    logo_path = "logo.png"
    # Display the image in the center
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_path,width=350)
    # st.title("Interview/Presentation Analysis App")

    st.sidebar.title("Upload Video")
    
    if "page" not in st.session_state:
        st.session_state.page = 0
   
    
    
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        video_path = f"./{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success("File uploaded successfully!")
    
    else:
        st.sidebar.write("Or choose from example videos below:")

    # Load example videos from 'videos' directory
    video_directory = "videos"
    example_videos = load_videos(video_directory)

    selected_video = None

    if example_videos:
        selected_video = st.sidebar.selectbox("Select an example video to play:", example_videos)

    if uploaded_file is not None or selected_video is not None:
        video_path = f"./{uploaded_file.name}" if uploaded_file is not None else selected_video

        st.sidebar.title("Select Models to Apply")
        apply_body_language = st.sidebar.checkbox("Body Language Recognition", value=True)
        apply_audio_processing = st.sidebar.checkbox("Audio Processing (Stutter, Dictation, Summarization)", value=True)
        apply_eye_contact = st.sidebar.checkbox("Eye Contact Detection", value=True)
        apply_emotion_detection = st.sidebar.checkbox("Emotion Detection", value=True)
        apply_hand_movement = st.sidebar.checkbox("Hand Movement Detection", value=True)
        apply_object_tracking = st.sidebar.checkbox("Body Movement Detection", value=True)

        if st.sidebar.button("Analyze Video"):
            st.subheader("Video Playback")
            st.video(video_path, format="video/mp4", start_time=0, subtitles=None, end_time=None, loop=False, autoplay=False, muted=False)
            st.subheader("Processing video...")

            # Body Language Recognition
            if apply_body_language:
                recognizer = BodyLanguageRecognizer(video_path)
                recognizer.process_video()
                body_language_summary = recognizer.print_summary()

            # Audio Processing
            if apply_audio_processing:
                audio_processor = AudioProcessor(video_path)
                audio_processor.extract_audio()  # Ensure audio is extracted
                if os.path.exists(audio_processor.output_audio_path):
                    classify_audio = audio_processor.classify_audio()
                    full_text = audio_processor.transcribe_audio()
                    text_summary = audio_processor.summarize_text(full_text)
                else:
                    st.error("Failed to extract audio from the video.")

            # Eye Contact Detection
            if apply_eye_contact:
                eye_contact_detector = EyeContact(video_source=video_path)
                eye_contact_results, _ = eye_contact_detector.run()

            # Emotion Detection
            if apply_emotion_detection:
                emotion_detector = EmotionDetector(video_source=video_path)
                emotion_results = emotion_detector.run()

            # Hand Movement Detection
            if apply_hand_movement:
                hand_movement_detector = HandMovementDetector(video_source=video_path)
                hand_movement_results = hand_movement_detector.detect_movement()
                movement_report = hand_movement_detector.generate_report()

            # Object Tracking
            if apply_object_tracking:
                model_path = 'yolov8n.pt'
                object_tracker = ObjectTracker(model_path, video_path)
                object_tracker.process_video()
                object_tracker_assessment = object_tracker.print_assessment()

            # Displaying results
            st.subheader("Analysis Results")

            # Display results based on selected checkboxes
            if apply_body_language:
                st.markdown("### Body Language Summary")
                for key, item in body_language_summary.items():
                    st.write(f"- {key.capitalize()}: {item:.2f}%")

            if apply_audio_processing:
                st.markdown("### Stutter Detection Report")
                for item in classify_audio:
                    st.write(f"- {item['label'].capitalize()}: {item['score']*100:.2f}%")
                st.markdown("### Transcribed Text")
                st.write(f"\"{full_text}\"")
                st.markdown("### Text Summary")
                st.write(f"\"{text_summary[0]['summary_text']}\"")

            if apply_eye_contact:
                st.markdown("### Eye Contact Analysis")
                for label, percentage in eye_contact_results.items():
                    st.write(f"- {label.capitalize()}: {percentage:.2f}%")

            if apply_emotion_detection:
                st.markdown("### Emotion Recognition")
                for emotion, percentage in emotion_results.items():
                    st.write(f"- {emotion.capitalize()}: {percentage:.2f}%")

            if apply_hand_movement:
                st.markdown("### Hand Movement Report")
                st.write(f"Average Movement per Frame: {movement_report:.2f}")
                if movement_report < 10:
                    st.write("Presenter's hands are relatively stationary. Consider using more gestures to engage the audience.")
                elif movement_report >= 20:
                    st.write("Presenter's hands are moving too much. Consider using less gestures to avoid distracting the audience.")
                else:
                    st.write("Presenter's hands are adequately used.")

            if apply_object_tracking:
                st.markdown("### Body Movement Report")
                st.write(object_tracker_assessment)

if __name__ == "__main__":
    main()
