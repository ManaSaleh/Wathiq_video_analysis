import streamlit as st
import openai
import os
from BodyLanguageRecognizer import BodyLanguageRecognizer
from AudioProcessor import AudioProcessor
from Eyecontact import EyeContact
from EmotionDetector import EmotionDetector
from HandMovementDetector import HandMovementDetector
from object_tracker import ObjectTracker
from openai.error import RateLimitError

# Set up OpenAI API key
openai.api_key = 'Use Your API Key'

# Set page config
st.set_page_config(page_title="Interview/Presentation Analysis App", page_icon=":video_camera:")

logo_path = "logo.png"
col1, col2, col3 = st.columns([1, 2, 1])

# Display the logo image
with col2:
    st.image(logo_path, width=350)

topics = st.text_input('Main topics:')

def gpt3(prompt, analysis_summary):
    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant providing presentation feedback."},
            {"role": "user", "content": f"Provide advice based on the following analysis results:\n\n{analysis_summary}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the correct GPT-3.5 model
            messages=messages,
            max_tokens=150
        )
    except RateLimitError:
        return "Rate limit exceeded. Please check your OpenAI plan and billing details."

    return response.choices[0].message['content'].strip()

# Function to load video files from a directory
def load_videos(directory):
    videos = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith((".mp4", ".avi", ".mov"))]
    return videos

def main():
    st.sidebar.title("Upload Video")
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        video_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success("File uploaded successfully!")
    else:
        st.sidebar.write("Or choose from example videos below:")

    video_directory = "videos"
    example_videos = load_videos(video_directory)
    selected_video = st.sidebar.selectbox("Select an example video to play:", example_videos) if example_videos else None

    if uploaded_file is not None or selected_video is not None:
        video_path = video_path if uploaded_file is not None else selected_video

        st.sidebar.title("Select Models to Apply")
        apply_body_language = st.sidebar.checkbox("Body Language Recognition", value=True)
        apply_audio_processing = st.sidebar.checkbox("Audio Processing (Stutter, Dictation, Summarization)", value=True)
        apply_eye_contact = st.sidebar.checkbox("Eye Contact Detection", value=True)
        apply_emotion_detection = st.sidebar.checkbox("Emotion Detection", value=True)
        apply_hand_movement = st.sidebar.checkbox("Hand Movement Detection", value=True)
        apply_object_tracking = st.sidebar.checkbox("Body Movement Detection", value=True)

        if st.sidebar.button("Analyze Video"):
            st.subheader("Video Playback")
            st.video(video_path)
            st.subheader("Processing video...")
            analyze_video(video_path, apply_body_language, apply_audio_processing, apply_eye_contact, apply_emotion_detection, apply_hand_movement, apply_object_tracking)

def analyze_video(video_path, apply_body_language, apply_audio_processing, apply_eye_contact, apply_emotion_detection, apply_hand_movement, apply_object_tracking):
    body_language_summary = {}
    movement_report = 0
    object_tracker_assessment = {}
    eye_contact_results = {}
    emotion_results = {}
    classify_audio = []
    full_text = ""
    text_summary = ""

    if apply_body_language:
        recognizer = BodyLanguageRecognizer(video_path)
        recognizer.process_video()
        body_language_summary = recognizer.print_summary() 

    if apply_hand_movement:
        hand_movement_detector = HandMovementDetector(video_source=video_path)
        hand_movement_results = hand_movement_detector.detect_movement()
        movement_report = hand_movement_detector.generate_report()

    if apply_object_tracking:
        model_path = 'yolov8n.pt'
        object_tracker = ObjectTracker(model_path, video_path)
        object_tracker.process_video()
        object_tracker_assessment = object_tracker.print_assessment()

    if apply_eye_contact:
        eye_contact_detector = EyeContact(video_source=video_path)
        eye_contact_results, _ = eye_contact_detector.run()

    if apply_emotion_detection:
        emotion_detector = EmotionDetector(video_source=video_path)
        emotion_results = emotion_detector.run()

    if apply_audio_processing:
        audio_processor = AudioProcessor(video_path)
        audio_processor.extract_audio()
        if os.path.exists(audio_processor.output_audio_path):
            classify_audio = audio_processor.classify_audio()
            full_text = audio_processor.transcribe_audio()
            text_summary = audio_processor.summarize_text(full_text)

    display_results(body_language_summary, movement_report, object_tracker_assessment, eye_contact_results, emotion_results, classify_audio, full_text, text_summary)

def display_results(body_language_summary, movement_report, object_tracker_assessment, eye_contact_results, emotion_results, classify_audio, full_text, text_summary):
    st.markdown("### Body Language Summary")
    for key, item in body_language_summary.items():
        st.write(f"- {key.capitalize()}: {item:.2f}%")

    st.markdown("### Hand Movement Report")
    st.write(f"Average Movement per Frame: {movement_report:.2f}")
    if movement_report < 10:
        st.write("Presenter's hands are relatively stationary. Consider using more gestures to engage the audience.")
    elif movement_report >= 20:
        st.write("Presenter's hands are moving too much. Consider using fewer gestures to avoid distracting the audience.")
    else:
        st.write("Presenter's hands are adequately used.")

    st.markdown("### Body Movement Report")
    st.write(object_tracker_assessment)

    advice = gpt3("Provide advice based on the following analysis results:", f"\n\nBody Language Summary:\n{body_language_summary}\n\nHand Movement Report:\n{movement_report}\n\nBody Movement Report:\n{object_tracker_assessment}")
    st.write("### Advice")
    st.write(advice)

    st.markdown("### Eye Contact Analysis")
    for label, percentage in eye_contact_results.items():
        st.write(f"- {label.capitalize()}: {percentage:.2f}%")

    advice = gpt3("Provide advice based on the following analysis results:", f"\n\nEye Contact Analysis:\n{eye_contact_results}")
    st.write("### Advice")
    st.write(advice)

    st.markdown("### Emotion Recognition")
    for emotion, percentage in emotion_results.items():
        st.write(f"- {emotion.capitalize()}: {percentage:.2f}%")

    advice = gpt3("Provide advice based on the following analysis results:", f"\n\nEmotion Recognition:\n{emotion_results}")
    st.write("### Advice")
    st.write(advice)

    st.markdown("### Stutter Detection Report")
    for item in classify_audio:
        st.write(f"- {item['label'].capitalize()}: {item['score']*100:.2f}%")

    advice = gpt3("Provide advice based on the following analysis results:", f"\n\nStutter Detection Report:\n{classify_audio}")
    st.write("### Advice")
    st.write(advice)

    st.markdown("### Transcribed Text")
    st.write(f"\"{full_text}\"")
    st.markdown(f"##### Main Topics: {topics}")
    gpt_input = f"Main Topics: {topics}\nTranscribed Text: {full_text}"
    advice = gpt3("Provide advice based on the following analysis results if main topics matching full text and what was missed:", gpt_input)

    st.write("### Advice")
    st.write(advice)

    st.markdown("### Text Summary")
    st.write(f"\"{text_summary[0]['summary_text']}\"")

if __name__ == "__main__":
    main()
