from moviepy.editor import VideoFileClip
from transformers import pipeline
import whisper
import os

class AudioProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.output_audio_path = "extracted_audio.wav"
        self.audio_classification_model = pipeline("audio-classification", model="HareemFatima/distilhubert-finetuned-stutterdetection")
        self.transcription_model = whisper.load_model("base")
        self.summarization_model = pipeline("summarization", model="knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM")
    
    def extract_audio(self):
        if os.path.exists(self.output_audio_path):
            os.remove(self.output_audio_path)
            print(f"Existing audio file '{self.output_audio_path}' removed.")

        video_clip = VideoFileClip(self.video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(self.output_audio_path, codec='pcm_s16le')
        video_clip.close()
        print("Audio extracted and saved to:", self.output_audio_path)
    
    def classify_audio(self):
        predicted_label = self.audio_classification_model(self.output_audio_path)
        # print("Predicted Labels:")
        # for label in predicted_label:
        #     print(f"Label: {label['label']}, Score: {label['score']:.4f}")
        return predicted_label

    def transcribe_audio(self):
        result = self.transcription_model.transcribe(self.output_audio_path)
        # print("Transcribed Text:")
        # print(result["text"])
        return result["text"]
    
    def summarize_text(self, text):
        summarized_text = self.summarization_model(text)
        # print("Summarized Text:")
        # print(summarized_text)
        return summarized_text
