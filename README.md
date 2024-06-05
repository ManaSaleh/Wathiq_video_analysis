---

<p align="center">
   <img src="PythonScripts/logo.png" alt="example1"/>
</p>

# Wathiq Interview/Presentation Analysis System

This project was created as the capstone project for [T5/Data Science and ML] by Tuwaiq/SDAIA

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->
## Table of Contents
* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [File Structure](#file-structure)
* [Libraries Used](#libraries-used)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
<!-- TOC end -->

## Overview

This project aims to analyze various aspects of human behavior from a video, including body language, hand movement, stutter detection, eye contact, emotion detection, as well as transcribe and summarize the spoken content.

## Features

- **Body Language Recognition**: Detects and summarizes body language cues.
- **Body Tracking**: Tracks body movement within the video.
- **Hand Movement Detection**: Detects and assesses hand movements.
- **Eye Contact Detection**: Analyzes the frequency and duration of eye contact.
- **Emotion Detection**: Identifies and quantifies emotions from the video.
- **Audio Processing**: Generates a stutter report, transcribes speech, and provides text summaries.
- **Recommendations**: Generates advice from OpenAI based on results.

## Installation

To get started with the project, follow these steps:

1. Clone this repository to your local machine:
    ```sh
    git clone https://github.com/ManaSaleh/FinalProject.git
    ```
2. Install the required dependencies by running:
    ```sh
    pip install -r requirements.txt
    ```
3. Ensure you have ffmpeg installed on your system. You can download it from [here](https://ffmpeg.org/download.html).

## Usage

To use the project:

1. Navigate to the project directory:
    ```sh
    cd Wathiq_video_analysis/PythonScripts
    ```
2. Run the following command to access the Streamlit app:
    ```sh
    streamlit run streamlit_app.py
    ```
3. Use preloaded example videos or upload your own, select the desired models to run your video through, and click analyze:

<p align="center">
   <img src="https://github.com/ManaSaleh/VideoAnalisis/assets/130837413/3814e16b-2aa7-49d3-985c-cd45d5bca20b" alt="example1" width="200"/>
</p>

**Note**: Some of the models may be inapplicable based on the use cases, such as:
- Eye detection is inapplicable if the person in the video is looking at another person when talking instead of the camera.
- Body movement and body language may misbehave if used on a stationary sitting user.
- Body language might misbehave if the camera level is lower than eye level, confusing itself when the person is looking up vs looking down vs looking straight ahead.

4. Click on Analyze and you can watch the original video while waiting for all the models to finish working:

<p align="center">
   <img src="https://github.com/ManaSaleh/VideoAnalisis/assets/130837413/f9c5b308-f9ef-4748-9583-302f77584501" alt="example2" width="900"/>
</p>

5. Read the generated report to improve your performance in interviews or presentations. Sample results:

<p align="left">
   
   **Body Language Summary**
   - The head looks to the left: 17.14%
   - Hands behind the back: 23.89%
   - Normal: 34.84%
   - Head looks down: 9.50%
   - Hand on waist: 1.21%
   - Hands joined together: 12.29%
   - The head looks to the right: 0.35%
   
   **Hand Movement Report**
   - Average Movement per Frame: 0.08
   - Presenter's hands are relatively stationary. Consider using more gestures to engage the audience.
   
   **Body Movement Report**
   - Total Movement Distance: 10363.02
   - Average Movement per Frame: 1.63
   - Assessment: Presenter is relatively stationary. Consider using more gestures and movements to engage the audience.
   
   **Recommendations for body language, hand movement, and body movement**
   - Incorporate more gestures and movements during the presentation to engage the audience.
   - Avoid being too stationary and make use of the stage area for more dynamic delivery.
</p>

## Demo Video
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/UaGAmifS8nk/0.jpg)](https://www.youtube.com/watch?v=UaGAmifS8nk)

## File Structure

```
FinalProject/
│
├── AudioProcessing/               # Audio Processing data
|   ├── A.ipynb                    # Audio Processing Notebook
|   └── extracted_audio.wav        # Extracted audio from notebook run
├── BodyLanguage/                  # Body Language data
|   ├── BL.ipynb                   # Body Language Notebook
|   ├── coords.csv                 # Coordinates CSV file
|   └── body_language.pkl          # Model Weights for Body Language model
├── EyeContact/                    # Eye Contact data
|   ├── Eyecontact.ipynb           # Eye Contact Notebook
|   ├── Eye.csv                    # Eye CSV file
|   └── Eye.pkl                    # Model Weights for Eye contact model
├── FaceEmotion/                   # Face Emotion data
|   ├── FaceEmotion.ipynb          # Face Emotion Notebook
|   └── bestFace.pt                # Model Weights for Face Emotion Model
├── HandMovement/                  # Hand Movement data
|   ├── hands.ipynb                # Hand Movement Notebook
|   └── hand_movement_data.csv     # Saved results from notebook run
├── PythonScripts/                 # Python scripts
|   ├── AudioProcessor.py          # Script for audio processing
|   ├── BodyLanguageRecognizer.py  # Script for body language recognition
|   ├── EmotionDetector.py         # Script for emotion detection
|   ├── EyeContact.py              # Script for eye contact analysis
|   ├── HandMovementDetector.py    # Script for hand movement detection
|   ├── ObjectTracker.py           # Script for object tracking
|   ├── main.py                    # Main script to run the analysis in CLI
|   └── streamlit_app.py           # Streamlit App script
├── requirements.txt               # Python dependencies
└── README.md                      # Project README file
```

## Libraries Used

1. MediaPipe
2. DeepFace
3. Ultralytics (YOLO)
4. Streamlit

## Contributing

We welcome contributions from the community! If you'd like to contribute to the project, please follow these guidelines:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes with descriptive messages.
- Open a pull request to the `master` branch.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or feedback regarding the project, feel free to contact us at [breued1@gmail.com](mailto:breued1@gmail.com) [adamixa@gmail.com](mailto:adamixa@gmail.com).

---

Feel free to modify this further according to your preferences and specific requirements.
