# Final Project

Welcome to the Final Project repository for [Your Course/Program Name]!

## Overview

This project aims to analyze various aspects of human behavior from a video, including body language, audio classification, eye contact, emotion detection, and hand movement analysis.

## Features

- **Body Language Recognition**: Detects and summarizes body language cues.
- **Audio Processing**: Classifies audio into stutter types, transcribes speech, and provides text summaries.
- **Eye Contact Detection**: Analyzes the frequency and duration of eye contact.
- **Emotion Detection**: Identifies and quantifies emotions from the video.
- **Hand Movement Detection**: Detects and assesses hand movements.
- **Object Tracking**: Tracks object movement within the video.

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
    cd FinalProject
    ```
2. Run the main script:
    ```sh
    python main.py
    ```

## File Structure

```
FinalProject/
│
├── BL/                            # Body Language data
│   └── coords.csv                 # Coordinates CSV file
├── EC/                            # Eye Contact data
│   └── Eye.csv                    # Eye CSV file
├── Py/                            # Python scripts
│   ├── AudioProcessor.py          # Script for audio processing
│   ├── BodyLanguageRecognizer.py  # Script for body language recognition
│   ├── EmotionDetector.py         # Script for emotion detection
│   ├── EyeContact.py              # Script for eye contact analysis
│   ├── HandMovementDetector.py    # Script for hand movement detection
│   ├── ObjectTracker.py           # Script for object tracking
│   └── main.py                    # Main script to run the analysis
├── requirements.txt               # Python dependencies
└── README.md                      # Project README file
```

## Contributing

We welcome contributions from the community! If you'd like to contribute to the project, please follow these guidelines:

- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes with descriptive messages.
- Open a pull request to the `master` branch.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or feedback regarding the project, feel free to contact us at [Contact Email Address].
