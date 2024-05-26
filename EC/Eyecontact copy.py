{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle \n",
    "import pandas as pd\n",
    "import numpy as np \n",
    "import mediapipe as mp # Import mediapipe\n",
    "import cv2 # Import opencv\n",
    "from collections import defaultdict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('Eye.pkl', 'rb') as f:\n",
    "    model = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class_counts = defaultdict(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class_counts = defaultdict(int)\n",
    "total_frames = 0\n",
    "\n",
    "cap = cv2.VideoCapture(0)\n",
    "mp_holistic = mp.solutions.holistic\n",
    "mp_drawing = mp.solutions.drawing_utils\n",
    "\n",
    "with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:\n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        if not ret:\n",
    "            break\n",
    "        \n",
    "        total_frames += 1\n",
    "        \n",
    "        # Recolor Feed\n",
    "        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "        image.flags.writeable = False\n",
    "\n",
    "        # Make Detections\n",
    "        results = holistic.process(image)\n",
    "\n",
    "        # Recolor image back to BGR for rendering\n",
    "        image.flags.writeable = True\n",
    "        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "\n",
    "        # Draw face landmarks\n",
    "        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,\n",
    "                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),\n",
    "                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))\n",
    "\n",
    "        # Draw right hand landmarks\n",
    "        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,\n",
    "                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),\n",
    "                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))\n",
    "\n",
    "        # Draw left hand landmarks\n",
    "        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,\n",
    "                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),\n",
    "                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))\n",
    "\n",
    "        # Draw pose landmarks\n",
    "        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,\n",
    "                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),\n",
    "                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))\n",
    "\n",
    "        # Export coordinates and make predictions\n",
    "        try:\n",
    "            # Extract Pose landmarks\n",
    "            pose = results.pose_landmarks.landmark\n",
    "            pose_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten().tolist()\n",
    "\n",
    "            # Extract Face landmarks\n",
    "            face = results.face_landmarks.landmark\n",
    "            face_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten().tolist()\n",
    "\n",
    "            # Concatenate rows\n",
    "            row = pose_row + face_row\n",
    "\n",
    "            # Make Detections\n",
    "            X = pd.DataFrame([row])\n",
    "            body_language_class = model.predict(X)[0]\n",
    "            body_language_prob = model.predict_proba(X)[0]\n",
    "\n",
    "            # Update class counts\n",
    "            class_counts[body_language_class] += 1\n",
    "\n",
    "            # Grab ear coords for annotation\n",
    "            coords = np.multiply(\n",
    "                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,\n",
    "                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y),\n",
    "                [640, 480]).astype(int)\n",
    "\n",
    "            # Draw rectangle and text for class prediction\n",
    "            cv2.rectangle(image, (coords[0], coords[1] + 5), (coords[0] + len(body_language_class) * 20, coords[1] - 30), (245, 117, 16), -1)\n",
    "            cv2.putText(image, body_language_class, coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)\n",
    "\n",
    "           # Display Class and Probability\n",
    "            class_box_color = (245, 117, 16)\n",
    "            text_color = (255, 255, 255)\n",
    "            text_size = 1\n",
    "            text_thickness = 2\n",
    "            text_font = cv2.FONT_HERSHEY_SIMPLEX\n",
    "\n",
    "            # Draw rectangle for class and probability display\n",
    "            cv2.rectangle(image, (0, 0), (250, 60), -1)\n",
    "\n",
    "            # Draw text for 'CLASS'\n",
    "            cv2.putText(image, 'CLASS', (50, 10), text_font, 0.2, text_color, 1, cv2.LINE_AA)\n",
    "\n",
    "            # Extract and display class name\n",
    "            class_name = body_language_class.split(' ')[0]\n",
    "            cv2.putText(image, class_name, (90, 40), text_font, text_size, text_color, text_thickness, cv2.LINE_AA)\n",
    "\n",
    "            # Draw text for 'PROB'\n",
    "            cv2.putText(image, 'PROB', (15, 12), text_font, 0.5, text_color, 1, cv2.LINE_AA)\n",
    "\n",
    "            # Extract and display maximum probability\n",
    "            max_prob = round(max(body_language_prob), 2)\n",
    "            cv2.putText(image, str(max_prob), (10, 40), text_font, text_size, text_color, text_thickness, cv2.LINE_AA)\n",
    "\n",
    "        except Exception as e:\n",
    "            print(f'Error: {e}')\n",
    "            pass\n",
    "\n",
    "        # Display the resulting frame\n",
    "        cv2.imshow('Raw Webcam Feed', image)\n",
    "\n",
    "        if cv2.waitKey(10) & 0xFF == ord('q'):\n",
    "            break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print the summary of class occurrences\n",
    "for class_name, count in class_counts.items():\n",
    "    percentage = (count / total_frames) * 100\n",
    "    print(f'{class_name}: {percentage:.2f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "gpu-env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
