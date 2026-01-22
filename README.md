🎵 Motion-Controlled Spotify Controller

Control Spotify playback using real-time hand gestures captured via webcam.
This project uses computer vision and hand tracking to detect gestures and map them to Spotify actions such as play, pause, next track, previous track, and volume control.

🚀 Project Overview

Traditional media control requires physical interaction with keyboard, mouse, or mobile devices. This project enables touchless control of Spotify using hand gestures, making it useful for:

Smart environments

Accessibility solutions

Hands-free interaction

Human–Computer Interaction experiments

The system captures video input, detects hand landmarks, recognizes gestures, and triggers Spotify commands automatically.

🧠 How It Works

Webcam Capture
Live video feed is captured using OpenCV.

Hand Detection & Tracking
MediaPipe identifies hand landmarks in real time.

Gesture Recognition
Finger positions and distances are analyzed to classify gestures.

Command Mapping
Each gesture is mapped to a Spotify action.

Spotify Control
Spotify Web API or local automation executes playback commands.

✋ Supported Gestures
Gesture	Action
Open Palm	Play / Pause
Swipe Right	Next Track
Swipe Left	Previous Track
Pinch Up	Volume Up
Pinch Down	Volume Down

(Gestures can be customized easily in code.)

🛠 Tech Stack

Python

OpenCV – Video capture and frame processing

MediaPipe – Hand landmark detection

Spotify Web API / Spotipy

NumPy
