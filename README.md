# Real-Time Sign Language to Speech Translator

An end-to-end Computer Vision and Machine Learning application that extracts hand landmark structures in real-time, classifies distinct hand gestures using a trained machine learning model, and translates those predictions into synthesized spoken audio.

## 🚀 Key Features
- **Real-Time Landmark Tracking:** Leverages the MediaPipe Tasks API to extract 21 distinct 3D spatial points on a single hand with minimal latency.
- **Machine Learning Classification:** Utilizes a Scikit-Learn Random Forest Classifier to dynamically predict gestures from structured coordinate frames.
- **Asynchronous Text-to-Speech:** Features multi-threaded `pyttsx3` audio execution, ensuring the live webcam feed remains perfectly smooth and frame rates don't lag during speech output.

## 🛠️ Tech Stack
- **Language:** Python 3.14
- **Computer Vision:** OpenCV, MediaPipe Tasks
- **Machine Learning & Data:** Scikit-Learn, Pandas, NumPy
- **Audio Output:** Pyttsx3 (Multi-threaded)

---

## 📦 Local Project Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/LikhithaCB04/sign-language-to-speech-translator.git](https://github.com/LikhithaCB04/sign-language-to-speech-translator.git)
   cd sign-language-to-speech-translator