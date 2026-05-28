import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle
import numpy as np
import pyttsx3
import threading

# 1. Initialize Text-to-Speech Engine
engine = pyttsx3.init()

def speak(text):
    def target():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=target, daemon=True).start()

# 2. Load your custom trained Machine Learning Model
with open('sign_language_model.p', 'rb') as f:
    model = pickle.load(f)

# 3. Initialize MediaPipe Hand Landmarker Task
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_spoken = ""

print("🚀 Starting real-time translator app... Press 'q' in the window to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
        
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Format frame for MediaPipe Tasks API
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Predict landmarks
    detection_result = detector.detect(mp_image)
    
    display_text = "Looking for signs..."
    
    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Flatten the 21 landmarks into a single 42-element array
        landmarks_row = []
        for lm in hand_landmarks:
            landmarks_row.extend([lm.x, lm.y])
            
            # Visual feedback: Draw tracking dots
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            
        # Convert list to a format Scikit-Learn expects: 2D array [[x1, y1, x2, y2...]]
        prediction_features = np.array([landmarks_row])
        
        # Make real-time prediction using our trained model
        predicted_sign = model.predict(prediction_features)[0]
        display_text = f"Sign: {predicted_sign}"
        
        # Trigger speech only when the user switches to a brand-new sign
        if predicted_sign != last_spoken:
            speak(predicted_sign)
            last_spoken = predicted_sign
            
    # Draw transparent overlay background box for clear UI text readability
    cv2.rectangle(frame, (0, 0), (320, 65), (0, 0, 0), -1)
    cv2.putText(frame, display_text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Real-Time Sign Language to Speech Translator', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()