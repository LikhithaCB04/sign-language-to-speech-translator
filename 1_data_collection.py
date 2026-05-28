import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import numpy as np

# CONFIGURATION
SIGNS = ["Hello", "Thank You", "Yes", "No", "Help", "Please","OK","I Love You","Stop","Good","MISS U"]  # List of signs to collect
SAMPLES_PER_SIGN = 100 

# Initialize the new MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
csv_file = open('sign_data.csv', 'a', newline='')
writer = csv.writer(csv_file)

print("Starting Data Collection (New API)...")

for sign in SIGNS:
    print(f"\nPreparation: Get ready to make the sign for: '{sign}'")
    print("Press 's' to start recording this sign when ready.")
    
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Ready for '{sign}'? Press 's' to start.", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Data Collection Pipeline', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    count = 0
    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        
        # Convert frame to MediaPipe Image format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect landmarks
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            landmarks_row = []
            
            # Extract 21 coordinates (x, y)
            for lm in hand_landmarks:
                landmarks_row.extend([lm.x, lm.y])
            
            landmarks_row.append(sign)
            writer.writerow(landmarks_row)
            count += 1
            
            # Draw tracking points manually for visual feedback
            for lm in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            
        cv2.putText(frame, f"Recording '{sign}': {count}/{SAMPLES_PER_SIGN}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Data Collection Pipeline', frame)
        cv2.waitKey(1)

print("\nData collection complete! 'sign_data.csv' generated cleanly.")
csv_file.close()
cap.release()
cv2.destroyAllWindows()