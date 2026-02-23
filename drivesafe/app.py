from flask import Flask, render_template, Response
import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import pygame  # For playing the alarm sound
import os

# --- 1. CONFIGURATION ---
EYE_AR_THRESH = 0.25        # If EAR is below this, eyes are "closed"
EYE_AR_CONSEC_FRAMES = 20   # Number of frames eyes must be closed to trigger alarm

# Initialize Flask App
app = Flask(__name__)

# Initialize Pygame Mixer for Alarm
pygame.mixer.init()

# Check if alarm file exists to prevent errors
if os.path.exists("static/alarm.wav"):
    alarm_sound = pygame.mixer.Sound("static/alarm.wav")
else:
    print("Warning: static/alarm.wav not found! Audio will not play.")
    alarm_sound = None

# --- 2. SETUP ML MODELS ---
# Ensure the .dat file is in the same folder as app.py
predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(predictor_path):
    print(f"ERROR: {predictor_path} not found. Please download it.")
    exit()

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Grab the indexes of the facial landmarks for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# --- 3. HELPER FUNCTIONS ---
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def generate_frames():
    camera = cv2.VideoCapture(0) # Use 0 for webcam
    
    COUNTER = 0
    ALARM_ON = False

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Convert to grayscale (needed for Dlib)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the EAR together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # Draw Eyes (Green Contours)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if eye aspect ratio is below the blink threshold
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                # If the eyes were closed for a sufficient number of frames
                # then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        if alarm_sound:
                            alarm_sound.play(-1) # Play in loop

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                COUNTER = 0
                ALARM_ON = False
                if alarm_sound:
                    alarm_sound.stop()

            # Display EAR on screen
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- 4. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Allows access from phone if connected to same WiFi
    app.run(host='0.0.0.0', port=5000, debug=True)
