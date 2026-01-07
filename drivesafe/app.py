import streamlit as st
import cv2
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- 1. SETUP & CONSTANTS ---
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- 2. THE VIDEO PROCESSOR CLASS ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.status = "Active"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            # Draw Eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            # Drowsiness Logic
            if ear < EYE_AR_THRESH:
                self.counter += 1
                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.status = "DROWSY"
            else:
                self.counter = 0
                self.status = "Active"

        return img

# --- 3. STREAMLIT UI ---
st.title("🚗 DriveSafe: Drowsiness Detection")
st.write("Click 'Start' to enable your webcam.")

webrtc_streamer(key="example", video_transformer_factory=VideoProcessor)

st.write("Ensure you have good lighting for accurate detection.")