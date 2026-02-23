import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="DriveSafe System", page_icon="🚘")
st.title("🚘 DriveSafe: Real-Time Drowsiness Detection")
st.markdown("This web application uses Computer Vision and Dlib to monitor driver alertness in real-time.")

# --- 2. SETUP ML MODELS (CACHED FOR SPEED) ---
@st.cache_resource
def load_models():
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        st.error(f"Error: `{predictor_path}` not found in the repository!")
        return None, None
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor

detector, predictor = load_models()

# --- 3. HELPER FUNCTIONS ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- 4. WEBRTC VIDEO PROCESSOR ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.COUNTER = 0
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20
        
        # Grab the indexes of the facial landmarks for the left and right eye
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if detector is None or predictor is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            # Draw Eyes (Green Contours)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if eyes are closed
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
                
                # If closed for enough frames, show alert
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    cv2.putText(img, "DROWSINESS ALERT! WAKE UP!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.COUNTER = 0

            # Display EAR on screen
            cv2.putText(img, f"EAR: {ear:.2f}", (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 5. START STREAMLIT UI ---
st.write("### Instructions:")
st.write("1. Click **START** below to turn on your webcam.")
st.write("2. Allow the browser to access your camera.")
st.write("3. The system will track your eyes. If you close them for too long, a warning will appear on the video feed.")

# Configuration for WebRTC to work on Cloud servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
