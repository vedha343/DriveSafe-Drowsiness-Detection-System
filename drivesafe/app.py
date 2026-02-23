import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- 1. SETTINGS ---
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 15

# MediaPipe Indices for EAR
# Left eye: [362, 385, 386, 263, 374, 380]
# Right eye: [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 386, 263, 374, 380]
R_EYE = [33, 160, 158, 133, 153, 144]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    # Extract coordinates
    pts = []
    for i in eye_indices:
        pt = landmarks[i]
        pts.append((pt.x * img_w, pt.y * img_h))
    
    # Standard EAR Formula: (dist1 + dist2) / (2 * dist3)
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

# --- 2. VIDEO PROCESSOR ---
class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_ear = calculate_ear(face_landmarks.landmark, L_EYE, w, h)
                right_ear = calculate_ear(face_landmarks.landmark, R_EYE, w, h)
                ear = (left_ear + right_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    self.counter += 1
                    if self.counter >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(img, "DROWSINESS ALERT!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    self.counter = 0
        
        return img

# --- 3. UI ---
st.set_page_config(page_title="DriveSafe", page_icon="🚗")
st.title("🚗 DriveSafe: AI Drowsiness Detection")
st.markdown("""
Detects eye fatigue in real-time. 
1. Click **Start** below.
2. Grant camera permissions.
""")

webrtc_streamer(key="drowsy-check", video_transformer_factory=DrowsinessTransformer)
