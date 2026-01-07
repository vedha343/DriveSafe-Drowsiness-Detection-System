import cv2
import time
from scipy.spatial import distance as dist

# TRYING TO IMPORT MEDIAPIPE SAFELY
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    HAS_MEDIAPIPE = True
except AttributeError:
    print("Warning: MediaPipe solutions not found. Falling back to OpenCV only.")
    HAS_MEDIAPIPE = False

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
        # --- STRATEGY A: MEDIAPIPE (High Accuracy) ---
        if HAS_MEDIAPIPE:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        # --- STRATEGY B: OPENCV CASCADES (Backup) ---
        else:
            # We will use standard Haar Cascades if MediaPipe fails
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Logic Variables
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20
        self.COUNTER = 0
        self.status = "Active"

    def __del__(self):
        self.video.release()

    def eye_aspect_ratio(self, eye_points):
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        return (A + B) / (2.0 * C)

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None, "Error"

        # ---------------------------------------------------------
        # OPTION 1: MEDIAPIPE LOGIC (Best for Python 3.10-3.12)
        # ---------------------------------------------------------
        if HAS_MEDIAPIPE:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            self.status = "No Face" # Default
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, c = image.shape
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                    # Eye indices
                    right_eye_indices = [33, 160, 158, 133, 153, 144]
                    left_eye_indices =  [362, 385, 387, 263, 373, 380]

                    rightEye = [landmarks[i] for i in right_eye_indices]
                    leftEye = [landmarks[i] for i in left_eye_indices]

                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    avgEAR = (leftEAR + rightEAR) / 2.0

                    if avgEAR < self.EYE_AR_THRESH:
                        self.COUNTER += 1
                        if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                            self.status = "Drowsy"
                            cv2.putText(image, "DROWSINESS ALERT!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        self.COUNTER = 0
                        self.status = "Active"

                    # Draw dots
                    for (x, y) in rightEye + leftEye:
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

        # ---------------------------------------------------------
        # OPTION 2: OPENCV ONLY LOGIC (Backup for Python 3.13)
        # ---------------------------------------------------------
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            self.status = "No Face"
            
            for (x, y, w, h) in faces:
                self.status = "Active" # Found a face, assume active initially
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]
                
                # Detect eyes inside the face area
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                # If face is found but NO eyes are found, assume closed/drowsy
                if len(eyes) == 0:
                    self.COUNTER += 1
                    if self.COUNTER >= 10: # Lower threshold for this method
                        self.status = "Drowsy"
                        cv2.putText(image, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.COUNTER = 0
                    self.status = "Active"
                    
                    # Draw rectangles around eyes
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes(), self.status