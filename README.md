# DriveSafe-Drowsiness-Detection-System# DriveSafe-Web-Based-Real-Time-Drowsiness-Detection-System
🚗 DriveSafe: Web-Based Real-Time Drowsiness Detection System
DriveSafe is a real-time web application designed to prevent accidents caused by driver fatigue. It uses computer vision and machine learning techniques to monitor the driver's eyes through a webcam and triggers an audio alert if drowsiness is detected.

📖 Table of Contents
About the Project

Features

How It Works

Tech Stack

Installation & Setup

Usage

Project Structure

Future Enhancements

🧐 About the Project
Drowsy driving is a major cause of road accidents. DriveSafe aims to solve this by providing a lightweight, web-based solution that requires no expensive hardware—just a standard webcam or smartphone camera.

The system calculates the Eye Aspect Ratio (EAR) to determine if the eyes are closed for a prolonged period (indicative of microsleep or drowsiness).

✨ Features
Real-Time Monitoring: Processes video feed instantly with low latency.

Web Interface: Accessible via any browser (Chrome/Edge/Firefox).

Audio Alarm: Plays a loud alert sound when drowsiness is detected.

Visual Indicators: Displays current status ("Active" vs "Drowsy") on the dashboard.

No Internet Required: Runs locally on the machine for privacy and speed.

🧠 How It Works
The system relies on the Eye Aspect Ratio (EAR) formula.

Face Detection: The system detects the face using Dlib's 68-point landmark predictor.

Eye Mapping: It extracts the coordinates of the left and right eyes.

EAR Calculation: It calculates the vertical and horizontal distances between eye eyelids.

If the EAR drops below a threshold (e.g., 0.25) for a specific number of consecutive frames (e.g., 20 frames), the system classifies it as "Drowsy".

Alert: An alarm is triggered until the eyes open again.

🛠 Tech Stack
Language: Python 3.x

Web Framework: Flask (to serve the video feed to the web page)

Computer Vision: OpenCV (cv2), Dlib

Frontend: HTML5, CSS3, JavaScript (Bootstrap for styling)

Audio: Pygame (for playing the alarm sound)

⚙️ Installation & Setup
Prerequisites
Ensure you have Python installed. You will also need CMake (required for compiling Dlib).

Step 1: Clone the Repository
Bash

git clone https://github.com/your-username/drivesafe.git
cd drivesafe
Step 2: Create a Virtual Environment (Optional but Recommended)
Bash

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
Step 3: Install Dependencies
Bash

pip install -r requirements.txt
Note: If dlib fails to install, try installing CMake first (pip install cmake) or download a pre-compiled .whl file for your specific Python version.

Step 4: Download the Shape Predictor
Download the shape_predictor_68_face_landmarks.dat file (it is too large for GitHub) and place it in the root folder.

Download Link (Official Dlib) (Extract it after downloading).

🚀 Usage
Run the Application:

Bash

python app.py
Open the Web Interface: Open your browser and go to:

http://127.0.0.1:5000/
Start Driving (Simulated):

Allow camera permissions.

Close your eyes for 3 seconds to test the alarm.

📂 Project Structure
drivesafe/
│
├── static/
│   ├── css/
│   │   └── style.css       # Styling for the web page
│   ├── js/
│   │   └── script.js       # Frontend logic
│   └── alarm.wav           # Audio file for the alert
│
├── templates/
│   └── index.html          # Main dashboard HTML
│
├── app.py                  # Main Flask application
├── camera.py               # Logic for OpenCV and Dlib detection
├── shape_predictor_68_face_landmarks.dat  # ML Model (Download separately)
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
🔮 Future Enhancements
Yawn Detection: Add mouth aspect ratio to detect yawning.

Head Tilt Detection: Detect if the driver's head is dropping.

Mobile App: Convert the web app into a dedicated Android/iOS app.

SMS Alert: Send an emergency SMS to a contact if drowsiness is detected repeatedly.

🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request.

📜 License
This project is licensed under the MIT License.

Developed by [D V V SAIKRISHNA] Student at [CHAITHANYA DEEMED TO BE UNIVERSITY]
