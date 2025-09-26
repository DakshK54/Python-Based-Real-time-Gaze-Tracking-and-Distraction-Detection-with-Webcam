# 🚗 Advanced Driver Gaze Monitoring System
This repository contains a Python-based application for real-time gaze tracking, specifically designed as a proof-of-concept for a driver monitoring system. It uses computer vision and machine learning to determine where a person is looking on a screen, identify blinks, and infer the driver's level of attention.

### ✨ Key Features
- **High-Fidelity Landmark Detection**: Utilizes MediaPipe Face Mesh to get precise 3D landmarks for the face, eyes, and iris.

- **Machine Learning-Powered Calibration:** Instead of simple geometric calculations, it employs a RandomForestRegressor model from Scikit-learn. The user performs a 9-point calibration, which trains the model to map gaze vectors and head pose to specific screen coordinates, providing person-specific accuracy.

- **Head Pose Compensation:** Incorporates 3D head pose estimation using OpenCV's solvePnP function, making the gaze prediction more robust to head movements.

- **Blink Detection:** Monitors the Eye Aspect Ratio (EAR) to reliably detect blinks, which can be used for drowsiness detection.

- **Interactive UI:** A clean interface built with OpenCV shows calibration status, eye status (blinking/open), gaze location, and a smoothed, real-time gaze point on the screen.

### 🛠️ How It Works
- **Detection:** MediaPipe processes the webcam feed to detect facial landmarks.

- **Feature Extraction:** The code calculates a gaze vector based on the iris's relative position within the eye landmarks and combines it with the head rotation vector.

- **Calibration:** The user looks at 9 points on the screen and presses a key. The application collects the feature vectors and their corresponding screen coordinates.

- **Training:** A RandomForestRegressor model is trained on this collected data.

- **Prediction:** Once calibrated, the model predicts the (x, y) screen coordinates for new gaze and head pose vectors in real-time.

- **Visualization:** The predicted gaze point is smoothed and displayed on the screen, along with status information in a dedicated UI panel.


### 📋 Prerequisites
Before you begin, ensure you have the following installed:

- Conda: The run.sh script is optimized to use Conda to create a consistent Python 3.9 environment. You can install it from the official Anaconda website.

- (Alternative) Python 3.9: If you do not have Conda, the script will fall back to using venv. Please ensure your system's python3 command points to a Python 3.9 installation for best results.

- A webcam: The application requires a webcam to capture video input.

### 🚀 Getting Started: Setup & Execution
Follow these simple steps to get the application running. The provided shell script automates the entire setup process.

1. Clone the Repository

- First, clone this repository to your local machine:

```console
git clone <your-repository-url>
cd <your-repository-directory>
```

2. Make the Run Script Executable

- In your terminal, give the run.sh script execute permissions. You only need to do this once.

```console
chmod +x run.sh
```

3. Run the Application

- Execute the script. It will handle the complete setup and launch the program.

```console
./run.sh
```

###  What does the script do?

It checks for a Conda installation.

If found, it creates a new Conda environment named gaze_tracker_env with Python 3.9.

It activates the environment and installs all necessary libraries from requirements.txt.

Finally, it launches the gaze tracking application.

If Conda is not found, it will attempt to use a local venv environment as a fallback.

### 💻 How to Use the Application
Once the window appears, follow these steps:

- Calibration:

    -  You will see a pulsing yellow dot on the screen. This is a calibration target.

    - Focus your gaze directly on this dot.

    - Press the c key on your keyboard. A green circle will flash to confirm the point has been captured.

    - The target will move to the next position. Repeat this process for all 9 points.

    - Once all 9 points are captured, the model will be trained, and the status will change to CALIBRATED.

- Tracking:

    - After calibration, a white, semi-transparent dot will appear, tracking your gaze in real time.

    - The UI panel on the left will show your eye status (Blinking/Eyes Open) and where your gaze is located (Road, Infotainment, or Distracted).

- Controls:

    - **c:** Capture a calibration point.

    - **r:** Reset the calibration process and start over.

    - **q:** Quit the application.