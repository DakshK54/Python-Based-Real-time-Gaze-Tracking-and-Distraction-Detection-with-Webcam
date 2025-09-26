# Copilot Instructions for Python-Based Real-time Eye Tracking with Webcam

## Overview
This project implements a real-time driver gaze tracking system using OpenCV, MediaPipe, and a Random Forest model. The system monitors driver attention and detects distractions by analyzing eye movements captured via a webcam.

## Key Components
- **`Eye-Tracking-system.py`**: The main script that integrates all functionalities, including webcam input, eye tracking, and distraction detection.
- **OpenCV**: Used for image processing and feature extraction.
- **MediaPipe**: Provides pre-trained models for facial and eye landmark detection.
- **Random Forest Model**: Classifies driver attention states based on extracted features.

## Developer Workflows
### Running the System
1. Ensure all dependencies are installed (see below).
2. Run the main script:
   ```bash
   python Eye-Tracking-system.py
   ```

### Debugging
- Use print statements to inspect intermediate outputs, such as detected landmarks or classification results.
- Check webcam permissions if the video feed does not appear.

## Project-Specific Conventions
- **Landmark Indices**: Follow MediaPipe's standard indices for facial landmarks.
- **Feature Extraction**: Custom logic is implemented to extract features from eye landmarks for classification.
- **Real-time Processing**: The system is optimized for real-time performance; avoid adding blocking operations.

## External Dependencies
- **OpenCV**: For image processing.
- **MediaPipe**: For landmark detection.
- **Scikit-learn**: For the Random Forest model.

Install dependencies using:
```bash
pip install opencv-python mediapipe scikit-learn
```

## Integration Points
- **Webcam Input**: Captures real-time video feed.
- **MediaPipe Models**: Provides facial and eye landmark detection.
- **Random Forest Classifier**: Predicts driver attention state.

## Example Patterns
### Eye Landmark Detection
```python
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
# Process a frame
results = mp_face_mesh.process(frame)
```

### Feature Extraction
```python
# Extract specific eye landmarks
left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
```

### Classification
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
prediction = clf.predict(features)
```

## Notes
- Ensure the webcam is properly connected and accessible.
- Test the system in various lighting conditions to ensure robustness.

For further details, refer to the `README.md` file.