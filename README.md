
Visionary – AI Age & Gender Prediction

Visionary is a real-time AI-powered application that predicts a person’s age range and gender from webcam images. Built with Python, PyTorch, FastAPI, and OpenCV, this project demonstrates an end-to-end AI system with an interactive frontend and production-ready backend.

Key Features

Real-Time Webcam Capture: Uses webcam input instead of static images for live predictions.
Multi-Face Detection: Detects multiple faces in a single frame and predicts each face’s age and gender.
Confidence Scores: Each prediction includes a confidence score for reliability.
Prediction History: Stores and displays the last 10 predictions for better tracking.
Interactive Frontend: Clean, responsive UI with dark/light mode toggle and live prediction updates.
Error Handling: Robust preprocessing and validation to handle invalid inputs safely.



Tech Stack
Backend: Python, FastAPI, PyTorch
Frontend: HTML, CSS, JavaScript
Computer Vision: OpenCV for face detection

How It Works
The webcam captures a live image.
Image is converted to a PyTorch tensor and preprocessed.
Multi-face detection identifies all faces in the frame.
Each face is passed through a CNN-based PyTorch model to predict age range and gender.
Predictions are returned with confidence scores and displayed in the frontend along with history.
