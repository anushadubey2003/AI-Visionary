from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import torch
import numpy as np
import cv2
from .utils import preprocess_image
from .model import predict

from .config import HISTORY_LIMIT

app = FastAPI(title="Visionary AI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Prediction history
prediction_history = []

def add_to_history(pred_list):
    global prediction_history
    for pred in pred_list:
        prediction_history.append(pred)
    if len(prediction_history) > HISTORY_LIMIT:
        prediction_history = prediction_history[-HISTORY_LIMIT:]

# Request model
class ImageRequest(BaseModel):
    image: str

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_np: np.ndarray):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Main endpoint
@app.post("/predict")
def predict_image(request: ImageRequest):
    image_tensor = preprocess_image(request.image)
    if image_tensor is None:
        return {"error": "Invalid image"}

    img_np = np.array(torch.permute(image_tensor[0], (1,2,0))*255, dtype=np.uint8)

    faces = detect_faces(img_np)
    results = []

    if len(faces) == 0:
        pred = predict(image_tensor)
        results.append({"bbox": None, **pred})
    else:
        for (x, y, w, h) in faces:
            face_crop = img_np[y:y+h, x:x+w]
            face_tensor = torch.tensor(
                (torch.from_numpy(face_crop)).permute(2,0,1),
                dtype=torch.float32
            ).unsqueeze(0) / 255.0
            pred = predict(face_tensor)
            results.append({"bbox": [int(x), int(y), int(w), int(h)], **pred})

    add_to_history(results)

    return {"predictions": results, "history": prediction_history}

@app.get("/")
def read_root():
    return {"message": "Visionary AI backend is running."}
