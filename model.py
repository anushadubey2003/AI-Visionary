import torch
import torch.nn as nn
import numpy as np
from .config import MODEL_PATH, IMG_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AGE_RANGES = ["0-10","11-20","21-30","31-40","41-50","51-60","61-70","71+"]

class VisionaryModel(nn.Module):
    def __init__(self, num_age_classes=8, num_gender_classes=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.age_head = nn.Linear(32*(IMG_SIZE[0]//4)*(IMG_SIZE[1]//4), num_age_classes)
        self.gender_head = nn.Linear(32*(IMG_SIZE[0]//4)*(IMG_SIZE[1]//4), num_gender_classes)

    def forward(self,x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        return self.age_head(x), self.gender_head(x)

# Load model
model = VisionaryModel()
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model.to(device)
model.eval()

def predict(image_tensor: torch.Tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        age_logits, gender_logits = model(image_tensor)
        age_probs = torch.softmax(age_logits, dim=1).cpu().numpy()[0]
        gender_probs = torch.softmax(gender_logits, dim=1).cpu().numpy()[0]

    age_index = int(np.argmax(age_probs))
    gender_index = int(np.argmax(gender_probs))
    confidence = float(max(max(age_probs), max(gender_probs)))
    gender = "Male" if gender_index==0 else "Female"
    age_range = AGE_RANGES[age_index]

    return {"age_range": age_range, "gender": gender, "confidence": round(confidence,2)}
