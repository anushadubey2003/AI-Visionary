import base64
import io
import numpy as np
from PIL import Image
import torch
from .config import IMG_SIZE

def preprocess_image(base64_str: str):
    """
    Converts a base64 image from frontend webcam into a PyTorch tensor.
    Output shape: (1, 3, H, W), values normalized [0,1]
    """
    try:
        # Remove base64 header if present
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # Decode image
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize(IMG_SIZE)

        # Convert to tensor
        image_tensor = torch.tensor(
            (torch.from_numpy(np.array(image))).permute(2,0,1),
            dtype=torch.float32
        ) / 255.0

        # Add batch dimension
        return image_tensor.unsqueeze(0)  # Shape: (1, 3, H, W)

    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return None
