from pathlib import Path
import argparse

import torch
from torchvision import transforms
from PIL import Image

from model import CameraConditionCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

from config import DEVICE, MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

RECOMMENDATIONS = {
    "clean": "Camera is operating normally.",
    "fog": "Heavy fog detected. Drive with caution.",
    "rain": "Rain detected. Visibility may be reduced.",
    "motion_blur": "Motion blur detected. Inspect camera stability.",
    "defocus_blur": "Camera appears out of focus. Recalibration recommended.",
    "overexposure": "Strong light detected. Camera image may be saturated.",
    "dirty_lens": "Lens contamination detected. Clean the camera.",
    "partial_occlusion": "Camera is partially blocked. Remove the obstruction.",
}

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = CameraConditionCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------------------------------------
# Argument Parser
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)

args = parser.parse_args()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

image = Image.open(args.image).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)

predicted_class = CLASS_NAMES[prediction.item()]

print("=" * 40)
print(f"Prediction : {predicted_class}")
print(f"Confidence : {confidence.item() * 100:.2f}%")
print()
print("Recommendation:")
print(RECOMMENDATIONS[predicted_class])
print("=" * 40)