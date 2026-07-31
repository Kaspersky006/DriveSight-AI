from pathlib import Path
import argparse
import csv

import torch
from torchvision import transforms
from PIL import Image

from model import CameraConditionCNN
from config import (
    DEVICE,
    MODEL_PATH,
    CLASS_NAMES,
    IMAGE_SIZE,
    OUTPUT_PREDICTIONS,
)


# --------------------------------------------------
# Output Directory
# --------------------------------------------------

OUTPUT_PREDICTIONS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Image Transform
# --------------------------------------------------

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

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

# --------------------------------------------------
# Command Line Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Predict camera condition for all images in a folder."
)

parser.add_argument(
    "--folder",
    required=True,
    help="Path to the folder containing images."
)

args = parser.parse_args()

folder = Path(args.folder)

if not folder.exists():
    raise FileNotFoundError(f"Folder not found: {folder}")

# --------------------------------------------------
# Predict Images
# --------------------------------------------------

results = []

valid_extensions = {".jpg", ".jpeg", ".png"}

for image_path in sorted(folder.iterdir()):

    if image_path.suffix.lower() not in valid_extensions:
        continue

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    results.append([
        image_path.name,
        CLASS_NAMES[prediction.item()],
        round(confidence.item() * 100, 2)
    ])

# --------------------------------------------------
# Save CSV
# --------------------------------------------------

csv_path = OUTPUT_PREDICTIONS / "predictions.csv"

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "Prediction",
        "Confidence (%)"
    ])

    writer.writerows(results)

# --------------------------------------------------
# Summary
# --------------------------------------------------

print(f"\nProcessed {len(results)} images.")
print(f"Results saved to: {csv_path}")