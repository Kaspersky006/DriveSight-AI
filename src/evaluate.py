from pathlib import Path

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

import matplotlib.pyplot as plt

from dataset import test_loader, class_names
from model import CameraConditionCNN


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("outputs/models/best_model.pth")

OUTPUT_DIR = Path("outputs/models/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = CameraConditionCNN(num_classes=len(class_names))

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)

model.eval()


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# --------------------------------------------------
# Accuracy
# --------------------------------------------------

accuracy = accuracy_score(all_labels, all_predictions)

print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# --------------------------------------------------
# Classification Report
# --------------------------------------------------

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names
)

print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
) as f:

    f.write(f"Test Accuracy: {accuracy*100:.2f}%\n\n")
    f.write(report)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45,
    colorbar=False,
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
)

plt.close()

print("\nEvaluation complete.")
print(f"Results saved to: {OUTPUT_DIR}")