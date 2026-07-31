from pathlib import Path
import torch

# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Image Settings
# --------------------------------------------------

IMAGE_SIZE = (128, 128)

# --------------------------------------------------
# Classes
# --------------------------------------------------

CLASS_NAMES = [
    "clean",
    "defocus_blur",
    "dirty_lens",
    "fog",
    "motion_blur",
    "overexposure",
    "partial_occlusion",
    "rain",
]

# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_PATH = Path("outputs/models/best_model.pth")

# --------------------------------------------------
# Output Directories
# --------------------------------------------------

OUTPUT_MODELS = Path("outputs/models")
OUTPUT_METRICS = OUTPUT_MODELS / "metrics"
OUTPUT_PREDICTIONS = Path("outputs/predictions")