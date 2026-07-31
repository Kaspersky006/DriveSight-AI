from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image size for training
IMAGE_SIZE = (128,128)

# Batch size
BATCH_SIZE = 32

# Dataset location
DATASET_ROOT = Path("data/generated_subset")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Load datasets
train_dataset = datasets.ImageFolder(
    DATASET_ROOT / "train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    DATASET_ROOT / "val",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    DATASET_ROOT / "test",
    transform=transform
)
class_names = train_dataset.classes

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

if __name__ == "__main__":
    print("Classes:")
    print(train_dataset.classes)

    print("\nTraining Images:", len(train_dataset))
    print("Validation Images:", len(val_dataset))
    print("Testing Images:", len(test_dataset))