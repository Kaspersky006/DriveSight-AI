from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import train_loader, val_loader
from model import CameraConditionCNN


# -----------------------------
# Configuration
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 8
LEARNING_RATE = 0.0005

MODEL_SAVE_PATH = Path("outputs/models/best_model.pth")
MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Model
# -----------------------------
model = CameraConditionCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_accuracy = 0.0


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)

        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total


    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total


    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_accuracy:.2f}%")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_accuracy:.2f}%")


    # -------------------------
    # Save Best Model
    # -------------------------
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_SAVE_PATH
        )

        print("Best model saved!")

print("\nTraining Finished.")