from pathlib import Path
import random
import shutil

# ----------------------------
# Configuration
# ----------------------------
random.seed(42)

SOURCE = Path("data/generated")
DESTINATION = Path("data/generated_subset")

# Images per split
LIMITS = {
    "train": 1400,
    "val": 300,
    "test": 300,
}

CLASSES = [
    "clean",
    "fog",
    "rain",
    "motion_blur",
    "defocus_blur",
    "overexposure",
    "dirty_lens",
    "partial_occlusion",
]


# ----------------------------
# Create subset
# ----------------------------
for split in LIMITS:

    total_required = LIMITS[split]

    per_class = total_required // len(CLASSES)

    print(f"\nCreating {split} subset...")

    for cls in CLASSES:

        source_folder = SOURCE / split / cls

        destination_folder = DESTINATION / split / cls

        destination_folder.mkdir(parents=True, exist_ok=True)

        images = list(source_folder.glob("*.jpg"))

        random.shuffle(images)

        selected = images[:per_class]

        for image in selected:

            shutil.copy2(
                image,
                destination_folder / image.name
            )

        print(
            f"{cls:<20} {len(selected)} images"
        )

print("\nSubset generation complete.")