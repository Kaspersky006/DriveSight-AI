from pathlib import Path

from PIL import Image

# Path to the dataset
DATASET_PATH = Path("data/raw/10k")


def inspect_folder(folder_name: str):
    folder = DATASET_PATH / folder_name

    images = list(folder.glob("*.jpg"))

    print(f"\n{folder_name.upper()} SET")
    print("-" * 30)
    print(f"Number of images: {len(images)}")

    if len(images) == 0:
        print("No images found.")
        return

    # Open the first image
    image = Image.open(images[0])

    print(f"First image: {images[0].name}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")


def main():
    inspect_folder("train")
    inspect_folder("val")
    inspect_folder("test")


if __name__ == "__main__":
    main()