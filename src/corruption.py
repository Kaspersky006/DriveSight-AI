from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "raw" / "10k"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "generated"

SPLITS = ("train", "val", "test")

CLASS_NAMES = (
    "clean",
    "fog",
    "rain",
    "motion_blur",
    "defocus_blur",
    "overexposure",
    "dirty_lens",
    "partial_occlusion",
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RANDOM_SEED = 42
JPEG_QUALITY = 92


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def open_rgb_image(image_path: Path) -> Image.Image:
    """Open an image and convert it to RGB format."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def save_image(image: Image.Image, output_path: Path) -> None:
    """Save an RGB image as a high-quality JPEG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=JPEG_QUALITY)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR array."""
    rgb_array = np.asarray(image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR array to a PIL RGB image."""
    rgb_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


# ---------------------------------------------------------------------------
# Camera-condition transformations
# ---------------------------------------------------------------------------

def create_clean(image: Image.Image, rng: random.Random) -> Image.Image:
    """Return an unchanged copy of the original image."""
    del rng
    return image.copy()


def create_fog(image: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate reduced visibility caused by fog."""
    image_array = np.asarray(image).astype(np.float32)

    height, width = image_array.shape[:2]

    fog_strength = rng.uniform(0.32, 0.55)
    brightness = rng.uniform(205, 245)

    fog_layer = np.full(
        shape=(height, width, 3),
        fill_value=brightness,
        dtype=np.float32,
    )

    # Fog becomes slightly stronger toward the horizon/top of the image.
    vertical_gradient = np.linspace(
        fog_strength,
        fog_strength * 0.55,
        height,
        dtype=np.float32,
    ).reshape(height, 1, 1)

    result = (
        image_array * (1.0 - vertical_gradient)
        + fog_layer * vertical_gradient
    )

    result = np.clip(result, 0, 255).astype(np.uint8)
    foggy_image = Image.fromarray(result)

    return foggy_image.filter(ImageFilter.GaussianBlur(radius=1.2))


def create_rain(image: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate rain streaks and reduced image brightness."""
    rainy_image = ImageEnhance.Brightness(image).enhance(
        rng.uniform(0.65, 0.82)
    )

    rain_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(rain_layer)

    width, height = image.size
    number_of_drops = max(100, int((width * height) / 2800))

    for _ in range(number_of_drops):
        start_x = rng.randint(-20, width)
        start_y = rng.randint(-20, height)

        drop_length = rng.randint(12, 35)
        slant = rng.randint(3, 10)
        thickness = rng.choice((1, 1, 2))
        opacity = rng.randint(70, 145)

        draw.line(
            (
                start_x,
                start_y,
                start_x + slant,
                start_y + drop_length,
            ),
            fill=(205, 220, 235, opacity),
            width=thickness,
        )

    rain_layer = rain_layer.filter(ImageFilter.GaussianBlur(radius=0.6))

    combined = Image.alpha_composite(
        rainy_image.convert("RGBA"),
        rain_layer,
    )

    return combined.convert("RGB")


def create_motion_blur(
    image: Image.Image,
    rng: random.Random,
) -> Image.Image:
    """Apply directional motion blur using a convolution kernel."""
    bgr_image = pil_to_bgr(image)

    kernel_size = rng.choice((9, 11, 13, 15))
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    direction = rng.choice(("horizontal", "diagonal_down", "diagonal_up"))

    if direction == "horizontal":
        kernel[kernel_size // 2, :] = 1.0
    elif direction == "diagonal_down":
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)

    kernel /= kernel.sum()

    blurred = cv2.filter2D(
        bgr_image,
        ddepth=-1,
        kernel=kernel,
    )

    return bgr_to_pil(blurred)


def create_defocus_blur(
    image: Image.Image,
    rng: random.Random,
) -> Image.Image:
    """Simulate a camera lens that is out of focus."""
    radius = rng.uniform(2.0, 4.5)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def create_overexposure(
    image: Image.Image,
    rng: random.Random,
) -> Image.Image:
    """Simulate strong sunlight or excessive camera exposure."""
    result = ImageEnhance.Brightness(image).enhance(
        rng.uniform(1.55, 2.05)
    )
    result = ImageEnhance.Contrast(result).enhance(
        rng.uniform(0.75, 0.95)
    )

    # Add a soft glare area.
    glare_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(glare_layer)

    width, height = image.size
    center_x = rng.randint(int(width * 0.15), int(width * 0.85))
    center_y = rng.randint(0, int(height * 0.45))
    radius = rng.randint(
        max(30, int(min(width, height) * 0.08)),
        max(60, int(min(width, height) * 0.22)),
    )

    draw.ellipse(
        (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ),
        fill=(255, 250, 225, rng.randint(100, 175)),
    )

    glare_layer = glare_layer.filter(
        ImageFilter.GaussianBlur(radius=max(15, radius // 2))
    )

    return Image.alpha_composite(
        result.convert("RGBA"),
        glare_layer,
    ).convert("RGB")


def create_dirty_lens(
    image: Image.Image,
    rng: random.Random,
) -> Image.Image:
    """Simulate dirt, mud, dust, or water marks on the camera lens."""
    dirty_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(dirty_layer)

    width, height = image.size
    number_of_spots = rng.randint(8, 18)

    for _ in range(number_of_spots):
        center_x = rng.randint(0, width)
        center_y = rng.randint(0, height)

        radius_x = rng.randint(
            max(8, width // 80),
            max(20, width // 18),
        )
        radius_y = rng.randint(
            max(8, height // 80),
            max(20, height // 15),
        )

        dirt_type = rng.choice(("mud", "dust", "water"))

        if dirt_type == "mud":
            color = (
                rng.randint(65, 115),
                rng.randint(45, 85),
                rng.randint(25, 55),
                rng.randint(90, 180),
            )
        elif dirt_type == "dust":
            color = (
                rng.randint(145, 195),
                rng.randint(125, 170),
                rng.randint(90, 135),
                rng.randint(60, 135),
            )
        else:
            color = (
                190,
                210,
                220,
                rng.randint(35, 85),
            )

        draw.ellipse(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            fill=color,
        )

    dirty_layer = dirty_layer.filter(
        ImageFilter.GaussianBlur(radius=rng.uniform(3.0, 7.0))
    )

    return Image.alpha_composite(
        image.convert("RGBA"),
        dirty_layer,
    ).convert("RGB")


def create_partial_occlusion(
    image: Image.Image,
    rng: random.Random,
) -> Image.Image:
    """Cover part of the camera view with an opaque object."""
    result = image.copy()
    draw = ImageDraw.Draw(result, "RGBA")

    width, height = image.size

    occlusion_width = rng.randint(
        max(40, int(width * 0.12)),
        max(80, int(width * 0.32)),
    )
    occlusion_height = rng.randint(
        max(40, int(height * 0.15)),
        max(80, int(height * 0.42)),
    )

    position = rng.choice(
        ("top_left", "top_right", "bottom_left", "bottom_right")
    )

    if position == "top_left":
        left, top = 0, 0
    elif position == "top_right":
        left, top = width - occlusion_width, 0
    elif position == "bottom_left":
        left, top = 0, height - occlusion_height
    else:
        left = width - occlusion_width
        top = height - occlusion_height

    right = left + occlusion_width
    bottom = top + occlusion_height

    color = rng.choice(
        (
            (25, 25, 25, 235),
            (70, 50, 35, 235),
            (210, 215, 220, 235),
        )
    )

    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=max(10, min(occlusion_width, occlusion_height) // 7),
        fill=color,
    )

    return result


TRANSFORMATIONS: dict[
    str,
    Callable[[Image.Image, random.Random], Image.Image],
] = {
    "clean": create_clean,
    "fog": create_fog,
    "rain": create_rain,
    "motion_blur": create_motion_blur,
    "defocus_blur": create_defocus_blur,
    "overexposure": create_overexposure,
    "dirty_lens": create_dirty_lens,
    "partial_occlusion": create_partial_occlusion,
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def find_images(folder: Path) -> list[Path]:
    """Return all supported image files from a folder."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def prepare_output_directories(
    output_root: Path,
    overwrite: bool,
) -> None:
    """Create the split and class directories."""
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)

    for split_name in SPLITS:
        for class_name in CLASS_NAMES:
            (output_root / split_name / class_name).mkdir(
                parents=True,
                exist_ok=True,
            )


def distribute_images_across_classes(
    image_paths: list[Path],
    seed: int,
) -> dict[str, list[Path]]:
    """Randomly and evenly allocate source images among all classes."""
    shuffled_images = image_paths.copy()

    rng = random.Random(seed)
    rng.shuffle(shuffled_images)

    allocation = {
        class_name: []
        for class_name in CLASS_NAMES
    }

    for index, image_path in enumerate(shuffled_images):
        class_name = CLASS_NAMES[index % len(CLASS_NAMES)]
        allocation[class_name].append(image_path)

    return allocation


def generate_split(
    split_name: str,
    input_root: Path,
    output_root: Path,
    seed: int,
    limit: int | None,
) -> None:
    """Generate one complete dataset split."""
    input_folder = input_root / split_name

    if not input_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder does not exist: {input_folder}"
        )

    image_paths = find_images(input_folder)

    if not image_paths:
        raise RuntimeError(
            f"No supported images were found in: {input_folder}"
        )

    if limit is not None:
        image_paths = image_paths[:limit]

    allocation = distribute_images_across_classes(
        image_paths=image_paths,
        seed=seed,
    )

    print(f"\nGenerating {split_name} split")
    print(f"Source images: {len(image_paths)}")

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_images = allocation[class_name]
        transformation = TRANSFORMATIONS[class_name]

        transformation_rng = random.Random(
            seed + class_index + len(split_name)
        )

        progress_bar = tqdm(
            class_images,
            desc=f"{split_name}/{class_name}",
            unit="image",
        )

        for image_path in progress_bar:
            try:
                original_image = open_rgb_image(image_path)

                transformed_image = transformation(
                    original_image,
                    transformation_rng,
                )

                output_filename = f"{image_path.stem}.jpg"
                output_path = (
                    output_root
                    / split_name
                    / class_name
                    / output_filename
                )

                save_image(transformed_image, output_path)

            except (OSError, ValueError, cv2.error) as error:
                print(
                    f"\nSkipped unreadable image "
                    f"{image_path.name}: {error}"
                )


def print_dataset_summary(output_root: Path) -> None:
    """Print the generated image count for every split and class."""
    print("\n" + "=" * 64)
    print("GENERATED DATASET SUMMARY")
    print("=" * 64)

    grand_total = 0

    for split_name in SPLITS:
        split_total = 0

        print(f"\n{split_name.upper()}")

        for class_name in CLASS_NAMES:
            class_folder = output_root / split_name / class_name
            count = len(find_images(class_folder))

            split_total += count
            print(f"  {class_name:<20} {count:>6}")

        grand_total += split_total
        print(f"  {'Total':<20} {split_total:>6}")

    print(f"\nGrand total: {grand_total}")
    print(f"Dataset location: {output_root}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced camera-health classification dataset "
            "from clean BDD10K driving images."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Folder containing train, val and test image directories.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination folder for the generated dataset.",
    )

    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help=(
            "Optional maximum number of source images used from each split. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the previous generated dataset before starting.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used for reproducible generation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if args.limit_per_split is not None and args.limit_per_split <= 0:
        raise ValueError("--limit-per-split must be greater than zero.")

    print("DriveSight AI dataset generator")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")

    prepare_output_directories(
        output_root=output_root,
        overwrite=args.overwrite,
    )

    for split_index, split_name in enumerate(SPLITS):
        generate_split(
            split_name=split_name,
            input_root=input_root,
            output_root=output_root,
            seed=args.seed + (split_index * 1000),
            limit=args.limit_per_split,
        )

    print_dataset_summary(output_root)


if __name__ == "__main__":
    main()

    