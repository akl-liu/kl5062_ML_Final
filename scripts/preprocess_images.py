import argparse
from pathlib import Path

from PIL import Image
from transformers import CLIPProcessor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_NAME = "openai/clip-vit-base-patch32"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check CLIP image preprocessing for a dataset split."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to process (default: train)",
    )
    return parser.parse_args()


def load_image_paths(image_dir):
    image_extensions = ["*.jpg"]
    image_paths = []

    for extension in image_extensions:
        image_paths.extend(image_dir.glob(extension))

    image_paths.sort()  # Sort the image paths for consistent ordering
    return image_paths


def preprocess_one_image(image_path, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    pixel_values = inputs["pixel_values"]
    return image, pixel_values


def main():
    args = parse_args()
    split = args.split
    image_dir = DATA_DIR / "images" / split

    print(f"Running image preprocessing check for split: {split}")

    image_paths = load_image_paths(image_dir)
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}. Please check the directory.")
        return 0
    print(f"Found {len(image_paths)} images in {image_dir}.")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    first_image_path = image_paths[0]
    original_image, pixel_values = preprocess_one_image(first_image_path, processor)

    print(f"First image size: {original_image.size}")
    print(f"Pixel values shape: {pixel_values.shape}")
    print ("Size",original_image.size)


    print("\nProcessed tensor shape:", pixel_values.shape)


if __name__ == "__main__":
    main()
