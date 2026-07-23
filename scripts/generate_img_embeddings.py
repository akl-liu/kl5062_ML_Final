import argparse
from pathlib import Path
import json

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image


MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CLIP image embeddings for a dataset split, using the metadata "
        "produced by 02_prepare_metadata.py as the source of image ids/paths."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to process (default: train)",
    )
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_metadata(metadata_json):
    if not metadata_json.exists():
        raise SystemExit(
            f"Metadata file not found: {metadata_json}. "
            f"Run scripts/02_prepare_metadata.py --split <split> first."
        )

    with open(metadata_json, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_ids = [item["image_id"] for item in metadata]
    image_paths = [Path(item["image_path"]) for item in metadata]
    return image_ids, image_paths


def load_images(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(image)
    return images


def encode_image_batch(images, model, processor, device):
    images = load_images(images)

    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        image_features = model.visual_projection(vision_outputs.pooler_output)

    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    return image_features.cpu().numpy()


def generate_image_embeddings(image_paths, model, processor, device, batch_size=32):
    all_embeddings = []

    for start_index in range(0, len(image_paths), batch_size):
        end_index = start_index + batch_size
        batch_paths = image_paths[start_index:end_index]

        batch_embedding = encode_image_batch(batch_paths, model, processor, device)

        all_embeddings.append(batch_embedding)

        print(f"Processed batch {start_index // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings


def save_image_ids(image_ids, output_ids_file):
    with open(output_ids_file, "w") as f:
        json.dump(image_ids, f, indent=4)


def main():
    args = parse_args()
    split = args.split

    metadata_json = PROCESSED_DIR / f"{split}_metadata.json"
    output_embeddings_file = PROCESSED_DIR / f"{split}_image_embeddings.npy"
    output_ids_file = PROCESSED_DIR / f"{split}_image_ids.json"

    print(f"Running embedding generation for split: {split}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    image_ids, image_paths = load_metadata(metadata_json)
    print(f"Loaded metadata from {metadata_json}")
    if len(image_paths) == 0:
        print(f"No images found in {metadata_json}. Please check the metadata file.")
        return 0

    device = get_device()
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(
        MODEL_NAME,
        use_safetensors=True
        ).to(device)
    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME, 
        use_fast=True
        )

    image_embeddings = generate_image_embeddings(image_paths, model, processor, device, batch_size=BATCH_SIZE)
    np.save(output_embeddings_file, image_embeddings)
    print(f"Saved embeddings to {output_embeddings_file}")

    save_image_ids(image_ids, output_ids_file)
    print(f"Saved image ids to {output_ids_file}")

    print("Image embeddings generation completed successfully!")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Embeddings shape: {image_embeddings.shape}")


if __name__ == "__main__":
    main()
