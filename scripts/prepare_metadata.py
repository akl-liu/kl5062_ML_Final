import argparse
from datasets import load_dataset
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import json

DATASET_NAME = "RIW/small-coco"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download images and build caption metadata for a dataset split."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to process (default: train)",
    )
    return parser.parse_args()


def clean_caption(caption):
    """
    Clean the caption by removing unwanted characters and formatting.
    """
    if not caption:
        return ""
    # Remove leading/trailing whitespace
    caption = caption.strip()
    # Replace multiple spaces with a single space
    caption = ' '.join(caption.split())
    return caption




def get_image_id(example,row_index):
    """
    Extract the image ID from the image path.
    """
    sha256 = example.get("sha256")
    if sha256:
        return sha256
    else:
        return f"image_{row_index:05d}"


def save_image(example, image_path):
    """
    Save the image to the specified path.
    """
    image = example.get("image")
    if image:
        image.save(image_path)
        return True
    return False



def main():
    args = parse_args()
    split = args.split

    images_dir = DATA_DIR / "images" / split
    output_json = PROCESSED_DIR / f"{split}_metadata.json"
    output_csv = PROCESSED_DIR / f"{split}_metadata.csv"

    print(f"Running metadata preparation for split: {split}")
    print(f"Loading dataset: {DATASET_NAME}, split: {split}")

    dataset = load_dataset(DATASET_NAME, split=split)

    images_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("\n Dataset loaded successfully!")
    print(f"\n Saving images to: {images_dir}")

    records = OrderedDict()
    saved_images = 0
    skipped_images = 0

    for row_index, example in enumerate(dataset):
        image_id = get_image_id(example,row_index)
        caption = clean_caption(example.get("caption"))

        image_path = images_dir/f"{image_id}.jpg"

        if image_id not in records:
            success = save_image(example, image_path)
            if not success:
                print(f"Warning: Failed to save image for example {row_index}.")
                skipped_images += 1
                continue
            records[image_id] = {
                "image_id": image_id,
                "split": split,
                "image_path": str(image_path),
                "caption": [],
                "url": example.get("url"),
                "width": example.get("width"),
                "height": example.get("height"),
                "row_indices":[]
            }

        records[image_id]["row_indices"].append(row_index)
        if caption and caption not in records[image_id]["caption"]:
            records[image_id]["caption"].append(caption)

        metadata = list(records.values())
        for item in metadata:
            item["num_caption"] = len(item["caption"])


    # Save metadata to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # Save metadata to CSV
    csv_records = []
    for item in metadata:
        csv_records.append({
            "image_id": item["image_id"],
            "split": item["split"],
            "image_path": item["image_path"],
            "caption": " | ".join(item["caption"]),
            "url": item["url"],
            "width": item["width"],
            "height": item["height"],
            "num_caption": item["num_caption"],
            "row_indices": ", ".join(map(str, item["row_indices"]))
        })

    df = pd.DataFrame(csv_records)
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print("\n Metadata saved successfully!")
    print(f"Saved metadata to {output_json}")
    print(f"Saved metadata to {output_csv}")
    print(f"\n Total images saved: {len(records)}")
    print(f" Total images skipped: {skipped_images}")
    print(f" Total captions saved: {sum(item['num_caption'] for item in metadata)}")

if __name__ == "__main__":
    main()

