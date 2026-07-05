from pathlib import Path
import argparse
import json

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

#python scripts/05_text_to_image_search.py --query "a dog sitting on grass" --top_k 5

def load_json(path):
    """
    Load a JSON file from disk.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_embeddings(processed_dir, split):
    """
    Load image embeddings.

    This function supports both .npy and .pt formats because different scripts
    may save embeddings in different formats.
    """
    npy_path = processed_dir / f"{split}_image_embeddings.npy"
    pt_path = processed_dir / f"{split}_image_embeddings.pt"

    if npy_path.exists():
        embeddings = np.load(npy_path)
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        print(f"Loaded image embeddings from: {npy_path}")
        return embeddings

    if pt_path.exists():
        embeddings = torch.load(pt_path, map_location="cpu")
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        elif not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"Unsupported embedding type: {type(embeddings)}")
        embeddings = embeddings.float()
        print(f"Loaded image embeddings from: {pt_path}")
        return embeddings

    raise FileNotFoundError(
        f"Could not find embeddings file for split='{split}'. "
        f"Expected either {npy_path} or {pt_path}."
    )


def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit length.

    After normalization, cosine similarity can be computed by dot product.
    """
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def build_metadata_lookup(metadata):
    """
    Build a dictionary from image_id to metadata row.

    This makes it easy to find image path and caption after retrieval.
    """
    lookup = {}

    for item in metadata:
        image_id = str(item.get("image_id"))
        lookup[image_id] = item

    return lookup

"""
lookup["abc123"] = {
    "image_id": "abc123",
    "image_path": "data/images/train/abc123.jpg",
    "caption": "a dog sitting on grass"
}
"""
def get_text_embedding(query, model, processor, device):
    """
    Convert user text query into a CLIP text embedding.
    """
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    if not isinstance(text_features, torch.Tensor):
        text_features = text_features.pooler_output

    text_features = normalize_embeddings(text_features)

    return text_features.cpu()


def search_images(query, split, top_k, model_name):
    """
    Main text-to-image retrieval function.
    """
    data_dir = Path("data")
    processed_dir = data_dir / "processed"

    metadata_path = processed_dir / f"{split}_metadata.json"
    image_ids_path = processed_dir / f"{split}_image_ids.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not image_ids_path.exists():
        raise FileNotFoundError(f"Image ids file not found: {image_ids_path}")

    print(f"Running text-to-image search on split: {split}")
    print(f"Query: {query}")
    print(f"Top-K: {top_k}")

    metadata = load_json(metadata_path)
    image_ids = load_json(image_ids_path)

    image_embeddings = load_image_embeddings(processed_dir, split)

    if len(image_ids) != image_embeddings.shape[0]:
        raise ValueError(
            f"Number of image ids ({len(image_ids)}) does not match "
            f"number of image embeddings ({image_embeddings.shape[0]})."
        )

    metadata_lookup = build_metadata_lookup(metadata)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained(
        model_name,
        use_safetensors=True
        ).to(device)
    processor = CLIPProcessor.from_pretrained(
        model_name,
        use_fast=True
        )
    model.eval()

    image_embeddings = normalize_embeddings(image_embeddings)

    text_embedding = get_text_embedding(
        query=query,
        model=model,
        processor=processor,
        device=device
    )


    similarities = text_embedding @ image_embeddings.T
    similarities = similarities.squeeze(0)

    top_k = min(top_k, similarities.shape[0])
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    print("\nSearch Results")
    print("=" * 60)

    results = []

    for rank, (score, index) in enumerate(zip(top_scores, top_indices), start=1):
        index = index.item()
        score = score.item()

        image_id = str(image_ids[index])
        item = metadata_lookup.get(image_id, {})

        image_path = item.get("image_path", "N/A")
        caption = item.get("caption", "N/A")

        result = {
            "rank": rank,
            "score": score,
            "image_id": image_id,
            "image_path": image_path,
            "caption": caption
        }

        results.append(result)

        print(f"Top {rank}")
        print(f"Score: {score:.4f}")
        print(f"Image ID: {image_id}")
        print(f"Image Path: {image_path}")
        print(f"Caption: {caption}")
        print("-" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-image search using CLIP embeddings."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text query used to search images."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to search. Default: train."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return. Default: 5."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Pretrained CLIP model name."
    )

    args = parser.parse_args()

    search_images(
        query=args.query,
        split=args.split,
        top_k=args.top_k,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()