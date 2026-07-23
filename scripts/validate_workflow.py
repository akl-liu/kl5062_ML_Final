"""Validate CLIP pipeline paths and artifacts after project restructure."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

REQUIRED_PATHS = [
    PROJECT_DIR / "app.py",
    PROJECT_DIR / "requirements.txt",
    DATA_DIR / "processed" / "train_metadata.json",
    DATA_DIR / "processed" / "train_image_ids.json",
    DATA_DIR / "processed" / "train_image_embeddings.npy",
    DATA_DIR / "processed" / "validation_metadata.json",
    DATA_DIR / "processed" / "validation_image_ids.json",
    DATA_DIR / "processed" / "validation_image_embeddings.npy",
]

EMBEDDING_SPLITS = {
    "train": [
        PROCESSED_DIR / "train_image_embeddings.npy",
        PROCESSED_DIR / "train_image_embeddings.pt",
    ],
    "validation": [
        PROCESSED_DIR / "validation_image_embeddings.npy",
        PROCESSED_DIR / "validation" / "validation_image_embeddings.npy",
        PROCESSED_DIR / "validation_image_embeddings.pt",
    ],
}


def check_exists(label: str, path: Path) -> bool:
    ok = path.exists()
    status = "OK" if ok else "MISSING"
    print(f"[{status}] {label}: {path.relative_to(PROJECT_DIR)}")
    return ok


def resolve_image_path(raw_path: str) -> Path | None:
    path = Path(raw_path)
    candidates = [
        path,
        PROJECT_DIR / path,
        PROJECT_DIR / str(raw_path).replace("\\", "/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def validate_metadata(split: str, metadata_path: Path, sample_size: int = 50) -> tuple[bool, str]:
    if not metadata_path.exists():
        return False, f"metadata missing: {metadata_path.name}"

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if not metadata:
        return False, f"{split} metadata is empty"

    missing = 0
    checked = min(sample_size, len(metadata))
    for item in metadata[:checked]:
        raw_path = item.get("image_path")
        if not raw_path:
            missing += 1
            continue
        if resolve_image_path(raw_path) is None:
            missing += 1

    if missing:
        return False, f"{split}: {missing}/{checked} sampled image_path entries not found on disk"

    return True, f"{split}: {len(metadata)} records, {checked} sampled paths resolve correctly"


def validate_ids_vs_metadata(split: str) -> tuple[bool, str]:
    ids_path = PROCESSED_DIR / f"{split}_image_ids.json"
    meta_path = PROCESSED_DIR / f"{split}_metadata.json"

    if not ids_path.exists() or not meta_path.exists():
        return False, f"{split}: missing ids or metadata file"

    with open(ids_path, "r", encoding="utf-8") as handle:
        image_ids = json.load(handle)
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    meta_ids = {str(item["image_id"]) for item in metadata}
    id_list = [str(image_id) for image_id in image_ids]

    if len(id_list) != len(set(id_list)):
        return False, f"{split}: duplicate ids in image_ids.json"

    missing_in_meta = [image_id for image_id in id_list[:20] if image_id not in meta_ids]
    if missing_in_meta:
        return False, f"{split}: ids not found in metadata (sample): {missing_in_meta[:3]}"

    return True, f"{split}: {len(id_list)} ids aligned with metadata ({len(meta_ids)} unique meta ids)"


def validate_embeddings(split: str) -> tuple[bool, str]:
    ids_path = PROCESSED_DIR / f"{split}_image_ids.json"
    if not ids_path.exists():
        return False, f"{split}: missing image ids"

    with open(ids_path, "r", encoding="utf-8") as handle:
        image_ids = json.load(handle)

    embedding_path = next((path for path in EMBEDDING_SPLITS[split] if path.exists()), None)
    if embedding_path is None:
        return False, f"{split}: no embedding file found"

    if embedding_path.suffix == ".npy":
        import numpy as np

        embeddings = np.load(embedding_path)
        count = embeddings.shape[0]
    else:
        import torch

        embeddings = torch.load(embedding_path, map_location="cpu")
        count = embeddings.shape[0]

    if count != len(image_ids):
        return False, f"{split}: embeddings ({count}) != ids ({len(image_ids)})"

    return True, f"{split}: embeddings OK at {embedding_path.relative_to(PROJECT_DIR)} ({count} vectors)"


def validate_app_paths() -> tuple[bool, str]:
    app_paths = {
        "embeddings": PROCESSED_DIR / "validation_image_embeddings.npy",
        "image_ids": PROCESSED_DIR / "validation_image_ids.json",
        "metadata": PROCESSED_DIR / "validation_metadata.json",
    }

    missing = [name for name, path in app_paths.items() if not path.exists()]
    if missing:
        alt_embeddings = PROCESSED_DIR / "validation" / "validation_image_embeddings.npy"
        if "embeddings" in missing and alt_embeddings.exists():
            missing.remove("embeddings")
        alt_meta = PROCESSED_DIR / "validation" / "validation_image_metadata.json"
        if "metadata" in missing and alt_meta.exists():
            missing.remove("metadata")

    if missing:
        return False, f"app.py expected files missing: {', '.join(missing)}"

    return True, "app.py path constants resolve to existing validation artifacts"


def main() -> int:
    print(f"Project root: {PROJECT_DIR}\n")

    failures = 0

    print("== Required files ==")
    for path in REQUIRED_PATHS:
        if not check_exists(path.name, path):
            failures += 1

    print("\n== Metadata path resolution ==")
    checks = [
        ("train", PROCESSED_DIR / "train_metadata.json"),
        ("validation", PROCESSED_DIR / "validation_metadata.json"),
    ]
    for split, metadata_path in checks:
        ok, message = validate_metadata(split, metadata_path)
        print(f"[{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            failures += 1

    print("\n== IDs vs metadata ==")
    for split in ("train", "validation"):
        ok, message = validate_ids_vs_metadata(split)
        print(f"[{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            failures += 1

    print("\n== Embeddings ==")
    for split in ("train", "validation"):
        ok, message = validate_embeddings(split)
        print(f"[{'OK' if ok else 'FAIL'}] {message}")
        if not ok:
            failures += 1

    print("\n== App entrypoint ==")
    ok, message = validate_app_paths()
    print(f"[{'OK' if ok else 'FAIL'}] {message}")
    if not ok:
        failures += 1

    print("\n== Script path robustness ==")
    scripts_data = PROJECT_DIR / "scripts" / "data"
    if scripts_data.exists():
        print("[FAIL] scripts/data still exists and can shadow project data/")
        failures += 1
    else:
        print("[OK] No scripts/data duplicate")

    generate_src = (PROJECT_DIR / "scripts" / "generate_img_embeddings.py").read_text(encoding="utf-8")
    if "MODEL_NAME，" in generate_src or "\uff0c" in generate_src:
        print("[FAIL] generate_img_embeddings.py still contains a fullwidth/Chinese comma")
        failures += 1
    else:
        print("[OK] generate_img_embeddings.py syntax looks clean")

    print()
    if failures:
        print(f"Validation finished with {failures} failure(s).")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
