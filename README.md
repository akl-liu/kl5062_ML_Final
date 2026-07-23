# CLIP Image Retrieval

A text-to-image retrieval demo built with **OpenAI CLIP**. Given a natural-language query, the system finds the most similar images from a local gallery using cosine similarity in a shared embedding space.

This project is designed as an end-to-end machine learning workflow: prepare a captioned image dataset, encode images with CLIP, then retrieve matches through a command-line tool or a Streamlit web app.

---

## Topic Overview

[CLIP](https://openai.com/research/clip) (Contrastive Language–Image Pre-training) learns a joint representation of images and text. That means:

1. A **text query** is encoded into a text embedding.
2. Each **gallery image** is encoded into an image embedding (precomputed offline).
3. **Cosine similarity** ranks images by how close they are to the query embedding.
4. The app returns the **top-K** matches with captions and scores.

```text
Text query ──► CLIP text encoder ──► Text embedding ──┐
                                                      ├──► Cosine similarity ──► Top-K images
small-COCO ──► CLIP image encoder ──► Image embeddings ┘
```

---

## Data Source

| Item | Detail |
|------|--------|
| Dataset | [`RIW/small-coco`](https://huggingface.co/datasets/RIW/small-coco) on Hugging Face |
| Origin | A compact subset of [MS COCO](https://cocodataset.org/) image–caption pairs |
| Splits used | `train`, `validation` |
| Scale (this project) | ~1,999 images per split (after local preparation) |
| Content | Local JPG images + multi-caption metadata |

Images and metadata are stored under `data/` after running the preparation scripts. Raw JPGs are ignored by Git (see `.gitignore`); regenerate them locally with the pipeline below if needed.

---

## Model

| Item | Detail |
|------|--------|
| Model | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) |
| Library | Hugging Face `transformers` |
| Embedding size | 512-dimensional vectors |
| Device | CPU by default (CUDA used automatically in the CLI search script when available) |

---

## Packages

Core dependencies are listed in `requirements.txt`:

| Package | Role |
|---------|------|
| `datasets` | Load `RIW/small-coco` from Hugging Face |
| `pandas` | Tabular metadata (CSV) handling |
| `pillow` | Image loading and display |
| `numpy` | Embedding arrays (`.npy`) |
| `torch` | Tensor ops and model inference |
| `transformers` | CLIP model + processor |
| `scikit-learn` | ML utilities for similarity / retrieval workflows |
| `streamlit` | Interactive web demo (`app.py`) |
| `matplotlib` | Optional visualization / debugging |
| `graphviz` | Pipeline flowchart in the Streamlit sidebar |

**Note:** The Python `graphviz` package also needs the Graphviz system binary (`dot`) available (e.g. via conda: `conda install -c conda-forge graphviz`).

---

## Project Structure

```text
project_CLIP/
├── app.py                 # Streamlit text-to-image retrieval UI
├── requirements.txt
├── .gitignore
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── validation/
│   └── processed/
│       ├── *_metadata.json
│       ├── *_image_ids.json
│       └── *_image_embeddings.npy
└── scripts/
    ├── prepare_metadata.py        # Download images + build metadata
    ├── preprocess_images.py       # Sanity-check CLIP preprocessing
    ├── generate_img_embeddings.py # Precompute image embeddings
    ├── text-img_query.py          # CLI text-to-image search
    ├── inspect_dataset.py         # Dataset inspection helpers
    └── validate_workflow.py       # Path / artifact consistency checks
```

---

## Setup

```powershell
conda activate clip   # or your preferred environment
cd path\to\project_CLIP
pip install -r requirements.txt
```

If Graphviz charts do not render on Windows, install the system binary into the same env:

```powershell
conda install -n clip -c conda-forge graphviz
```

---

## Pipeline (Recommended Order)

Run commands from the `project_CLIP` directory.

### 1. Prepare metadata and images

```powershell
python scripts/prepare_metadata.py --split train
python scripts/prepare_metadata.py --split validation
```

### 2. (Optional) Check image preprocessing

```powershell
python scripts/preprocess_images.py --split train
```

### 3. Generate image embeddings

```powershell
python scripts/generate_img_embeddings.py --split train
python scripts/generate_img_embeddings.py --split validation
```

### 4. Search from the command line

```powershell
python scripts/text-img_query.py --query "a dog sitting on grass" --split train --top_k 5
python scripts/text-img_query.py --query "a person riding a bike" --split validation --top_k 5
```

### 5. Launch the web app

The Streamlit app uses the **validation** split by default:

```powershell
streamlit run app.py
```

Then open the local URL (typically http://localhost:8501).

### 6. Validate project consistency

```powershell
python scripts/validate_workflow.py
```

---

## Web App Features

- Combined search box: pick an example query or type your own
- Top-K slider and pipeline flowchart in the sidebar
- Image-first results with similarity scores and captions
- Dual-branch flowchart: text path vs small-COCO image path merging at cosine similarity

---

## Notes

- Scripts resolve the `data/` folder from the project root, so they work even if launched from `scripts/`.
- Image files (`*.jpg`) and CSV exports (`*.csv`) are gitignored to keep the repository light.
- Precomputed embeddings (`.npy`) and JSON metadata are expected under `data/processed/` for the app and CLI search to run.
