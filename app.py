from pathlib import Path
import html
import json
import os
import sys

# Ensure the Graphviz `dot` binary from the conda env is findable on Windows.
_GRAPHVIZ_BIN = Path(sys.prefix) / "Library" / "bin"
if (_GRAPHVIZ_BIN / "dot.exe").exists():
    os.environ["PATH"] = str(_GRAPHVIZ_BIN) + os.pathsep + os.environ.get("PATH", "")

import graphviz
import numpy as np
import torch
import streamlit as st
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# =========================
# 1. Basic Configuration
# =========================

MODEL_NAME = "openai/clip-vit-base-patch32"
SPLIT = "validation"

PROJECT_DIR = Path(__file__).parent

DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images" / SPLIT
PROCESSED_DIR = DATA_DIR / "processed"

EMBEDDINGS_PATH = PROCESSED_DIR / f"{SPLIT}_image_embeddings.npy"
IMAGE_IDS_PATH = PROCESSED_DIR / f"{SPLIT}_image_ids.json"
METADATA_PATH = PROCESSED_DIR / f"{SPLIT}_metadata.json"


# =========================
# 2. Streamlit Page Setup
# =========================

st.set_page_config(
    page_title="CLIP Image Retrieval",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;500;600&display=swap');

    :root {
        --ink: #141a22;
        --muted: #667180;
        --paper: #eef1f4;
        --panel: #f7f8fa;
        --line: #d5dde5;
        --accent: #0f6e6a;
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 12% -10%, rgba(15, 110, 106, 0.08), transparent 55%),
            radial-gradient(900px 420px at 90% 0%, rgba(28, 36, 48, 0.06), transparent 50%),
            linear-gradient(180deg, #f4f6f8 0%, var(--paper) 100%);
        color: var(--ink);
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
        line-height: 1.55;
    }

    h1, h2, h3, .brand-title, .results-label {
        font-family: "Fraunces", Georgia, serif !important;
        letter-spacing: -0.025em;
        color: var(--ink) !important;
        font-weight: 700 !important;
        line-height: 1.15 !important;
    }

    .brand-title {
        font-size: 2.55rem;
        margin: 0.15rem 0 0.85rem 0;
    }

    .brand-sub {
        color: var(--muted);
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.65;
        max-width: 40rem;
        margin: 0 0 2rem 0;
    }

    .search-shell {
        max-width: 46rem;
        margin: 0 0 2rem 0;
    }

    /* Extra breathing room around the search widget */
    div[data-testid="stSelectbox"] {
        margin-bottom: 0.75rem;
    }

    .results-label {
        font-size: 1.55rem;
        margin: 1.75rem 0 1.15rem 0;
        padding-top: 0.35rem;
    }

    .result-meta {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 0.75rem;
        margin: 0.65rem 0 0.4rem 0;
        font-size: 0.92rem;
        color: var(--muted);
    }

    .result-rank {
        font-family: "Fraunces", Georgia, serif;
        font-weight: 700;
        font-size: 1rem;
        color: var(--ink);
    }

    .result-score {
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
        font-variant-numeric: tabular-nums;
        color: var(--accent);
        font-weight: 600;
    }

    .result-caption {
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
        color: var(--muted);
        font-size: 0.9rem;
        font-weight: 400;
        line-height: 1.55;
        margin: 0 0 1.1rem 0;
    }

    .empty-hint {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.85rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid #dde2e8;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-family: "Fraunces", Georgia, serif !important;
        font-weight: 700 !important;
        margin-top: 0.15rem !important;
        margin-bottom: 0.85rem !important;
    }

    div[data-testid="stSidebarContent"] {
        color: var(--ink);
    }

    .sidebar-kicker {
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0.35rem 0 0.55rem 0;
    }

    .sidebar-meta {
        font-family: "Source Sans 3", "Segoe UI", sans-serif;
        font-size: 0.92rem;
        font-weight: 400;
        color: var(--muted);
        line-height: 1.7;
        margin-bottom: 0.35rem;
    }

    .pipeline-steps {
        font-size: 0.9rem;
        color: var(--muted);
        line-height: 1.7;
    }

    /* Soften default alert/info chrome */
    div[data-testid="stAlert"] {
        border-radius: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# 3. Load Data
# =========================

@st.cache_data
def load_json(path):
    """
    Load a JSON file.

    We use this function for both metadata and image ids.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_image_embeddings(path):
    """
    Load precomputed CLIP image embeddings from a .npy file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    embeddings = np.load(path)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)

    return embeddings


@st.cache_resource
def load_clip_model(model_name):
    """
    Load CLIP model and CLIP processor.

    This is cached because loading CLIP is expensive.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, processor, device


# =========================
# 4. Helper Functions
# =========================

def normalize_embeddings(embeddings):
    """
    Normalize embeddings to unit vectors.

    After normalization, dot product is equivalent to cosine similarity.
    """
    return embeddings / embeddings.norm(dim=-1, keepdim=True)


def build_metadata_lookup(metadata):
    """
    Convert metadata into a dictionary.

    Goal:
    image_id -> metadata item

    This makes it easier to find the caption and image path for a retrieved image.
    """
    lookup = {}

    if isinstance(metadata, list):
        for item in metadata:
            image_id = item.get("image_id") or item.get("id") or item.get("file_name")
            if image_id is not None:
                lookup[str(image_id)] = item

    elif isinstance(metadata, dict):
        for key, value in metadata.items():
            if isinstance(value, dict):
                image_id = value.get("image_id") or value.get("id") or key
                lookup[str(image_id)] = value

    return lookup


def encode_text_query(query, model, processor, device):
    """
    Convert the user's text query into a CLIP text embedding.
    """
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)

    if not torch.is_tensor(text_embedding):
        # transformers>=5.0 returns a BaseModelOutputWithPooling whose
        # pooler_output holds the projected text embedding.
        text_embedding = text_embedding.pooler_output

    text_embedding = normalize_embeddings(text_embedding)

    return text_embedding.cpu()

def retrieve_top_k(query, model, processor, device, image_embeddings, top_k):
    """
    Retrieve Top-K images according to cosine similarity.
    """
    text_embedding = encode_text_query(
        query=query,
        model=model,
        processor=processor,
        device=device
    )

    image_embeddings = normalize_embeddings(image_embeddings)

    similarities = image_embeddings @ text_embedding.T
    similarities = similarities.squeeze(1)

    top_k = min(top_k, len(similarities))

    scores, indices = torch.topk(similarities, k=top_k)

    results = []

    for rank, (score, index) in enumerate(zip(scores, indices), start=1):
        results.append({
            "rank": rank,
            "index": index.item(),
            "score": score.item()
        })

    return results


def get_caption(metadata_item):
    """
    Get caption from metadata.
    """
    if metadata_item is None:
        return "No caption available."

    possible_keys = [
        "caption",
        "captions",
        "text",
        "description"
    ]

    for key in possible_keys:
        if key in metadata_item and metadata_item[key]:
            value = metadata_item[key]

            if isinstance(value, list):
                return " / ".join(str(v) for v in value[:2])

            return str(value)

    return "No caption available."


def find_image_path(image_id, metadata_item):
    """
    Find local image path.

    We first check metadata.
    If metadata does not contain a valid path, we search inside data/images/validation.
    """
    if metadata_item is not None:
        possible_path_keys = [
            "image_path",
            "local_image_path",
            "path",
            "file_path"
        ]

        for key in possible_path_keys:
            if key in metadata_item and metadata_item[key]:
                path = Path(metadata_item[key])

                if path.exists():
                    return path

                project_relative_path = PROJECT_DIR / path
                if project_relative_path.exists():
                    return project_relative_path

    image_id = str(image_id)

    candidates = [
        IMAGE_DIR / image_id,
        IMAGE_DIR / f"{image_id}.jpg",
        IMAGE_DIR / f"{image_id}.jpeg",
        IMAGE_DIR / f"{image_id}.png",
        IMAGE_DIR / f"{image_id}.webp"
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def build_pipeline_flowchart():
    """
    Build a dual-branch Graphviz flowchart for the CLIP retrieval pipeline.

    Text branch: query → text encoder → text embedding
    Image branch: small-COCO → image encoder → image embeddings
    Both merge at cosine similarity → top-K images.

    Embedding nodes use diamond (rhombus) shapes.
    """
    flow = graphviz.Digraph(
        name="clip_pipeline",
        graph_attr={
            "rankdir": "TB",
            "bgcolor": "transparent",
            "pad": "0.06",
            "nodesep": "0.22",
            "ranksep": "0.22",
            "fontname": "Helvetica",
            "splines": "true",
            "size": "3.2,3.6",
            "ratio": "compress",
        },
        node_attr={
            "style": "filled",
            "fontname": "Helvetica",
            "fontsize": "8.5",
            "fontcolor": "#141a22",
            "margin": "0.06,0.04",
        },
        edge_attr={
            "arrowsize": "0.5",
            "penwidth": "1.0",
            "fontsize": "7.5",
            "fontcolor": "#5b6675",
        },
    )

    # ---- Text branch (teal family) ----
    flow.node(
        "text_query",
        "Text query",
        shape="ellipse",
        fillcolor="#d9efe8",
        color="#0f6e6a",
    )
    flow.node(
        "text_encoder",
        "CLIP text\nencoder",
        shape="hexagon",
        fillcolor="#cfe8e2",
        color="#0f6e6a",
    )
    flow.node(
        "text_emb",
        "Text\nembedding",
        shape="diamond",
        fillcolor="#b7ddd4",
        color="#0b5a56",
    )

    # ---- Image branch (blue / warm family) ----
    flow.node(
        "image_data",
        "Image dataset\n(small-COCO)",
        shape="folder",
        fillcolor="#f3e8d8",
        color="#a56b2d",
    )
    flow.node(
        "image_encoder",
        "CLIP image\nencoder",
        shape="hexagon",
        fillcolor="#e4ecf7",
        color="#3d5a80",
    )
    flow.node(
        "image_emb",
        "Image\nembeddings",
        shape="diamond",
        fillcolor="#cfdcf0",
        color="#3d5a80",
    )

    # ---- Shared scoring + output ----
    flow.node(
        "similarity",
        "Cosine\nsimilarity",
        shape="box",
        style="rounded,filled",
        fillcolor="#eee6f5",
        color="#6b5b8c",
    )
    flow.node(
        "topk",
        "Top-K images",
        shape="oval",
        style="filled",
        fillcolor="#e7f2d9",
        color="#4f7a28",
        peripheries="2",
    )

    # Keep branch stages aligned side-by-side.
    with flow.subgraph() as inputs:
        inputs.attr(rank="same")
        inputs.node("text_query")
        inputs.node("image_data")

    with flow.subgraph() as encoders:
        encoders.attr(rank="same")
        encoders.node("text_encoder")
        encoders.node("image_encoder")

    with flow.subgraph() as embeddings:
        embeddings.attr(rank="same")
        embeddings.node("text_emb")
        embeddings.node("image_emb")

    flow.edge("text_query", "text_encoder", color="#0f6e6a")
    flow.edge("text_encoder", "text_emb", color="#0f6e6a")
    flow.edge("image_data", "image_encoder", color="#a56b2d")
    flow.edge("image_encoder", "image_emb", color="#3d5a80")
    flow.edge("text_emb", "similarity", color="#0f6e6a")
    flow.edge("image_emb", "similarity", color="#3d5a80")
    flow.edge("similarity", "topk", color="#6b5b8c")

    return flow


# =========================
# 5. Main App
# =========================

st.markdown('<p class="brand-title">CLIP Image Retrieval</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="brand-sub">Search the validation set with natural language. '
    "CLIP matches your query to images by cosine similarity.</p>",
    unsafe_allow_html=True,
)

try:
    metadata = load_json(METADATA_PATH)
    image_ids = load_json(IMAGE_IDS_PATH)
    image_embeddings = load_image_embeddings(EMBEDDINGS_PATH)
    model, processor, device = load_clip_model(MODEL_NAME)

except Exception as error:
    st.error("Failed to load model or project files.")
    st.exception(error)
    st.stop()

metadata_lookup = build_metadata_lookup(metadata)

if len(image_ids) != len(image_embeddings):
    st.warning(
        "The number of image ids does not match the number of image embeddings. "
        "Please check your embedding generation step."
    )

with st.sidebar:
    st.markdown('<p class="sidebar-kicker">Studio</p>', unsafe_allow_html=True)
    st.header("Controls")

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5,
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-kicker">Status</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sidebar-meta">'
        f"{len(image_ids):,} images · {SPLIT} split<br>"
        f"Device: {device}<br>"
        f"Model: {MODEL_NAME}"
        f"</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<p class="sidebar-kicker">Pipeline</p>', unsafe_allow_html=True)
    st.graphviz_chart(build_pipeline_flowchart(), width="stretch")

example_queries = [
    "a dog playing outside",
    "a person riding a bike",
    "a cat sitting on a sofa",
    "food on a table",
    "a car on the street",
    "people sitting together",
]

query = st.selectbox(
    "Search query",
    options=example_queries,
    index=None,
    accept_new_options=True,
    placeholder="Choose an example or type your own query",
)

query = (query or "").strip()

if query:
    with st.spinner("Matching images…"):
        results = retrieve_top_k(
            query=query,
            model=model,
            processor=processor,
            device=device,
            image_embeddings=image_embeddings,
            top_k=top_k,
        )

    st.markdown('<p class="results-label">Results</p>', unsafe_allow_html=True)

    # Keep a stable 3-column grid so layout does not jump with top_k.
    columns = st.columns(3)

    for i, result in enumerate(results):
        rank = result["rank"]
        index = result["index"]
        score = result["score"]

        image_id = str(image_ids[index])
        metadata_item = metadata_lookup.get(image_id)
        caption = html.escape(get_caption(metadata_item))
        image_path = find_image_path(image_id, metadata_item)

        with columns[i % 3]:
            if image_path is not None:
                image = Image.open(image_path).convert("RGB")
                st.image(image, width="stretch")
            else:
                st.warning(f"Image not found for image_id: {image_id}")

            st.markdown(
                f'<div class="result-meta">'
                f'<span class="result-rank">#{rank}</span>'
                f'<span class="result-score">{score:.4f}</span>'
                f"</div>"
                f'<p class="result-caption">{caption}</p>',
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        '<p class="empty-hint">Choose an example or type a query to start retrieval.</p>',
        unsafe_allow_html=True,
    )
