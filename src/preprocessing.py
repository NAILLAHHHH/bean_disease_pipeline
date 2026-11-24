"""
Data acquisition + data processing for the Bean Leaf Disease pipeline.

Dataset: iBeans (Makerere University AI Lab, Uganda) - photos of bean leaves
taken in the field with smartphones, labelled by the National Crops Resources
Research Institute (NaCRRI). Three classes:

    angular_leaf_spot | bean_rust | healthy

Responsibilities of this module
-------------------------------
1. Acquisition   - download the dataset (Hugging Face parquet mirror) and write
                   it to data/train/<class>/ and data/test/<class>/ as JPEGs.
2. Ingestion     - save user-uploaded images (single files or a .zip of class
                   folders) into data/uploads/<class>/ for later retraining.
3. Processing    - decode / resize / normalise images, build tf.data pipelines
                   with augmentation for training.
4. Feature stats - compute interpretable per-image features (greenness,
                   brown/lesion pixel ratio, brightness, contrast, ...) that the
                   dashboard and notebook visualise.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import database as db

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
STATS_PATH = DATA_DIR / "feature_stats.json"

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
IMG_SIZE = (160, 160)
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

HF_BASE = "https://huggingface.co/api/datasets/AI-Lab-Makerere/beans/parquet/default"


# --------------------------------------------------------------------------- #
# 1. Data acquisition
# --------------------------------------------------------------------------- #
def download_dataset(force: bool = False) -> dict:
    """Download iBeans and materialise it as class folders.

    The original train split goes to data/train. The original validation and
    test splits are merged into data/test (held out; never used for fitting).
    A validation split for early stopping is carved out of data/train at
    training time.
    """
    import pandas as pd
    import requests

    if not force and _count_images(TRAIN_DIR) > 0 and _count_images(TEST_DIR) > 0:
        return {"train": _count_images(TRAIN_DIR), "test": _count_images(TEST_DIR), "skipped": True}

    for d in (TRAIN_DIR, TEST_DIR):
        shutil.rmtree(d, ignore_errors=True)
        for c in CLASS_NAMES:
            (d / c).mkdir(parents=True, exist_ok=True)

    targets = {"train": TRAIN_DIR, "validation": TEST_DIR, "test": TEST_DIR}
    counts = {"train": 0, "test": 0}
    for split, out_dir in targets.items():
        url = f"{HF_BASE}/{split}/0.parquet"
        print(f"Downloading {split} split ...")
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content))
        # columns: image_file_path, image {bytes, path}, labels (int)
        for i, row in df.iterrows():
            label = CLASS_NAMES[int(row["labels"])]
            img_bytes = row["image"]["bytes"]
            (out_dir / label / f"{split}_{i:05d}.jpg").write_bytes(img_bytes)
            counts["train" if out_dir == TRAIN_DIR else "test"] += 1
    return counts


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in ALLOWED_EXT)


def list_images(directory: Path) -> list[tuple[Path, int]]:
    """Return [(path, class_index)] for every image under directory/<class>/."""
    items = []
    for idx, c in enumerate(CLASS_NAMES):
        cdir = directory / c
        if cdir.exists():
            items += [(p, idx) for p in sorted(cdir.iterdir()) if p.suffix.lower() in ALLOWED_EXT]
    return items


def dataset_counts() -> dict:
    out = {}
    for name, d in (("train", TRAIN_DIR), ("test", TEST_DIR), ("uploads", UPLOAD_DIR)):
        out[name] = {c: _count_images(d / c) for c in CLASS_NAMES}
    return out


# --------------------------------------------------------------------------- #
# 2. Ingestion of new (uploaded) data
# --------------------------------------------------------------------------- #
def validate_image_bytes(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def save_uploaded_images(files: Iterable[tuple[str, bytes]], label: str | None) -> dict:
    """Save uploaded files into the SQLite database and data/uploads/<class>/.

    `files` is an iterable of (filename, bytes). Each item may be an image
    (requires `label`) or a .zip containing <class_name>/<image> entries
    (label taken from the folder name inside the zip). The image bytes are
    stored in data/bean_leaf.db, table uploaded_images, for retraining.
    """
    saved, rejected, saved_ids = {c: 0 for c in CLASS_NAMES}, [], []

    def _store(name: str, data: bytes, cls: str):
        cls = normalise_label(cls)
        if cls is None:
            rejected.append(f"{name}: unknown label")
            return
        if Path(name).suffix.lower() not in ALLOWED_EXT or not validate_image_bytes(data):
            rejected.append(f"{name}: not a valid image")
            return
        out = UPLOAD_DIR / cls
        out.mkdir(parents=True, exist_ok=True)
        dest = out / f"{uuid.uuid4().hex[:12]}{Path(name).suffix.lower()}"
        dest.write_bytes(data)
        saved[cls] += 1
        width = height = None
        try:
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
        except (UnidentifiedImageError, OSError, ValueError):
            pass
        saved_ids.append(db.insert_upload(
            original_name=Path(name).name,
            label=cls,
            content=data,
            width=width,
            height=height,
            raw_path=str(dest.relative_to(ROOT)),
        ))

    for name, data in files:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for entry in zf.infolist():
                    if entry.is_dir() or "__MACOSX" in entry.filename:
                        continue
                    parts = Path(entry.filename).parts
                    cls = parts[-2] if len(parts) >= 2 else label
                    _store(entry.filename, zf.read(entry), cls or "")
        else:
            _store(name, data, label or "")
    return {"saved": saved, "total_saved": sum(saved.values()), "rejected": rejected,
            "database_ids": saved_ids}


def normalise_label(label: str) -> str | None:
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"als": "angular_leaf_spot", "angular": "angular_leaf_spot", "rust": "bean_rust"}
    key = aliases.get(key, key)
    return key if key in CLASS_NAMES else None


def pending_upload_count() -> int:
    return _count_images(UPLOAD_DIR)


def preprocess_pending_uploads() -> dict:
    """Decode, convert to RGB and resize every uploaded image that is still raw.

    The processed JPEG is written back into the uploaded_images row (and to
    data/uploads_preprocessed/ so retraining can read a file). Originals stay
    in the database until retraining consumes the row.
    """
    rows = db.unpreprocessed()
    by_class = {c: 0 for c in CLASS_NAMES}
    ids: list[int] = []
    for row in rows:
        arr = load_image(bytes(row["content"]))
        image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        processed = buf.getvalue()
        out_dir = DATA_DIR / "uploads_preprocessed" / row["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{row['id']}.jpg"
        path.write_bytes(processed)
        db.mark_preprocessed(int(row["id"]), processed, IMG_SIZE[0], IMG_SIZE[1],
                             str(path.relative_to(ROOT)))
        by_class[row["label"]] += 1
        ids.append(int(row["id"]))
    return {"preprocessed": len(ids), "by_class": by_class, "ids": ids}


def merge_uploads_into_train() -> dict:
    """Move preprocessed uploads into data/train and mark those database rows used.

    Also picks up any image sitting in data/uploads/ with no database row.
    """
    preprocess_pending_uploads()
    ready = db.ready_for_training()
    moved_ids: list[int] = []
    for row in ready:
        src = ROOT / row["processed_path"] if row.get("processed_path") else None
        if (src is None or not src.exists()) and row.get("processed"):
            fallback = DATA_DIR / "uploads_preprocessed" / row["label"] / f"{row['id']}.jpg"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_bytes(row["processed"])
            src = fallback
        if src is None or not src.exists():
            continue
        dst_dir = TRAIN_DIR / row["label"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst_dir / f"upload_{row['id']}.jpg")
        moved_ids.append(int(row["id"]))
        raw = row.get("raw_path")
        if raw:
            raw_path = ROOT / raw
            if raw_path.is_file():
                raw_path.unlink()
    db.mark_used(moved_ids)

    legacy = 0
    claimed = {ROOT / r["raw_path"] for r in ready if r.get("raw_path")}
    for c in CLASS_NAMES:
        src_dir = UPLOAD_DIR / c
        if not src_dir.exists():
            continue
        dst_dir = TRAIN_DIR / c
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in list(src_dir.iterdir()):
            if p.suffix.lower() not in ALLOWED_EXT or p in claimed:
                continue
            shutil.move(str(p), dst_dir / f"upload_{p.name}")
            legacy += 1
    return {"moved": len(moved_ids) + legacy, "ids": moved_ids}


def prepare_uploads_for_training(progress=None) -> dict:
    """Preprocess database uploads, then merge them into the training set.

    This is the step retraining runs before it warm-starts the production model.
    """
    prep = preprocess_pending_uploads()
    if progress:
        w, h = IMG_SIZE
        progress(
            f"preprocessed {prep['preprocessed']} uploaded images: "
            f"decode, convert to RGB, resize to {w}×{h}, save the processed image in the database"
        )
        if prep["preprocessed"]:
            bits = ", ".join(f"{name} {n}" for name, n in prep["by_class"].items() if n)
            progress(f"preprocessed by class: {bits}")
    merged = merge_uploads_into_train()
    if progress:
        progress(f"merged {merged['moved']} preprocessed images into the training set")
    return merged


# --------------------------------------------------------------------------- #
# 3. Image processing
# --------------------------------------------------------------------------- #
def load_image(source, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Decode an image (path or bytes) -> float32 RGB array in [0, 255], resized."""
    if isinstance(source, (bytes, bytearray)):
        im = Image.open(io.BytesIO(source))
    else:
        im = Image.open(source)
    im = im.convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def make_tf_dataset(items: list[tuple[Path, int]], batch_size: int = 32,
                    training: bool = False, seed: int = 42):
    """Build a tf.data pipeline: decode -> resize -> (augment) -> batch -> prefetch.

    Pixel scaling to [-1, 1] is done inside the model (MobileNetV2 preprocess
    layer) so that raw [0,255] images can be fed at inference time.
    """
    import tensorflow as tf

    paths = [str(p) for p, _ in items]
    labels = [y for _, y in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        return tf.cast(img, tf.float32), label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        aug = augmentation_layer()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def augmentation_layer():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmentation")


def stratified_split(items: list[tuple[Path, int]], val_fraction: float = 0.15, seed: int = 42):
    from sklearn.model_selection import train_test_split
    labels = [y for _, y in items]
    tr, va = train_test_split(items, test_size=val_fraction, stratify=labels, random_state=seed)
    return tr, va


# --------------------------------------------------------------------------- #
# 4. Interpretable image features (for visualisation / storytelling)
# --------------------------------------------------------------------------- #
def image_features(arr: np.ndarray) -> dict:
    """Hand-crafted, human-interpretable features of a leaf image.

    - brightness:      mean luminance (0-255)
    - contrast:        std of luminance
    - mean_r/g/b:      average channel intensity
    - green_ratio:     share of pixels where green clearly dominates (healthy tissue)
    - brown_ratio:     share of reddish-brown / yellow-brown pixels (lesions, rust pustules)
    - edge_density:    share of strong gradient pixels (texture: spots/pustules add edges)
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    green = (g > r * 1.08) & (g > b * 1.08) & (g > 60)
    brown = (r > g * 0.95) & (r > b * 1.25) & (r > 70) & (lum < 200)
    gy, gx = np.gradient(lum)
    edges = np.hypot(gx, gy) > 25
    return {
        "brightness": float(lum.mean()),
        "contrast": float(lum.std()),
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "green_ratio": float(green.mean()),
        "brown_ratio": float(brown.mean()),
        "edge_density": float(edges.mean()),
    }


def compute_feature_stats(directory: Path = TRAIN_DIR, max_per_class: int | None = None,
                          save: bool = True) -> dict:
    """Compute features for every image and summarise per class.

    Returns {"per_image": [...], "summary": {class: {feature: {mean,std,hist}}}}.
    """
    rows = []
    for path, y in list_images(directory):
        cls = CLASS_NAMES[y]
        if max_per_class and sum(1 for r in rows if r["label"] == cls) >= max_per_class:
            continue
        arr = load_image(path, (128, 128))
        rows.append({"label": cls, "file": path.name, **image_features(arr)})

    feats = [k for k in rows[0] if k not in ("label", "file")] if rows else []
    summary = {}
    for f in feats:
        allv = np.array([r[f] for r in rows])
        lo, hi = float(allv.min()), float(allv.max())
        edges = np.linspace(lo, hi, 21)
        summary[f] = {"bins": edges.tolist(), "classes": {}}
        for c in CLASS_NAMES:
            v = np.array([r[f] for r in rows if r["label"] == c])
            if len(v) == 0:
                continue
            hist, _ = np.histogram(v, bins=edges)
            summary[f]["classes"][c] = {"mean": float(v.mean()), "std": float(v.std()),
                                        "median": float(np.median(v)), "hist": hist.tolist()}
    result = {"n_images": len(rows), "per_image": rows, "summary": summary,
              "counts": dataset_counts()}
    if save:
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    print(download_dataset())
    print(dataset_counts())
    s = compute_feature_stats()
    print(f"feature stats computed for {s['n_images']} images -> {STATS_PATH}")
