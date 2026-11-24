"""
Model creation, training, evaluation and retraining.

Model: MobileNetV2 (ImageNet weights) used as a feature extractor + a small
classification head, trained with transfer learning in two phases:

    Phase 1 - base frozen, train only the head          (fast, stable)
    Phase 2 - unfreeze the top of the base, low LR       (fine-tuning)

Retraining reuses the *current production model* as its starting point
(warm start), adds any uploaded images to the training set, evaluates the
candidate on the held-out test set and only promotes it if it is not worse
than the production model (guard against bad uploads).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import database as db
from . import preprocessing as pp

MODELS_DIR = pp.ROOT / "models"
MODEL_PATH = MODELS_DIR / "bean_disease_mobilenetv2.h5"
VERSIONS_DIR = MODELS_DIR / "versions"
METRICS_PATH = MODELS_DIR / "metrics.json"
REGISTRY_PATH = MODELS_DIR / "registry.json"

SEED = 42


# --------------------------------------------------------------------------- #
# Model creation
# --------------------------------------------------------------------------- #
def build_model(num_classes: int = len(pp.CLASS_NAMES), dropout: float = 0.3,
                l2: float = 1e-4, base_weights: str | None = "imagenet"):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)

    inputs = tf.keras.Input(shape=(*pp.IMG_SIZE, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_to_-1_1")(inputs)
    base = tf.keras.applications.MobileNetV2(input_shape=(*pp.IMG_SIZE, 3), include_top=False,
                                             weights=base_weights)
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inputs, outputs, name="bean_disease_mobilenetv2")
    compile_model(model, 1e-3)
    return model


def compile_model(model, lr: float):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def _base_of(model):
    return next(l for l in model.layers if l.name.startswith("mobilenetv2"))


def unfreeze_top(model, n_layers: int = 40, lr: float = 1e-5):
    base = _base_of(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:  # keep BN statistics frozen for small datasets
        if layer.__class__.__name__ == "BatchNormalization":
            layer.trainable = False
    compile_model(model, lr)


def _callbacks(patience: int = 4):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                         restore_best_weights=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                             min_lr=1e-7),
    ]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(epochs_head: int = 12, epochs_finetune: int = 8, batch_size: int = 32,
          model=None, progress=None) -> tuple:
    """Train on data/train (with a stratified validation split).

    Returns (model, history_dict).
    """
    items = pp.list_images(pp.TRAIN_DIR)
    tr, va = pp.stratified_split(items, 0.15, SEED)
    ds_tr = pp.make_tf_dataset(tr, batch_size, training=True)
    ds_va = pp.make_tf_dataset(va, batch_size)

    if model is None:
        model = build_model()
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "phase": []}

    def _log(h, phase):
        for k in ("loss", "val_loss", "accuracy", "val_accuracy"):
            history[k] += [float(v) for v in h.history[k]]
        history["phase"] += [phase] * len(h.history["loss"])

    if progress:
        progress(f"phase 1: training head on {len(tr)} images ({len(va)} for validation)")
    if epochs_head:
        _log(model.fit(ds_tr, validation_data=ds_va, epochs=epochs_head,
                       callbacks=_callbacks(), verbose=2), "head")
    if epochs_finetune:
        if progress:
            progress("phase 2: fine-tuning top of MobileNetV2")
        unfreeze_top(model)
        _log(model.fit(ds_tr, validation_data=ds_va, epochs=epochs_finetune,
                       callbacks=_callbacks(), verbose=2), "finetune")
    return model, history


# --------------------------------------------------------------------------- #
# Testing / evaluation
# --------------------------------------------------------------------------- #
def evaluate(model, directory: Path = pp.TEST_DIR, batch_size: int = 32) -> dict:
    """Full evaluation on a labelled folder: accuracy, precision, recall, F1
    (macro + per class), ROC-AUC (one-vs-rest), log-loss and confusion matrix."""
    items = pp.list_images(directory)
    if not items:
        raise ValueError(f"no labelled images found in {directory}")
    ds = pp.make_tf_dataset(items, batch_size)
    y_true = np.array([y for _, y in items])
    t0 = time.perf_counter()
    probs = model.predict(ds, verbose=0)
    infer_ms = (time.perf_counter() - t0) * 1000 / len(items)
    return metrics_from_probs(y_true, probs, infer_ms)


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, infer_ms: float | None = None) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                                 precision_recall_fscore_support, roc_auc_score)
    labels = list(range(len(pp.CLASS_NAMES)))
    y_pred = probs.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(s[i])} for i, c in enumerate(pp.CLASS_NAMES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": pp.CLASS_NAMES,
    }
    try:
        out["log_loss"] = float(log_loss(y_true, np.clip(probs, 1e-7, 1), labels=labels))
        out["roc_auc_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", labels=labels))
    except ValueError:  # e.g. only one class present in a small production batch
        out["log_loss"] = out["roc_auc_ovr"] = None
    if infer_ms is not None:
        out["avg_inference_ms_per_image"] = float(infer_ms)
    return out


# --------------------------------------------------------------------------- #
# Persistence / registry
# --------------------------------------------------------------------------- #
def load_model(path: Path = MODEL_PATH):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def _read_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"current": None, "versions": []}


def save_and_register(model, metrics: dict, history: dict, note: str, promote: bool = True) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    reg = _read_registry()
    version = f"v{len(reg['versions']) + 1}"
    vpath = VERSIONS_DIR / f"bean_disease_mobilenetv2_{version}.h5"
    model.save(vpath)
    entry = {
        "version": version,
        "file": str(vpath.relative_to(pp.ROOT)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "train_counts": pp.dataset_counts()["train"],
        "metrics": metrics,
        "history": history,
        "promoted": promote,
    }
    reg["versions"].append(entry)
    if promote:
        shutil.copy(vpath, MODEL_PATH)
        reg["current"] = version
        METRICS_PATH.write_text(json.dumps(entry, indent=2))
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    return entry


def current_model_info() -> dict:
    if METRICS_PATH.exists():
        info = json.loads(METRICS_PATH.read_text())
        return info
    return {}


# --------------------------------------------------------------------------- #
# Retraining
# --------------------------------------------------------------------------- #
def retrain(epochs_head: int = 4, epochs_finetune: int = 4, min_gain: float = -0.01,
            progress=None) -> dict:
    """Retrain the production model with newly uploaded data.

    1. Preprocess rows in the upload database (decode, RGB, resize to 160×160).
    2. Move those images into data/train/<class>/.
    3. Warm-start from the current production model (the custom pre-trained weights).
    4. Train (head, then fine-tune), evaluate on the untouched test set.
    5. Promote only if test accuracy >= production accuracy + min_gain.
    """
    log = progress or (lambda m: None)
    prepared = pp.prepare_uploads_for_training(progress=log)
    moved = prepared["moved"]

    prev = current_model_info()
    prev_acc = prev.get("metrics", {}).get("accuracy")
    if MODEL_PATH.exists():
        log(f"warm-starting from production model {prev.get('version')}")
        model = load_model()
        # make sure the base starts frozen again for the head phase
        _base_of(model).trainable = False
        compile_model(model, 3e-4)
    else:
        log("no production model found - training from ImageNet weights")
        model = build_model()

    model, history = train(epochs_head, epochs_finetune, model=model, progress=log)
    log("evaluating candidate on held-out test set")
    metrics = evaluate(model)
    promote = prev_acc is None or metrics["accuracy"] >= prev_acc + min_gain
    entry = save_and_register(model, metrics, history,
                              note=f"retrain with {moved} new images", promote=promote)
    db.set_trained_version(prepared["ids"], entry["version"])
    entry["previous_accuracy"] = prev_acc
    entry["new_images"] = moved
    log(f"candidate {entry['version']} accuracy={metrics['accuracy']:.4f} "
        f"(prev={prev_acc}) -> {'PROMOTED' if promote else 'kept previous model'}")
    pp.compute_feature_stats()  # refresh dashboard visualisations
    return entry


if __name__ == "__main__":
    m, h = train()
    met = evaluate(m)
    print(json.dumps({k: v for k, v in met.items() if k != "per_class"}, indent=2))
    save_and_register(m, met, h, note="initial training")
