"""
Prediction utilities: load the production model once, predict single images
or batches, and hot-swap the model after a retraining run.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import model as model_lib
from . import preprocessing as pp

ADVICE = {
    "angular_leaf_spot": "Angular leaf spot (Pseudocercospora griseola): remove infected leaves, "
                         "avoid overhead watering, rotate crops and use certified clean seed.",
    "bean_rust": "Bean rust (Uromyces appendiculatus): remove infected debris, improve air flow, "
                 "consider a recommended fungicide and plant rust-resistant varieties.",
    "healthy": "No disease detected. Keep monitoring the crop regularly.",
}


class Predictor:
    """Thread-safe wrapper around the production Keras model."""

    def __init__(self):
        self._lock = threading.RLock()
        self.model = None
        self.version = None
        self.loaded_at = None
        self.reload()

    def reload(self):
        with self._lock:
            if not model_lib.MODEL_PATH.exists():
                self.model, self.version = None, None
                return False
            m = model_lib.load_model()
            m.predict(np.zeros((1, *pp.IMG_SIZE, 3), np.float32), verbose=0)  # warm-up
            self.model = m
            self.version = model_lib.current_model_info().get("version", "unknown")
            self.loaded_at = time.time()
            return True

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict_arrays(self, batch: np.ndarray) -> np.ndarray:
        with self._lock:
            if self.model is None:
                raise RuntimeError("model not loaded - train a model first")
            # direct call is much faster than .predict() for small batches
            return np.asarray(self.model(batch, training=False))

    def predict_bytes(self, data: bytes) -> dict:
        arr = pp.load_image(data)[None, ...]
        t0 = time.perf_counter()
        probs = self.predict_arrays(arr)[0]
        ms = (time.perf_counter() - t0) * 1000
        return format_prediction(probs, ms, self.version)

    def predict_many(self, blobs: list[bytes]) -> list[dict]:
        arr = np.stack([pp.load_image(b) for b in blobs])
        probs = self.predict_arrays(arr)
        return [format_prediction(p, None, self.version) for p in probs]


def format_prediction(probs: np.ndarray, ms: float | None, version: str | None) -> dict:
    idx = int(np.argmax(probs))
    label = pp.CLASS_NAMES[idx]
    return {
        "label": label,
        "confidence": float(probs[idx]),
        "probabilities": {c: float(p) for c, p in zip(pp.CLASS_NAMES, probs)},
        "advice": ADVICE[label],
        "inference_ms": round(ms, 2) if ms is not None else None,
        "model_version": version,
    }


def predict_file(path) -> dict:
    """Convenience helper for notebooks / CLI: predict a single image file."""
    with open(path, "rb") as f:
        return Predictor().predict_bytes(f.read())


if __name__ == "__main__":
    import sys
    print(predict_file(sys.argv[1]))
