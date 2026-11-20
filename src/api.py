"""
FastAPI service for the Bean Leaf Disease pipeline.

Run locally:   uvicorn src.api:app --port 8000
Docs:          http://localhost:8000/docs
Dashboard UI:  http://localhost:8000/

Several replicas can run behind a load balancer and share the data/ and
models/ folders (a Docker volume). Retraining state lives in a JSON file and a
lock file so every replica sees it, and each replica hot-reloads the model
when the registry file changes.
"""
from __future__ import annotations

import collections
import fcntl
import io
import json
import os
import random
import socket
import threading
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import model as model_lib
from . import preprocessing as pp
from .prediction import Predictor

RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "30"))      # new images
DRIFT_ACCURACY_THRESHOLD = float(os.environ.get("DRIFT_ACCURACY_THRESHOLD", "0.80"))
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "true").lower() == "true"
STATUS_PATH = pp.DATA_DIR / "retrain_status.json"
LOCK_PATH = pp.DATA_DIR / ".retrain.lock"
UI_DIR = pp.ROOT / "ui"

START_TIME = time.time()
HOSTNAME = socket.gethostname()

app = FastAPI(title="Bean Leaf Disease Classifier",
              description="End-to-end ML pipeline: predict, upload data, retrain, monitor.",
              version="1.0.0")
predictor = Predictor()
_registry_mtime = model_lib.REGISTRY_PATH.stat().st_mtime if model_lib.REGISTRY_PATH.exists() else 0


# --------------------------------------------------------------------------- #
# Monitoring middleware (latency per endpoint, request counts, errors)
# --------------------------------------------------------------------------- #
class Monitor:
    def __init__(self, window: int = 2000):
        self.lock = threading.Lock()
        self.latencies = collections.defaultdict(lambda: collections.deque(maxlen=window))
        self.counts = collections.Counter()
        self.errors = collections.Counter()
        self.predictions = collections.Counter()

    def record(self, path: str, ms: float, status: int):
        with self.lock:
            self.latencies[path].append(ms)
            self.counts[path] += 1
            if status >= 500:
                self.errors[path] += 1

    def snapshot(self) -> dict:
        with self.lock:
            out = {}
            for path, lat in self.latencies.items():
                s = sorted(lat)
                out[path] = {
                    "count": self.counts[path], "errors": self.errors[path],
                    "avg_ms": round(sum(s) / len(s), 2),
                    "p50_ms": round(s[len(s) // 2], 2),
                    "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 2),
                    "max_ms": round(s[-1], 2),
                }
            return {"endpoints": out, "predictions_by_class": dict(self.predictions)}


monitor = Monitor()


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    _maybe_reload_model()
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ms = (time.perf_counter() - t0) * 1000
        path = request.url.path
        if not path.startswith(("/ui", "/static")) and path not in ("/", "/favicon.ico"):
            monitor.record(path, ms, status)


def _maybe_reload_model():
    """Hot-reload when another replica (or a retrain thread) promoted a model."""
    global _registry_mtime
    try:
        m = model_lib.REGISTRY_PATH.stat().st_mtime
    except FileNotFoundError:
        return
    if m != _registry_mtime:
        _registry_mtime = m
        predictor.reload()


# --------------------------------------------------------------------------- #
# Health / uptime
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["monitoring"])
def health():
    up = time.time() - START_TIME
    return {
        "status": "ok" if predictor.ready else "degraded",
        "model_ready": predictor.ready,
        "model_version": predictor.version,
        "uptime_seconds": round(up, 1),
        "started_at": datetime.fromtimestamp(START_TIME, timezone.utc).isoformat(timespec="seconds"),
        "container": HOSTNAME,
        "retraining": read_status().get("state") == "running",
    }


@app.get("/monitoring", tags=["monitoring"])
def monitoring():
    return {"container": HOSTNAME, "uptime_seconds": round(time.time() - START_TIME, 1),
            **monitor.snapshot()}


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@app.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(..., description="A bean leaf image")):
    data = await file.read()
    if not pp.validate_image_bytes(data):
        raise HTTPException(400, "file is not a valid image")
    if not predictor.ready:
        raise HTTPException(503, "model not trained yet")
    result = predictor.predict_bytes(data)
    monitor.predictions[result["label"]] += 1
    result["container"] = HOSTNAME
    return result


@app.post("/predict/batch", tags=["prediction"])
async def predict_batch(files: list[UploadFile] = File(...)):
    blobs = [await f.read() for f in files]
    good = [(f.filename, b) for f, b in zip(files, blobs) if pp.validate_image_bytes(b)]
    if not good:
        raise HTTPException(400, "no valid images")
    preds = predictor.predict_many([b for _, b in good])
    return {"results": [{"file": n, **p} for (n, _), p in zip(good, preds)]}


# --------------------------------------------------------------------------- #
# Data: stats for visualisations, samples, bulk upload
# --------------------------------------------------------------------------- #
@app.get("/data/stats", tags=["data"])
def data_stats():
    stats = json.loads(pp.STATS_PATH.read_text()) if pp.STATS_PATH.exists() else {}
    stats.pop("per_image", None)
    stats["counts"] = pp.dataset_counts()
    stats["pending_uploads"] = pp.pending_upload_count()
    stats["retrain_threshold"] = RETRAIN_THRESHOLD
    return stats


@app.get("/data/scatter", tags=["data"])
def data_scatter(x: str = "green_ratio", y: str = "brown_ratio", limit: int = 600):
    if not pp.STATS_PATH.exists():
        return {"points": []}
    rows = json.loads(pp.STATS_PATH.read_text())["per_image"]
    random.Random(0).shuffle(rows)
    return {"x": x, "y": y, "points": [{"label": r["label"], "x": r[x], "y": r[y]} for r in rows[:limit]]}


@app.get("/data/sample/{label}", tags=["data"])
def data_sample(label: str, seed: int | None = None):
    items = [p for p, y in pp.list_images(pp.TEST_DIR) if pp.CLASS_NAMES[y] == label]
    if not items:
        raise HTTPException(404, "unknown label")
    return FileResponse(random.Random(seed).choice(items))


@app.post("/upload", tags=["data"])
async def upload(background: BackgroundTasks,
                 files: list[UploadFile] = File(..., description="Images, or a .zip with class folders"),
                 label: str | None = Form(None, description="Class of the images (not needed for a zip)")):
    payload = [(f.filename or "file", await f.read()) for f in files]
    result = pp.save_uploaded_images(payload, label)
    pending = pp.pending_upload_count()
    result["pending_uploads"] = pending
    result["auto_retrain_triggered"] = False
    # ---- retraining trigger #1: enough new data has arrived ----------------
    if AUTO_RETRAIN and pending >= RETRAIN_THRESHOLD and read_status().get("state") != "running":
        background.add_task(start_retrain, f"auto: {pending} new images >= threshold {RETRAIN_THRESHOLD}")
        result["auto_retrain_triggered"] = True
    return result


# --------------------------------------------------------------------------- #
# Model: metrics, production evaluation, (re)training
# --------------------------------------------------------------------------- #
@app.get("/model/metrics", tags=["model"])
def model_metrics():
    info = model_lib.current_model_info()
    reg = json.loads(model_lib.REGISTRY_PATH.read_text()) if model_lib.REGISTRY_PATH.exists() else {}
    versions = [{k: v.get(k) for k in ("version", "created_at", "note", "promoted", "train_counts")}
                | {"accuracy": v["metrics"]["accuracy"], "f1_macro": v["metrics"]["f1_macro"]}
                for v in reg.get("versions", [])]
    return {"current": info, "versions": versions}


@app.post("/model/evaluate", tags=["model"])
async def evaluate_in_production(background: BackgroundTasks,
                                 files: list[UploadFile] = File(...),
                                 label: str | None = Form(None),
                                 retrain_on_drift: bool = Form(False)):
    """Evaluate the live model on newly labelled data (a zip of class folders,
    or images + a label). Reports all metrics and flags drift when accuracy
    falls below DRIFT_ACCURACY_THRESHOLD (retraining trigger #2)."""
    payload = [(f.filename or "file", await f.read()) for f in files]
    blobs, ys = [], []
    for name, data in payload:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for e in zf.infolist():
                    parts = Path(e.filename).parts
                    if e.is_dir() or "__MACOSX" in e.filename or len(parts) < 2:
                        continue
                    c = pp.normalise_label(parts[-2])
                    b = zf.read(e)
                    if c and pp.validate_image_bytes(b):
                        blobs.append(b); ys.append(pp.CLASS_NAMES.index(c))
        else:
            c = pp.normalise_label(label or "")
            if c and pp.validate_image_bytes(data):
                blobs.append(data); ys.append(pp.CLASS_NAMES.index(c))
    if not blobs:
        raise HTTPException(400, "no labelled images found (zip with class folders, or images + label)")
    t0 = time.perf_counter()
    probs = np.concatenate([predictor.predict_arrays(np.stack([pp.load_image(b) for b in blobs[i:i + 32]]))
                            for i in range(0, len(blobs), 32)])
    ms = (time.perf_counter() - t0) * 1000 / len(blobs)
    metrics = model_lib.metrics_from_probs(np.array(ys), probs, ms)
    metrics["model_version"] = predictor.version
    metrics["drift_detected"] = metrics["accuracy"] < DRIFT_ACCURACY_THRESHOLD
    metrics["drift_threshold"] = DRIFT_ACCURACY_THRESHOLD
    if metrics["drift_detected"] and retrain_on_drift:
        _save_labelled(blobs, ys)  # keep the hard examples for retraining
        background.add_task(start_retrain, f"drift: accuracy {metrics['accuracy']:.3f} < {DRIFT_ACCURACY_THRESHOLD}")
        metrics["retrain_triggered"] = True
    log = pp.DATA_DIR / "production_evaluations.jsonl"
    with log.open("a") as fh:
        fh.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             **{k: metrics[k] for k in ("model_version", "n_samples", "accuracy",
                                                         "f1_macro", "drift_detected")}}) + "\n")
    return metrics


def _save_labelled(blobs, ys):
    for b, y in zip(blobs, ys):
        pp.save_uploaded_images([("drift.jpg", b)], pp.CLASS_NAMES[y])


@app.post("/retrain", tags=["model"])
def retrain(background: BackgroundTasks, from_scratch: bool = Form(False)):
    """Manual retraining trigger (button in the UI)."""
    if read_status().get("state") == "running":
        raise HTTPException(409, "a retraining job is already running")
    reason = "manual: full training from ImageNet weights" if from_scratch else "manual"
    background.add_task(start_retrain, reason, from_scratch)
    return {"started": True, "reason": reason, "pending_uploads": pp.pending_upload_count()}


@app.get("/retrain/status", tags=["model"])
def retrain_status():
    return read_status()


# --------------------------------------------------------------------------- #
# Retraining job (runs in a background thread, guarded by a file lock so only
# one replica retrains at a time)
# --------------------------------------------------------------------------- #
def read_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"state": "idle", "log": []}


def write_status(**kw):
    s = read_status() | kw
    STATUS_PATH.write_text(json.dumps(s, indent=2))


def start_retrain(reason: str, from_scratch: bool = False):
    threading.Thread(target=_retrain_job, args=(reason, from_scratch), daemon=True).start()


def _retrain_job(reason: str, from_scratch: bool):
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lockf:
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # another replica is already retraining
        logs: list[str] = []

        def progress(msg):
            logs.append(f"{datetime.now().strftime('%H:%M:%S')} {msg}")
            write_status(log=logs[-50:])

        write_status(state="running", reason=reason, started_at=time.time(), finished_at=None,
                     container=HOSTNAME, log=[], result=None, error=None)
        try:
            progress(f"retraining started ({reason})")
            if from_scratch:
                pp.merge_uploads_into_train()
                m, h = model_lib.train(progress=progress)
                met = model_lib.evaluate(m)
                entry = model_lib.save_and_register(m, met, h, note=reason, promote=True)
            else:
                entry = model_lib.retrain(progress=progress)
            predictor.reload()
            result = {k: entry.get(k) for k in ("version", "promoted", "previous_accuracy", "new_images")}
            result["accuracy"] = entry["metrics"]["accuracy"]
            result["f1_macro"] = entry["metrics"]["f1_macro"]
            write_status(state="done", finished_at=time.time(), result=result)
        except Exception as exc:  # noqa: BLE001
            progress(f"ERROR: {exc}")
            write_status(state="failed", finished_at=time.time(), error=traceback.format_exc()[-2000:])
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


@app.get("/", include_in_schema=False)
def index():
    return HTMLResponse((UI_DIR / "index.html").read_text())


# reset a stale "running" status left behind by a crashed/restarted container
if read_status().get("state") == "running" and time.time() - read_status().get("started_at", 0) > 3600:
    write_status(state="failed", error="stale job reset on startup")
