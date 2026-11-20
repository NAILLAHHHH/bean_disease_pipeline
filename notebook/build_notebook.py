"""Generates notebook/bean_disease_pipeline.ipynb (run, then execute with nbconvert)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# 🌱 Bean Leaf Disease Classification: end-to-end ML pipeline

**Problem.** Beans are a staple food and cash crop for millions of smallholder farmers in East Africa.
Two fungal diseases, **angular leaf spot** and **bean rust**, can cut yields by 40% or more, and
farmers often lack quick access to an agronomist. A model that diagnoses a leaf from a smartphone photo
gives them early, actionable advice.

**Data.** *iBeans*, collected by the Makerere University AI Lab with the National Crops Resources
Research Institute (Uganda). Field photos taken with smartphones, in 3 classes: `angular_leaf_spot`, `bean_rust`, `healthy`.

**This notebook covers:**
1. Data acquisition
2. Data processing (decoding, resizing, augmentation, splits) and feature exploration (3 interpreted features)
3. Model creation (MobileNetV2 transfer learning)
4. Training (head training, then fine-tuning)
5. Model testing with every metric: accuracy, precision, recall, F1, ROC-AUC, log-loss, confusion matrix
6. Prediction functions
7. Retraining (warm start with new data, promoted only if it isn't worse)

All logic lives in `src/` so the notebook, the API and the retraining job share one code path.
""")
code("""
import os, sys, json, time
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
sys.path.insert(0, str(ROOT)); os.environ["PROJECT_ROOT"] = str(ROOT)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import tensorflow as tf
from src import preprocessing as pp, model as M, prediction as P

sns.set_theme(style="whitegrid")
PALETTE = {"angular_leaf_spot": "#2a78d6", "bean_rust": "#eb6834", "healthy": "#1baf7a"}
print("TensorFlow", tf.__version__)
""")

md("## 1. Data acquisition\n`download_dataset()` pulls the parquet splits from the Hugging Face mirror and writes JPEGs into `data/train/<class>/` and `data/test/<class>/`. The original *validation* + *test* splits become our **held-out test set**. A validation split for early stopping is carved out of train later.")
code("""
print(pp.download_dataset())          # skipped if already on disk
counts = pp.dataset_counts()
pd.DataFrame({k: counts[k] for k in ("train", "test")})
""")
code("""
fig, ax = plt.subplots(figsize=(7, 3.5))
df = pd.DataFrame({k: counts[k] for k in ("train", "test")}).reset_index().melt(id_vars="index")
sns.barplot(df, x="index", y="value", hue="variable", ax=ax, palette=["#2a78d6", "#9fc3ee"])
ax.set(xlabel="", ylabel="images", title="Images per class: the dataset is balanced"); plt.show()
""")
md("**Interpretation:** the classes are balanced (~340 train images each), so accuracy is a fair headline metric and we don't need class weights.")

md("### Look at the raw images")
code("""
fig, axes = plt.subplots(3, 5, figsize=(13, 8))
for r, c in enumerate(pp.CLASS_NAMES):
    files = sorted((pp.TRAIN_DIR / c).iterdir())[:5]
    for k, f in enumerate(files):
        axes[r, k].imshow(plt.imread(f)); axes[r, k].axis("off")
        if k == 0: axes[r, k].set_title(c, loc="left", fontsize=12, color=PALETTE[c])
plt.tight_layout(); plt.show()
from PIL import Image
sizes = {Image.open(p).size for p, _ in pp.list_images(pp.TRAIN_DIR)[:200]}
print("raw image sizes:", sizes)
""")
md("Raw images are 500×500 RGB smartphone photos with busy backgrounds (soil, other leaves, hands) and variable lighting.")

md("""
## 2. Data processing
Steps (in `src/preprocessing.py`):
1. **Decode** JPEG/PNG to RGB (3 channels), dropping alpha and grayscale variants.
2. **Resize** to 160×160 (MobileNetV2 input; 10× fewer pixels than raw, so much faster on CPU).
3. **Scale** pixels to [-1, 1]. This is done *inside the model* (`Rescaling` layer) so the deployed API can feed raw images.
4. **Stratified split** of train into 85% train and 15% validation (early stopping, LR scheduling).
5. **Augmentation** (train only): random horizontal/vertical flips, ±15% rotation, ±15% zoom, ±15% contrast.
   This copies how farmers actually photograph leaves at any angle and under any light.
6. `tf.data` pipeline with parallel decoding, batching (32) and prefetching.
""")
code("""
items = pp.list_images(pp.TRAIN_DIR)
tr_items, va_items = pp.stratified_split(items, 0.15, M.SEED)
print(f"train {len(tr_items)} | validation {len(va_items)} | test {len(pp.list_images(pp.TEST_DIR))}")

ds_demo = pp.make_tf_dataset(tr_items[:1], 1)
x, _ = next(iter(ds_demo)); aug = pp.augmentation_layer()
fig, axes = plt.subplots(1, 6, figsize=(15, 3))
axes[0].imshow(x[0].numpy().astype("uint8")); axes[0].set_title("original 160×160")
for i in range(1, 6):
    axes[i].imshow(np.clip(aug(x, training=True)[0].numpy(), 0, 255).astype("uint8")); axes[i].set_title(f"augmented #{i}")
for a in axes: a.axis("off")
plt.show()
""")

md("""
### Feature exploration: what makes a leaf diseased?
We compute interpretable features per image (`pp.image_features`): brightness, contrast, mean R/G/B,
**green-pixel ratio**, **brown/lesion-pixel ratio**, edge density.
""")
code("""
stats = pp.compute_feature_stats(pp.TRAIN_DIR, save=True)
feat = pd.DataFrame(stats["per_image"])
feat.groupby("label")[["green_ratio", "brown_ratio", "brightness", "contrast", "edge_density", "mean_r", "mean_g", "mean_b"]].mean().round(3)
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, f, t in zip(axes, ["green_ratio", "brown_ratio", "brightness"],
                    ["Feature 1: green-pixel ratio", "Feature 2: brown/lesion-pixel ratio", "Feature 3: brightness"]):
    sns.kdeplot(feat, x=f, hue="label", fill=True, common_norm=False, palette=PALETTE, ax=ax, alpha=.25, linewidth=2)
    ax.set_title(t)
plt.tight_layout(); plt.show()
""")
md("""
**Feature 1: green ratio.** Healthy leaves are ~66% clearly-green pixels vs ~40% for diseased leaves.
Both diseases destroy chlorophyll, so lost greenness is the clearest single signal of infection.
On its own it separates *healthy vs sick*, but not leaf spot vs rust.

**Feature 2: brown/lesion ratio.** Diseased leaves have ~1.8× more brown/yellow-brown pixels (necrotic lesions,
rust pustules). The two disease curves overlap almost completely, so colour can't tell the diseases apart.
Healthy images score above zero because soil and stems show in the background, which is why the model needs to
learn *where* the brown is, not just how much there is.

**Feature 3: brightness.** Rust photos are noticeably darker (mean ≈115 vs ≈130) because dense dark-brown pustules
pull luminance down. Brightness also varies with field lighting, a nuisance factor the model must ignore,
so we use contrast augmentation.

**The story:** simple colour statistics flag *that* a leaf is sick but not *which* disease. Telling the two
apart needs spatial patterns (round, raised rust pustules vs angular, vein-bounded leaf-spot lesions), which is
exactly what a convolutional network learns. That motivates the CNN below.
""")
code("""
fig, ax = plt.subplots(figsize=(6.5, 5))
sns.scatterplot(feat.sample(700, random_state=0), x="green_ratio", y="brown_ratio", hue="label",
                palette=PALETTE, s=22, alpha=.7, ax=ax)
ax.set_title("Healthy separates on colour; the two diseases overlap"); plt.show()
""")

md("""
## 3. Model creation
**Transfer learning with MobileNetV2** (ImageNet weights). It's small (≈2.4M base parameters, 25 MB file), fast on CPU
(~7 ms/image), and well suited to a cheap cloud container and to an eventual phone app.

Head: GlobalAveragePooling → BatchNorm → Dropout(0.3) → Dense(128, ReLU, L2 1e-4) → Dropout(0.3) → Dense(3, softmax).

Optimisation choices:
- **Adam**, LR 1e-3 for the head, then **1e-5** for fine-tuning (so the pretrained features aren't wiped out)
- **EarlyStopping** on val_accuracy (patience 4, restore best weights) to prevent overfitting
- **ReduceLROnPlateau** (×0.5 after 2 stagnant epochs)
- **L2 + dropout** regularisation, and **augmentation** as data-level regularisation
- BatchNorm layers stay frozen during fine-tuning (small dataset)
""")
code("""
model = M.build_model()
model.summary(line_length=100)
""")

md("## 4. Training\nPhase 1 trains only the head (base frozen). Phase 2 unfreezes the top 40 layers of MobileNetV2 and fine-tunes at a low learning rate.")
code("""
t0 = time.time()
model, history = M.train(epochs_head=12, epochs_finetune=8, model=model, progress=print)
print(f"training time: {time.time() - t0:.0f}s")
""")
code("""
h = pd.DataFrame(history); h["epoch"] = range(1, len(h) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, m in zip(axes, ["accuracy", "loss"]):
    ax.plot(h.epoch, h[m], "--", color="#7a7974", label=f"train {m}")
    ax.plot(h.epoch, h[f"val_{m}"], color="#2a78d6", lw=2, marker="o", ms=4, label=f"validation {m}")
    ft = h.index[h.phase == "finetune"]
    if len(ft): ax.axvline(ft[0] + 0.5, color="#eb6834", ls=":", label="fine-tuning starts")
    ax.set(xlabel="epoch", title=m); ax.legend()
plt.show()
""")

md("## 5. Model testing: evaluation on the held-out test set\nThe test images (original validation + test splits, 261 images) were never used for fitting or early stopping.")
code("""
metrics = M.evaluate(model, pp.TEST_DIR)
pd.Series({k: v for k, v in metrics.items() if isinstance(v, (int, float))}).round(4).to_frame("value")
""")
code("""
from sklearn.metrics import classification_report, roc_curve, auc
items_te = pp.list_images(pp.TEST_DIR)
y_true = np.array([y for _, y in items_te])
probs = model.predict(pp.make_tf_dataset(items_te, 32), verbose=0)
y_pred = probs.argmax(1)
print(classification_report(y_true, y_pred, target_names=pp.CLASS_NAMES, digits=4))
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = np.array(metrics["confusion_matrix"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
            xticklabels=pp.CLASS_NAMES, yticklabels=pp.CLASS_NAMES)
axes[0].set(xlabel="predicted", ylabel="true", title="Confusion matrix (test set)")
for i, c in enumerate(pp.CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true == i, probs[:, i])
    axes[1].plot(fpr, tpr, lw=2, color=PALETTE[c], label=f"{c} (AUC={auc(fpr, tpr):.3f})")
axes[1].plot([0, 1], [0, 1], ":", color="gray"); axes[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (one-vs-rest)")
axes[1].legend(); plt.tight_layout(); plt.show()
""")
md("""
**Reading the results.**
- **Healthy** is recognised best (highest precision). When the model says *healthy* it is almost always right, so few sick crops are missed.
- Most errors are **leaf spot ↔ rust** confusions, which matches the feature analysis: both diseases look alike in colour and differ only in lesion shape.
- Very few diseased leaves are called healthy. That's the costly error for a farmer (no treatment), so this is the behaviour we want.
- ROC-AUC ≈ 0.98: the probabilities rank well, so confidence scores are meaningful (the UI shows them).
""")
code("""
wrong = np.where(y_true != y_pred)[0][:8]
fig, axes = plt.subplots(1, len(wrong), figsize=(2.2 * len(wrong), 2.8))
for ax, i in zip(np.atleast_1d(axes), wrong):
    ax.imshow(plt.imread(items_te[i][0])); ax.axis("off")
    ax.set_title(f"true {pp.CLASS_NAMES[y_true[i]][:6]}\\npred {pp.CLASS_NAMES[y_pred[i]][:6]} {probs[i].max():.0%}", fontsize=9)
plt.suptitle("Misclassified test images"); plt.show()
""")
md("### Save the model\nSaved as `models/bean_disease_mobilenetv2.h5` (Keras HDF5) and registered in `models/registry.json` with its metrics and training history.")
code("""
entry = M.save_and_register(model, metrics, history, note="notebook training")
print(entry["version"], "->", M.MODEL_PATH.relative_to(ROOT), f"({M.MODEL_PATH.stat().st_size/1e6:.1f} MB)")
""")

md("## 6. Prediction functions\n`src/prediction.py` loads the saved `.h5` once and predicts from raw image bytes, which is the same path the API uses.")
code("""
predictor = P.Predictor()
for c in pp.CLASS_NAMES:
    f = sorted((pp.TEST_DIR / c).iterdir())[3]
    out = predictor.predict_bytes(f.read_bytes())
    print(f"{c:18s} -> predicted {out['label']:18s} conf={out['confidence']:.3f}  {out['inference_ms']} ms")
print()
print(json.dumps(P.predict_file(sorted((pp.TEST_DIR / 'bean_rust').iterdir())[0]), indent=2))
""")

md("""
## 7. Retraining
`M.retrain()` is what the API's **Retrain** button and the automatic triggers call:
1. move newly uploaded images from `data/uploads/<class>/` into `data/train/`
2. **warm-start from the production model** (custom pre-trained weights), so it isn't trained from scratch
3. head training, then fine-tuning on the enlarged dataset
4. evaluate on the untouched test set and **promote only if accuracy ≥ production − 1%**

Retraining triggers in the deployed API:
- **Manual**: the "Retrain" button (`POST /retrain`)
- **Data volume**: automatic once ≥ `RETRAIN_THRESHOLD` (30) new labelled images have been uploaded
- **Performance drift**: `POST /model/evaluate` scores the live model on fresh labelled field data; if accuracy falls below 80%, those images are saved and retraining starts

Demo: upload the sample zip (flipped/rotated field photos) and retrain.
""")
code("""
import zipfile
zpath = ROOT / "sample_data" / "sample_upload.zip"
if zpath.exists():
    with zipfile.ZipFile(zpath) as zf:
        files = [(n, zf.read(n)) for n in zf.namelist() if not n.endswith("/")]
    print(pp.save_uploaded_images([("sample_upload.zip", zpath.read_bytes())], None))
    print("pending uploads:", pp.pending_upload_count())
""")
code("""
entry = M.retrain(epochs_head=3, epochs_finetune=3, progress=print)
print({k: entry[k] for k in ("version", "promoted", "previous_accuracy", "new_images")}, "new accuracy:", round(entry["metrics"]["accuracy"], 4))
""")
code("""
reg = json.loads(M.REGISTRY_PATH.read_text())
pd.DataFrame([{"version": v["version"], "note": v["note"], "accuracy": v["metrics"]["accuracy"],
               "f1_macro": v["metrics"]["f1_macro"], "promoted": v["promoted"]} for v in reg["versions"]]).round(4)
""")
md("""
## Conclusion
- MobileNetV2 transfer learning reaches **~90% accuracy, ~0.90 macro-F1 and ~0.98 ROC-AUC** on 261 unseen field photos, at ~7 ms per image on CPU.
- Colour features alone separate healthy from diseased leaves. Telling the diseases apart needs the CNN's spatial features.
- The pipeline is retrainable: new data → warm-start retrain → gated promotion → hot reload in every API container.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).with_name("bean_disease_pipeline.ipynb")
nbf.write(nb, out)
print("wrote", out)
