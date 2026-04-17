"""
ph_vs_cnn_classifier.py
───────────────────────
Compares three feature modalities on Gleason grade classification:
  1. PH features  — ranked persistence vector (baseline paper method)
  2. CNN features — 128-dim vector from a pretrained ResNet18 (ImageNet)
  3. Combined    — PH + CNN concatenated

Classifier: Random Forest with stratified k-fold cross-validation
Metrics: macro-F1, per-class F1, balanced accuracy, confusion matrix

Inputs (from existing cache):
  cache/roi_images.npz          — (N,128,128,3) float32 thumbnails
  cache/roi_meta.npz            — grades
  cache/ph_vectors.npz          — X_raw ranked PH vectors

Outputs:
  clf_metrics.csv               — per-fold and mean metrics
  clf_comparison.png            — bar chart + confusion matrices
  clf_roc.png                   — ROC curves per class per modality

Run:
  python ph_vs_cnn_classifier.py [--force-cnn]
  --force-cnn   recompute CNN embeddings even if cached
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    f1_score, balanced_accuracy_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR       = "cache"
CACHE_IMAGES    = os.path.join(CACHE_DIR, "roi_images.npz")
CACHE_META      = os.path.join(CACHE_DIR, "roi_meta.npz")
CACHE_PH        = os.path.join(CACHE_DIR, "ph_vectors.npz")
CACHE_CNN       = os.path.join(CACHE_DIR, "cnn_embeddings.npz")

OUT_CSV         = "clf_metrics.csv"
OUT_BAR         = "clf_comparison.png"
OUT_ROC         = "clf_roc.png"
OUT_CONF        = "clf_confusion.png"

CNN_DIM         = 128       # final embedding dimension after projection
N_FOLDS         = 5         # stratified k-fold
N_TREES         = 500       # random forest trees
RANDOM_SEED     = 42
BATCH_SIZE      = 64        # CNN inference batch size

FORCE_CNN = "--force-cnn" in sys.argv[1:]

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── 1. load cached data ───────────────────────────────────────────────────────
ts("Loading cached data …")
all_imgs   = np.load(CACHE_IMAGES)["thumbnails"].astype(np.float32)  # (N,128,128,3)
grades_all = np.load(CACHE_META, allow_pickle=True)["grades"].astype(np.int32)
X_ph_raw   = np.load(CACHE_PH)["X_raw"].astype(np.float32)

N = len(all_imgs)
ts(f"  {N} ROIs  grades: { {g: int((grades_all==g).sum()) for g in [3,4,5]} }")
ts(f"  PH feature shape: {X_ph_raw.shape}")

# ── 2. CNN embeddings ─────────────────────────────────────────────────────────
if not FORCE_CNN and os.path.exists(CACHE_CNN):
    ts("Loading cached CNN embeddings …")
    X_cnn = np.load(CACHE_CNN)["embeddings"].astype(np.float32)
    ts(f"  Loaded {X_cnn.shape}")
else:
    ts("Extracting CNN embeddings (ResNet18 + projection head) …")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts(f"  device: {device}")

    # ResNet18 pretrained on ImageNet, remove final FC layer
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final layer with a projection to CNN_DIM
    in_features = backbone.fc.in_features   # 512 for ResNet18
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, CNN_DIM)
    )
    backbone = backbone.to(device)
    backbone.eval()

    # ImageNet normalisation
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    embeddings = []
    with torch.no_grad():
        for start in range(0, N, BATCH_SIZE):
            batch_imgs = all_imgs[start:start+BATCH_SIZE]
            tensors    = []
            for img in batch_imgs:
                # img is float32 [0,1], shape (128,128,3)
                pil = Image.fromarray((img * 255).astype(np.uint8))
                tensors.append(preprocess(pil))
            batch = torch.stack(tensors).to(device)
            emb   = backbone(batch).cpu().numpy()
            embeddings.append(emb)
            if (start + BATCH_SIZE) % 100 == 0 or start+BATCH_SIZE >= N:
                ts(f"  {min(start+BATCH_SIZE, N)}/{N} …")

    X_cnn = np.vstack(embeddings).astype(np.float32)
    ts(f"  CNN embedding shape: {X_cnn.shape}")
    np.savez_compressed(CACHE_CNN, embeddings=X_cnn)
    ts(f"  Saved → {CACHE_CNN}")

# ── 3. build feature sets ─────────────────────────────────────────────────────
# PH: standardise (RF doesn't strictly need this but helps with combined)
X_ph  = X_ph_raw.copy()
X_comb = np.concatenate([X_ph, X_cnn], axis=1)

feature_sets = {
    "PH only"   : X_ph,
    "CNN only"  : X_cnn,
    "PH + CNN"  : X_comb,
}

ts("\nFeature matrix shapes:")
for name, X in feature_sets.items():
    ts(f"  {name:<15} {X.shape}")

# ── 4. classifier ─────────────────────────────────────────────────────────────
def make_pipeline(X):
    """StandardScaler + RandomForest pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=N_TREES,
            class_weight="balanced",   # handles G5 imbalance
            random_state=RANDOM_SEED,
            n_jobs=-1
        ))
    ])

def run_cv(X, y, label):
    """
    Stratified k-fold cross-validation.
    Returns per-fold metrics and OOF predictions for ROC/confusion.
    """
    ts(f"  CV: {label} …")
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    grades = [3, 4, 5]

    fold_metrics = []
    all_true, all_pred, all_proba = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe = make_pipeline(X)
        pipe.fit(X_tr, y_tr)
        y_pred  = pipe.predict(X_te)
        y_proba = pipe.predict_proba(X_te)   # (n_test, 3)

        macro_f1 = f1_score(y_te, y_pred, average="macro", labels=grades)
        bal_acc  = balanced_accuracy_score(y_te, y_pred)
        per_cls  = f1_score(y_te, y_pred, average=None, labels=grades)

        fold_metrics.append({
            "fold"      : fold+1,
            "macro_f1"  : macro_f1,
            "bal_acc"   : bal_acc,
            "f1_G3"     : per_cls[0],
            "f1_G4"     : per_cls[1],
            "f1_G5"     : per_cls[2],
        })

        all_true.extend(y_te.tolist())
        all_pred.extend(y_pred.tolist())
        all_proba.append(y_proba)

        ts(f"    fold {fold+1}: macro-F1={macro_f1:.3f}  bal_acc={bal_acc:.3f}  "
           f"F1[G3={per_cls[0]:.3f} G4={per_cls[1]:.3f} G5={per_cls[2]:.3f}]")

    all_proba = np.vstack(all_proba)
    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)

    df_folds = pd.DataFrame(fold_metrics)
    means    = df_folds.mean(numeric_only=True)
    stds     = df_folds.std(numeric_only=True)

    ts(f"  MEAN macro-F1={means['macro_f1']:.3f}±{stds['macro_f1']:.3f}  "
       f"bal_acc={means['bal_acc']:.3f}±{stds['bal_acc']:.3f}  "
       f"F1[G3={means['f1_G3']:.3f} G4={means['f1_G4']:.3f} G5={means['f1_G5']:.3f}]")

    return {
        "label"       : label,
        "fold_df"     : df_folds,
        "means"       : means,
        "stds"        : stds,
        "all_true"    : all_true,
        "all_pred"    : all_pred,
        "all_proba"   : all_proba,
    }

ts("\nRunning cross-validation …")
results = {}
for name, X in feature_sets.items():
    results[name] = run_cv(X, grades_all, name)

# ── 5. save metrics CSV ───────────────────────────────────────────────────────
rows = []
for name, res in results.items():
    m, s = res["means"], res["stds"]
    rows.append({
        "modality"        : name,
        "macro_f1_mean"   : m["macro_f1"],
        "macro_f1_std"    : s["macro_f1"],
        "bal_acc_mean"    : m["bal_acc"],
        "bal_acc_std"     : s["bal_acc"],
        "f1_G3_mean"      : m["f1_G3"],
        "f1_G3_std"       : s["f1_G3"],
        "f1_G4_mean"      : m["f1_G4"],
        "f1_G4_std"       : s["f1_G4"],
        "f1_G5_mean"      : m["f1_G5"],
        "f1_G5_std"       : s["f1_G5"],
    })
df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False)
ts(f"Saved metrics → {OUT_CSV}")

# ── 6. bar chart comparison ───────────────────────────────────────────────────
ts("Plotting comparison bar chart …")

modalities = list(results.keys())
metrics    = [
    ("macro_f1",  "Macro F1"),
    ("bal_acc",   "Balanced Accuracy"),
    ("f1_G3",     "F1 — Gleason 3"),
    ("f1_G4",     "F1 — Gleason 4"),
    ("f1_G5",     "F1 — Gleason 5"),
]
colors = ["#2c3e50", "#e74c3c", "#27ae60"]

fig_b, axes_b = plt.subplots(1, len(metrics), figsize=(20, 5))
fig_b.patch.set_facecolor("white")

x = np.arange(len(modalities))
for ax, (mkey, mtitle) in zip(axes_b, metrics):
    vals = [results[m]["means"][mkey]  for m in modalities]
    errs = [results[m]["stds"][mkey]   for m in modalities]
    bars = ax.bar(x, vals, 0.55, color=colors,
                  yerr=errs, capsize=5,
                  error_kw={"elinewidth":1.2, "ecolor":"#333"})
    ax.set_xticks(x)
    ax.set_xticklabels(modalities, fontsize=9)
    ax.set_title(mtitle, fontsize=10, fontweight="bold")
    ax.set_ylim(0, min(1.05, max(vals)*1.35 + 0.05))
    ax.spines[["top","right"]].set_visible(False)
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+e+0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

fig_b.suptitle(
    f"PH vs CNN vs Combined — Random Forest ({N_FOLDS}-fold CV)\n"
    f"CNN: ResNet18 ImageNet → {CNN_DIM}-dim  |  "
    f"PH: ranked persistence vector  |  "
    f"RF: {N_TREES} trees, class_weight=balanced",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig_b.savefig(OUT_BAR, dpi=150, bbox_inches="tight")
ts(f"Saved → {OUT_BAR}")

# ── 7. confusion matrices ─────────────────────────────────────────────────────
ts("Plotting confusion matrices …")

fig_c, axes_c = plt.subplots(1, 3, figsize=(15, 4))
fig_c.patch.set_facecolor("white")
grade_labels = ["G3", "G4", "G5"]

for ax, (name, res) in zip(axes_c, results.items()):
    cm = confusion_matrix(res["all_true"], res["all_pred"], labels=[3,4,5])
    # normalise by true class (recall per class)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(grade_labels); ax.set_yticklabels(grade_labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                   ha="center", va="center", fontsize=9,
                   color="white" if cm_norm[i,j] > 0.5 else "black")

    m = results[name]["means"]
    ax.set_title(f"{name}\nmacro-F1={m['macro_f1']:.3f}  bal_acc={m['bal_acc']:.3f}",
                fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig_c.suptitle("Confusion Matrices (normalised by true class — OOF predictions)",
               fontsize=11, fontweight="bold")
plt.tight_layout()
fig_c.savefig(OUT_CONF, dpi=150, bbox_inches="tight")
ts(f"Saved → {OUT_CONF}")

# ── 8. ROC curves ─────────────────────────────────────────────────────────────
ts("Plotting ROC curves …")

grade_names = {3:"G3", 4:"G4", 5:"G5"}
line_styles = ["-","--","-."]
fig_r, axes_r = plt.subplots(1, 3, figsize=(15, 5))
fig_r.patch.set_facecolor("white")

for ax, (name, res) in zip(axes_r, results.items()):
    y_true_bin = label_binarize(res["all_true"], classes=[3,4,5])
    y_proba    = res["all_proba"]

    for gi, (g, ls) in enumerate(zip([3,4,5], line_styles)):
        fpr, tpr, _ = roc_curve(y_true_bin[:,gi], y_proba[:,gi])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, ls=ls, lw=2,
                label=f"{grade_names[g]}  AUC={roc_auc:.3f}")

    ax.plot([0,1],[0,1],"k--",lw=0.8)
    ax.set_xlabel("FPR", fontsize=9); ax.set_ylabel("TPR", fontsize=9)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title(f"{name}", fontsize=10, fontweight="bold")

fig_r.suptitle("ROC Curves per Gleason grade (OOF predictions, one-vs-rest)",
               fontsize=11, fontweight="bold")
plt.tight_layout()
fig_r.savefig(OUT_ROC, dpi=150, bbox_inches="tight")
ts(f"Saved → {OUT_ROC}")

# ── 9. summary table ──────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Modality':<15} {'Macro-F1':>12} {'Bal-Acc':>12} "
      f"{'F1-G3':>8} {'F1-G4':>8} {'F1-G5':>8}")
print("="*75)
for name, res in results.items():
    m = res["means"]
    s = res["stds"]
    print(f"{name:<15} "
          f"{m['macro_f1']:>6.3f}±{s['macro_f1']:.3f}  "
          f"{m['bal_acc']:>6.3f}±{s['bal_acc']:.3f}  "
          f"{m['f1_G3']:>6.3f}  "
          f"{m['f1_G4']:>6.3f}  "
          f"{m['f1_G5']:>6.3f}")
print("="*75)

# Best modality by macro-F1
best = max(results, key=lambda n: results[n]["means"]["macro_f1"])
print(f"\nBest modality: {best}  "
      f"macro-F1={results[best]['means']['macro_f1']:.3f}")
print(f"\nSaved: {OUT_CSV}, {OUT_BAR}, {OUT_CONF}, {OUT_ROC}")