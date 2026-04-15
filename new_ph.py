"""
replacement_compact_features.py
───────────────────────────────
Compact replacement for novel_features.py.

Goal:
- Keep the working baseline PH backbone.
- Add only compact, biologically-grounded summaries.
- Use supervised feature selection before clustering.
- Evaluate fairly under the same bootstrap-balanced Ward clustering.

Inputs expected from existing cache:
  cache/roi_images.npz   -> thumbnails (N,H,W,3), float32 in [0,1]
  cache/roi_meta.npz     -> grades

Outputs:
  compact_metrics.csv
  compact_metrics.png
  compact_tsne_best.png

Optional:
  --force   recompute feature cache even if it exists

Author: ChatGPT
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage import binary_fill_holes, label as ndi_label
from skimage import color, filters, measure
from skimage.transform import resize as sk_resize
import gudhi

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
CACHE_DIR = "cache"
OUT_DIR   = "."

CACHE_IMAGES = os.path.join(CACHE_DIR, "roi_images.npz")
CACHE_META   = os.path.join(CACHE_DIR, "roi_meta.npz")

FEATURE_CACHE = os.path.join(CACHE_DIR, "compact_feature_sets.npz")

FORCE = "--force" in sys.argv[1:]

K_CLUSTERS   = 6
N_BOOTSTRAP  = 50
RANDOM_SEED  = 7

# Core PH settings
BASELINE_GRID_SIZE = 64   # keep same spirit as working baseline
BETTI_GRID_SIZE    = 64   # same grid used for compact Betti summaries
TOPK_H0            = 20
TOPK_H1            = 20
BETTI_STEPS        = 16

# Feature selection / compression search
TOPK_FEATURE_OPTIONS = [10, 20, 30, 40]
PCA_OPTIONS          = [None, 6, 8]

# Morphology
MIN_COMPONENT_AREA = 8  # filter tiny junk after thresholding


def ts(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Basic image / PH helpers
# ──────────────────────────────────────────────────────────────────────────────
def hema_from_thumb(thumb_f32: np.ndarray) -> np.ndarray:
    """
    Convert RGB thumbnail float32 [0,1] -> hematoxylin channel [0,1].
    """
    img_u8 = (np.clip(thumb_f32, 0, 1) * 255).astype(np.uint8)
    hed = color.rgb2hed(img_u8)
    h = hed[:, :, 0]
    hmin = np.percentile(h, 1)
    hmax = np.percentile(h, 99)
    return np.clip((h - hmin) / (hmax - hmin + 1e-9), 0, 1).astype(np.float64)


def normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def ph_on_grid(grid_f64: np.ndarray):
    """
    Run cubical PH on a 2D float array.
    Returns finite H0 and H1 diagrams as arrays of shape (n,2).
    """
    cc = gudhi.CubicalComplex(top_dimensional_cells=grid_f64)
    cc.compute_persistence()

    def get_fin(dim: int) -> np.ndarray:
        arr = np.array(cc.persistence_intervals_in_dimension(dim), dtype=float)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=float)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size == 0:
            return np.zeros((0, 2), dtype=float)
        return arr

    return get_fin(0), get_fin(1)


def topk_lifetimes(dgm: np.ndarray, k: int) -> np.ndarray:
    """
    Sorted descending top-k lifetimes, zero-padded.
    """
    if len(dgm) == 0:
        return np.zeros(k, dtype=np.float32)
    life = np.sort(dgm[:, 1] - dgm[:, 0])[::-1]
    out = np.zeros(k, dtype=np.float32)
    m = min(k, len(life))
    out[:m] = life[:m]
    return out


def safe_summary_stats(life: np.ndarray, thresholds=(0.02, 0.05, 0.10, 0.20)) -> np.ndarray:
    """
    Compact summary features for one diagram's lifetimes.
    """
    if life.size == 0:
        return np.zeros(9 + len(thresholds), dtype=np.float32)

    q25, q50, q75 = np.percentile(life, [25, 50, 75])
    feats = [
        float(life.mean()),
        float(life.std()),
        float(np.median(life)),
        float(life.max()),
        float(life.sum()),
        float(q25),
        float(q50),
        float(q75),
        float(len(life)),
    ]
    feats.extend([float((life > t).sum()) for t in thresholds])
    return np.array(feats, dtype=np.float32)


def lifetime_summary(dgm: np.ndarray) -> np.ndarray:
    if len(dgm) == 0:
        return np.zeros(13, dtype=np.float32)
    life = dgm[:, 1] - dgm[:, 0]
    return safe_summary_stats(life)


def betti_numbers_from_mask(mask: np.ndarray) -> tuple[int, int]:
    """
    Approximate Betti-0 and Betti-1 from a binary mask.
    For 2D binary masks:
      B0 = connected components
      B1 = holes = B0 - Euler
    """
    labeled = measure.label(mask, connectivity=1)
    b0 = int(labeled.max())
    euler = int(measure.euler_number(mask, connectivity=1))
    b1 = max(0, b0 - euler)
    return b0, b1


def betti_curve_features(grid: np.ndarray, n_steps: int = BETTI_STEPS) -> np.ndarray:
    """
    Compact Betti-curve summaries from sublevel sets of the intensity grid.
    Output:
      H0 curve (n_steps)
      H1 curve (n_steps)
      summary stats (8)
    """
    vals = np.linspace(float(grid.min()), float(grid.max()), n_steps)
    b0_curve, b1_curve = [], []

    for v in vals:
        mask = (grid <= v).astype(np.uint8)
        b0, b1 = betti_numbers_from_mask(mask)
        b0_curve.append(b0)
        b1_curve.append(b1)

    b0_curve = np.array(b0_curve, dtype=np.float32)
    b1_curve = np.array(b1_curve, dtype=np.float32)

    extra = np.array([
        float(b0_curve.mean()),
        float(b0_curve.max()),
        float(np.argmax(b0_curve)),
        float(b0_curve.sum()),
        float(b1_curve.mean()),
        float(b1_curve.max()),
        float(np.argmax(b1_curve)),
        float(b1_curve.sum()),
    ], dtype=np.float32)

    return np.concatenate([b0_curve, b1_curve, extra]).astype(np.float32)


def morphology_features(hema: np.ndarray) -> np.ndarray:
    """
    Small, interpretable morphology block.
    """
    thr = filters.threshold_otsu(hema)
    fg = (hema > thr).astype(np.uint8)

    # remove tiny junk
    lbl, n = ndi_label(fg)
    cleaned = np.zeros_like(fg, dtype=np.uint8)
    if n > 0:
        props = measure.regionprops(lbl)
        kept_labels = [p.label for p in props if p.area >= MIN_COMPONENT_AREA]
        if kept_labels:
            cleaned[np.isin(lbl, kept_labels)] = 1

    fg = cleaned
    filled = binary_fill_holes(fg).astype(np.uint8)

    lbl2 = measure.label(fg, connectivity=1)
    props2 = measure.regionprops(lbl2)

    if len(props2) == 0:
        comp_areas = np.array([0.0], dtype=np.float32)
        perims = np.array([0.0], dtype=np.float32)
        eccs = np.array([0.0], dtype=np.float32)
    else:
        comp_areas = np.array([p.area for p in props2], dtype=np.float32)
        perims = np.array([p.perimeter for p in props2], dtype=np.float32)
        eccs = np.array([p.eccentricity for p in props2], dtype=np.float32)

    hole_pixels = float(filled.sum() - fg.sum())
    euler = float(measure.euler_number(fg, connectivity=1))
    fg_frac = float(fg.mean())

    feats = np.array([
        fg_frac,
        float(len(props2)),
        float(comp_areas.mean()),
        float(np.median(comp_areas)),
        float(comp_areas.std()),
        float(comp_areas.max()),
        float(perims.mean()),
        float(eccs.mean()),
        hole_pixels,
        euler,
    ], dtype=np.float32)

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────
def build_feature_blocks(all_imgs: np.ndarray):
    """
    Build compact feature families per ROI:
      1) baseline top lifetimes
      2) lifetime summaries
      3) Betti summaries
      4) morphology summaries
    """
    N = len(all_imgs)

    hema_all = []
    X_baseline = []
    X_life_summary = []
    X_betti = []
    X_morph = []

    ts("Precomputing hematoxylin channels and compact features ...")
    for i in range(N):
        hema = hema_from_thumb(all_imgs[i])
        hema_all.append(hema)

        grid_base = normalise(
            sk_resize(hema, (BASELINE_GRID_SIZE, BASELINE_GRID_SIZE), anti_aliasing=True)
        ).astype(np.float64)

        dgm0, dgm1 = ph_on_grid(grid_base)

        x_base = np.concatenate([
            topk_lifetimes(dgm0, TOPK_H0),
            topk_lifetimes(dgm1, TOPK_H1),
        ]).astype(np.float32)

        x_life = np.concatenate([
            lifetime_summary(dgm0),
            lifetime_summary(dgm1),
            np.array([
                float((dgm1[:, 1] - dgm1[:, 0]).sum() / ((dgm0[:, 1] - dgm0[:, 0]).sum() + 1e-9))
            ], dtype=np.float32) if len(dgm0) > 0 or len(dgm1) > 0 else np.array([0.0], dtype=np.float32)
        ]).astype(np.float32)

        grid_betti = normalise(
            sk_resize(hema, (BETTI_GRID_SIZE, BETTI_GRID_SIZE), anti_aliasing=True)
        ).astype(np.float64)
        x_betti = betti_curve_features(grid_betti, BETTI_STEPS)

        x_morph = morphology_features(hema)

        X_baseline.append(x_base)
        X_life_summary.append(x_life)
        X_betti.append(x_betti)
        X_morph.append(x_morph)

        if (i + 1) % 50 == 0 or (i + 1) == N:
            ts(f"  {i + 1}/{N}")

    return {
        "hema_all": np.stack(hema_all),
        "baseline_toplife": np.stack(X_baseline).astype(np.float32),
        "life_summary": np.stack(X_life_summary).astype(np.float32),
        "betti_summary": np.stack(X_betti).astype(np.float32),
        "morph_summary": np.stack(X_morph).astype(np.float32),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Clustering / evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
def cluster_fracs(lbls: np.ndarray, grades: np.ndarray, k: int = K_CLUSTERS) -> np.ndarray:
    out = np.zeros((k, 3), dtype=np.float64)
    for ci, c in enumerate(range(1, k + 1)):
        mask = lbls == c
        total = mask.sum() + 1e-9
        for gi, g in enumerate([3, 4, 5]):
            out[ci, gi] = (grades[mask] == g).sum() / total
    return out


def agglomerative_coeff(Z: np.ndarray) -> float:
    """
    Simple version consistent with your previous script:
      AC = 1 - mean(merge heights)/max height
    """
    heights = Z[:, 2]
    max_h = heights.max() + 1e-9
    return float(1.0 - heights.mean() / max_h)


def composition_stability(boot_cfracs: list[np.ndarray]) -> float:
    """
    Stability = 1 / (1 + average deviation from meta-centroid)
    Higher = more stable.
    """
    arr = np.stack(boot_cfracs)
    meta = arr.mean(axis=0)
    dists = np.array([np.linalg.norm(x - meta) for x in arr], dtype=float)
    return float(1.0 / (1.0 + dists.mean()))


def reorder_clusters_by_grade_severity(lbls: np.ndarray, grades: np.ndarray) -> np.ndarray:
    cfrac = cluster_fracs(lbls, grades)
    scores = cfrac @ np.array([3, 4, 5], dtype=float)
    order = np.argsort(scores)
    remap = {old + 1: new + 1 for new, old in enumerate(order)}
    return np.array([remap[l] for l in lbls], dtype=np.int32)


def score_features_fclassif(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    F, _ = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


def score_features_mi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=RANDOM_SEED)
    mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
    return mi


def select_top_features(X: np.ndarray, y: np.ndarray, top_k: int, method: str = "fscore") -> tuple[np.ndarray, np.ndarray]:
    if top_k is None or top_k >= X.shape[1]:
        idx = np.arange(X.shape[1])
        return X, idx

    if method == "mi":
        scores = score_features_mi(X, y)
    else:
        scores = score_features_fclassif(X, y)

    idx = np.argsort(scores)[::-1][:top_k]
    idx = np.sort(idx)
    return X[:, idx], idx


def prepare_embedding(X_train: np.ndarray, pca_dim: int | None):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    if pca_dim is None:
        return Xs, scaler, None

    n_comp = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
    if n_comp < 1:
        return Xs, scaler, None

    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    Xe = pca.fit_transform(Xs)
    return Xe, scaler, pca


def transform_embedding(X: np.ndarray, scaler: StandardScaler, pca: PCA | None):
    Xs = scaler.transform(X)
    if pca is None:
        return Xs
    return pca.transform(Xs)


def run_bootstrap_pipeline(
    X: np.ndarray,
    grades: np.ndarray,
    label: str,
    top_k: int | None,
    pca_dim: int | None,
    selection_method: str = "fscore",
):
    """
    Balanced bootstrap by grade -> feature selection -> scaling -> optional PCA -> Ward clustering
    Returns metrics and representative bootstrap for plotting.
    """
    ts(f"Pipeline: {label} | top_k={top_k} | pca={pca_dim}")

    grades_u = np.array(sorted(np.unique(grades)))
    min_count = min((grades == g).sum() for g in grades_u)

    rng = np.random.default_rng(RANDOM_SEED)

    silhouettes = []
    acs = []
    g5_purities = []
    cfrac_list = []
    boot_results = []

    for b in range(N_BOOTSTRAP):
        idx_boot = []
        for g in grades_u:
            g_idx = np.where(grades == g)[0]
            chosen = rng.choice(g_idx, size=min_count, replace=False)
            idx_boot.extend(chosen.tolist())
        idx_boot = np.array(idx_boot, dtype=np.int32)

        X_b = X[idx_boot]
        y_b = grades[idx_boot]

        X_sel, feat_idx = select_top_features(X_b, y_b, top_k, method=selection_method)
        Xe, scaler, pca = prepare_embedding(X_sel, pca_dim)

        Z = linkage(Xe, method="ward")
        lbls = fcluster(Z, K_CLUSTERS, criterion="maxclust")
        lbls = reorder_clusters_by_grade_severity(lbls, y_b)

        sil = silhouette_score(Xe, lbls) if len(np.unique(lbls)) > 1 else 0.0
        ac = agglomerative_coeff(Z)
        cfrac = cluster_fracs(lbls, y_b)
        g5p = float(cfrac[:, 2].max())

        silhouettes.append(float(sil))
        acs.append(float(ac))
        g5_purities.append(g5p)
        cfrac_list.append(cfrac)

        boot_results.append({
            "idx_boot": idx_boot,
            "grades": y_b,
            "clusters": lbls,
            "feat_idx": feat_idx,
            "scaler": scaler,
            "pca": pca,
            "embedding": Xe,
        })

    stability = composition_stability(cfrac_list)

    # Representative bootstrap
    all_cfrac = np.stack(cfrac_list)
    meta_cent = all_cfrac.mean(axis=0)
    dists = np.array([np.linalg.norm(cf - meta_cent) for cf in all_cfrac], dtype=float)
    rep_b = int(np.argmin(dists))
    rep = boot_results[rep_b]

    # t-SNE on representative bootstrap
    perp = min(40, max(5, len(rep["embedding"]) // 4))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=RANDOM_SEED,
        max_iter=2000,
        learning_rate="auto",
        init="pca",
    )
    Xtsne = tsne.fit_transform(rep["embedding"])

    composite = 0.35 * np.mean(silhouettes) + 0.45 * np.mean(g5_purities) + 0.20 * stability

    return {
        "label": label,
        "top_k": top_k,
        "pca_dim": pca_dim,
        "selection_method": selection_method,

        "silhouette_mean": float(np.mean(silhouettes)),
        "silhouette_std": float(np.std(silhouettes)),
        "ac_mean": float(np.mean(acs)),
        "ac_std": float(np.std(acs)),
        "g5_purity_mean": float(np.mean(g5_purities)),
        "g5_purity_std": float(np.std(g5_purities)),
        "stability": float(stability),
        "composite_score": float(composite),

        "rep_boot_idx": rep_b,
        "rep_embedding": rep["embedding"],
        "rep_tsne": Xtsne,
        "rep_clusters": rep["clusters"],
        "rep_grades": rep["grades"],
        "rep_global_idx": rep["idx_boot"],
        "rep_feat_idx": rep["feat_idx"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────
def save_metrics_barplot(df: pd.DataFrame, out_path: str):
    metrics = [
        ("silhouette_mean", "Silhouette"),
        ("ac_mean", "Agglomerative Coefficient"),
        ("g5_purity_mean", "G5 Purity"),
        ("stability", "Stability"),
        ("composite_score", "Composite Score"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(24, 5))
    fig.patch.set_facecolor("white")

    names = df["experiment"].tolist()
    x = np.arange(len(names))

    for ax, (col, title) in zip(axes, metrics):
        vals = df[col].values
        ax.bar(x, vals)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ymax = max(vals) * 1.25 + 1e-6
        ax.set_ylim(0, ymax if ymax > 0 else 1)
        for xi, v in zip(x, vals):
            ax.text(xi, v + ymax * 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    fig.suptitle("Compact PH redesign: experiment comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_best_tsne(best_result: dict, out_path: str):
    CLUSTER_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]
    GRADE_MARKERS = {3: "o", 4: "^", 5: "s"}
    roman = ["i", "ii", "iii", "iv", "v", "vi"]

    Xtsne = best_result["rep_tsne"]
    clust = best_result["rep_clusters"]
    grades = best_result["rep_grades"]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")

    for c in range(1, K_CLUSTERS + 1):
        col = CLUSTER_COLORS[c - 1]
        for g in [3, 4, 5]:
            mask = (clust == c) & (grades == g)
            if mask.sum() == 0:
                continue
            ax.scatter(
                Xtsne[mask, 0], Xtsne[mask, 1],
                c=col, marker=GRADE_MARKERS[g],
                s=42, alpha=0.85, linewidths=0.3, edgecolors="white"
            )

    for c in range(1, K_CLUSTERS + 1):
        mask = clust == c
        if mask.sum() == 0:
            continue
        mx, my = Xtsne[mask, 0].mean(), Xtsne[mask, 1].mean()
        ax.text(
            mx, my, roman[c - 1],
            fontsize=11, fontweight="bold", ha="center", va="center",
            color=CLUSTER_COLORS[c - 1],
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
        )

    grade_legend = [
        plt.Line2D([0], [0], marker="o", color="grey", ls="", ms=7, label="G3"),
        plt.Line2D([0], [0], marker="^", color="grey", ls="", ms=7, label="G4"),
        plt.Line2D([0], [0], marker="s", color="grey", ls="", ms=7, label="G5"),
    ]
    cluster_legend = [
        plt.Line2D([0], [0], marker="o", color=CLUSTER_COLORS[c - 1], ls="", ms=7, label=roman[c - 1])
        for c in range(1, K_CLUSTERS + 1)
    ]
    leg1 = ax.legend(handles=grade_legend, fontsize=8, loc="lower left", framealpha=0.85, title="Grade")
    ax.add_artist(leg1)
    ax.legend(handles=cluster_legend, fontsize=8, loc="upper right", framealpha=0.85, title="Cluster", ncol=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.set_title(
        f"Best compact pipeline\n"
        f"{best_result['label']} | top_k={best_result['top_k']} | pca={best_result['pca_dim']}\n"
        f"sil={best_result['silhouette_mean']:.3f}  "
        f"AC={best_result['ac_mean']:.3f}  "
        f"G5={best_result['g5_purity_mean']:.3f}  "
        f"stab={best_result['stability']:.3f}",
        fontsize=10,
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ts("Loading cached ROI images ...")
    if not os.path.exists(CACHE_IMAGES) or not os.path.exists(CACHE_META):
        raise FileNotFoundError(
            "Missing cache/roi_images.npz or cache/roi_meta.npz. "
            "Run your ROI extraction / figure4 cache pipeline first."
        )

    img_data = np.load(CACHE_IMAGES)
    meta_data = np.load(CACHE_META, allow_pickle=True)

    all_imgs = img_data["thumbnails"].astype(np.float32)
    grades_all = meta_data["grades"].astype(np.int32)

    ts(f"Loaded {len(all_imgs)} ROIs")
    ts(f"Grade counts: { {g: int((grades_all == g).sum()) for g in [3, 4, 5]} }")

    # Build or load compact feature families
    if (not FORCE) and os.path.exists(FEATURE_CACHE):
        ts("Loading compact feature cache ...")
        fc = np.load(FEATURE_CACHE)
        feat_blocks = {
            "baseline_toplife": fc["baseline_toplife"],
            "life_summary": fc["life_summary"],
            "betti_summary": fc["betti_summary"],
            "morph_summary": fc["morph_summary"],
        }
    else:
        feat_blocks = build_feature_blocks(all_imgs)
        np.savez_compressed(
            FEATURE_CACHE,
            baseline_toplife=feat_blocks["baseline_toplife"],
            life_summary=feat_blocks["life_summary"],
            betti_summary=feat_blocks["betti_summary"],
            morph_summary=feat_blocks["morph_summary"],
        )
        ts(f"Saved feature cache -> {FEATURE_CACHE}")

    X_base = feat_blocks["baseline_toplife"]
    X_life = feat_blocks["life_summary"]
    X_betti = feat_blocks["betti_summary"]
    X_morph = feat_blocks["morph_summary"]

    # Compact experiment families
    feature_sets = {
        "BaselineTopLife": X_base,
        "Base+Life": np.concatenate([X_base, X_life], axis=1),
        "Base+Betti": np.concatenate([X_base, X_betti], axis=1),
        "Base+Morph": np.concatenate([X_base, X_morph], axis=1),
        "Base+Betti+Morph": np.concatenate([X_base, X_betti, X_morph], axis=1),
        "Base+Life+Betti": np.concatenate([X_base, X_life, X_betti], axis=1),
        "Base+Life+Morph": np.concatenate([X_base, X_life, X_morph], axis=1),
        "Base+Life+Betti+Morph": np.concatenate([X_base, X_life, X_betti, X_morph], axis=1),
    }

    ts("Feature matrix sizes:")
    for name, X in feature_sets.items():
        ts(f"  {name:<24} {X.shape}")

    # Run experiment grid
    results = []
    for feat_name, X in feature_sets.items():
        for top_k in TOPK_FEATURE_OPTIONS:
            for pca_dim in PCA_OPTIONS:
                res = run_bootstrap_pipeline(
                    X=X,
                    grades=grades_all,
                    label=feat_name,
                    top_k=top_k,
                    pca_dim=pca_dim,
                    selection_method="fscore",
                )
                results.append(res)

    df = pd.DataFrame(results)
    df["experiment"] = df.apply(
        lambda r: f"{r['label']} | k={r['top_k']} | pca={r['pca_dim']}",
        axis=1
    )
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(OUT_DIR, "compact_metrics.csv")
    df.to_csv(csv_path, index=False)
    ts(f"Saved metrics -> {csv_path}")

    barplot_path = os.path.join(OUT_DIR, "compact_metrics.png")
    save_metrics_barplot(df.head(15), barplot_path)
    ts(f"Saved metrics plot -> {barplot_path}")

    # Best result
    best_result = results[int(df.index[0])]
    # safer: map back by exact experiment row
    best_row = df.iloc[0]
    for r in results:
        if (
            r["label"] == best_row["label"] and
            r["top_k"] == best_row["top_k"] and
            str(r["pca_dim"]) == str(best_row["pca_dim"])
        ):
            best_result = r
            break

    tsne_path = os.path.join(OUT_DIR, "compact_tsne_best.png")
    save_best_tsne(best_result, tsne_path)
    ts(f"Saved best t-SNE -> {tsne_path}")

    print("\n" + "=" * 112)
    print(
        f"{'Rank':<5} {'Experiment':<48} {'Silhouette':>12} {'AC':>10} "
        f"{'G5-Purity':>12} {'Stability':>12} {'Composite':>12}"
    )
    print("=" * 112)

    for i, (_, row) in enumerate(df.head(20).iterrows(), start=1):
        print(
            f"{i:<5} "
            f"{row['experiment'][:48]:<48} "
            f"{row['silhouette_mean']:>6.3f}±{row['silhouette_std']:.3f}  "
            f"{row['ac_mean']:>5.3f}±{row['ac_std']:.3f}  "
            f"{row['g5_purity_mean']:>6.3f}±{row['g5_purity_std']:.3f}  "
            f"{row['stability']:>10.3f}  "
            f"{row['composite_score']:>10.3f}"
        )

    print("=" * 112)
    print("\nBest experiment:")
    print(f"  {best_row['experiment']}")
    print(f"  Silhouette : {best_row['silhouette_mean']:.4f} ± {best_row['silhouette_std']:.4f}")
    print(f"  AC         : {best_row['ac_mean']:.4f} ± {best_row['ac_std']:.4f}")
    print(f"  G5 Purity  : {best_row['g5_purity_mean']:.4f} ± {best_row['g5_purity_std']:.4f}")
    print(f"  Stability  : {best_row['stability']:.4f}")
    print(f"  Composite  : {best_row['composite_score']:.4f}")

    print("\nSaved files:")
    print(f"  {csv_path}")
    print(f"  {barplot_path}")
    print(f"  {tsne_path}")


if __name__ == "__main__":
    main()