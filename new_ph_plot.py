"""
plot_compact_figure4.py
───────────────────────
Figure-4-style panel for a selected compact feature method.

Default:
  Base+Life+Betti | top_k=30 | pca=6

Inputs required:
  cache/roi_images.npz
  cache/roi_meta.npz
  cache/compact_feature_sets.npz
  compact_metrics.csv

Output:
  figure4_compact_best.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

K_CLUSTERS = 6
N_BOOTSTRAP = 50
RANDOM_SEED = 7

CACHE_DIR = "cache"
CACHE_IMAGES = os.path.join(CACHE_DIR, "roi_images.npz")
CACHE_META = os.path.join(CACHE_DIR, "roi_meta.npz")
CACHE_FEATURES = os.path.join(CACHE_DIR, "compact_feature_sets.npz")
METRICS_CSV = "compact_metrics.csv"

TARGET_LABEL = "Base+Life+Betti"
TARGET_TOPK = 30
TARGET_PCA = 6

OUT_PATH = "figure4_compact_best.png"


def cluster_fracs(lbls, grades, k=K_CLUSTERS):
    out = np.zeros((k, 3))
    for ci, c in enumerate(range(1, k + 1)):
        mask = lbls == c
        total = mask.sum() + 1e-9
        for gi, g in enumerate([3, 4, 5]):
            out[ci, gi] = (grades[mask] == g).sum() / total
    return out


def reorder_clusters_by_grade_severity(lbls, grades):
    cfrac = cluster_fracs(lbls, grades)
    scores = cfrac @ np.array([3, 4, 5], dtype=float)
    order = np.argsort(scores)
    remap = {old + 1: new + 1 for new, old in enumerate(order)}
    return np.array([remap[l] for l in lbls], dtype=np.int32)


def select_top_features(X, y, top_k):
    F, _ = f_classif(X, y)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(F)[::-1][:top_k]
    idx = np.sort(idx)
    return X[:, idx], idx


def prepare_embedding(X, pca_dim):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if pca_dim is None:
        return Xs, scaler, None
    n_comp = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    Xe = pca.fit_transform(Xs)
    return Xe, scaler, pca


def transform_embedding(X, scaler, pca):
    Xs = scaler.transform(X)
    if pca is None:
        return Xs
    return pca.transform(Xs)


# ── load data ────────────────────────────────────────────────────────────────
img_data = np.load(CACHE_IMAGES)
meta_data = np.load(CACHE_META, allow_pickle=True)
feat_data = np.load(CACHE_FEATURES)

all_imgs = img_data["thumbnails"].astype(np.float32)
grades_all = meta_data["grades"].astype(np.int32)

X_base = feat_data["baseline_toplife"]
X_life = feat_data["life_summary"]
X_betti = feat_data["betti_summary"]
X_morph = feat_data["morph_summary"]

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

X = feature_sets[TARGET_LABEL]

# ── rerun representative bootstrap for chosen method ─────────────────────────
grades_u = np.array(sorted(np.unique(grades_all)))
min_count = min((grades_all == g).sum() for g in grades_u)

rng = np.random.default_rng(RANDOM_SEED)
boot_results = []
all_cfrac = []

for b in range(N_BOOTSTRAP):
    idx_boot = []
    for g in grades_u:
        g_idx = np.where(grades_all == g)[0]
        chosen = rng.choice(g_idx, size=min_count, replace=False)
        idx_boot.extend(chosen.tolist())
    idx_boot = np.array(idx_boot, dtype=np.int32)

    X_b = X[idx_boot]
    y_b = grades_all[idx_boot]

    X_sel, feat_idx = select_top_features(X_b, y_b, TARGET_TOPK)
    Xe, scaler, pca = prepare_embedding(X_sel, TARGET_PCA)

    Z = linkage(Xe, method="ward")
    lbls = fcluster(Z, K_CLUSTERS, criterion="maxclust")
    lbls = reorder_clusters_by_grade_severity(lbls, y_b)

    cf = cluster_fracs(lbls, y_b)
    all_cfrac.append(cf)

    boot_results.append({
        "idx_boot": idx_boot,
        "grades": y_b,
        "clusters": lbls,
        "feat_idx": feat_idx,
        "scaler": scaler,
        "pca": pca,
        "embedding": Xe
    })

all_cfrac = np.stack(all_cfrac)
meta_cent = all_cfrac.mean(axis=0)
dists = np.array([np.linalg.norm(cf - meta_cent) for cf in all_cfrac], dtype=float)
rep_b = int(np.argmin(dists))
rep = boot_results[rep_b]

# ── t-SNE for representative bootstrap ───────────────────────────────────────
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

# ── representative ROIs per (cluster, grade) ────────────────────────────────
# Use selected features from representative bootstrap, then embed all ROIs
X_all_sel = X[:, rep["feat_idx"]]
X_all_emb = transform_embedding(X_all_sel, rep["scaler"], rep["pca"])

n_use = X_all_emb.shape[1]
centroids = np.zeros((K_CLUSTERS, n_use), dtype=float)
for c in range(1, K_CLUSTERS + 1):
    mask = rep["clusters"] == c
    centroids[c - 1] = rep["embedding"][mask].mean(axis=0)

rep_rois = {}
for c in range(1, K_CLUSTERS + 1):
    for g in [3, 4, 5]:
        g_idx = np.where(grades_all == g)[0]
        if len(g_idx) == 0:
            rep_rois[(c, g)] = None
            continue
        d = np.linalg.norm(X_all_emb[g_idx] - centroids[c - 1], axis=1)
        best = g_idx[np.argmin(d)]
        rep_rois[(c, g)] = all_imgs[best]

# ── bootstrap std for cluster bars ───────────────────────────────────────────
rng2 = np.random.default_rng(42)
boot_cf = []
for _ in range(200):
    idx = rng2.integers(0, len(rep["grades"]), size=len(rep["grades"]))
    boot_cf.append(cluster_fracs(rep["clusters"][idx], rep["grades"][idx]))
cfrac_rep = cluster_fracs(rep["clusters"], rep["grades"])
cfrac_std = np.std(boot_cf, axis=0)

# ── get metric row for title ─────────────────────────────────────────────────
df = pd.read_csv(METRICS_CSV)
row = df[
    (df["label"] == TARGET_LABEL) &
    (df["top_k"] == TARGET_TOPK) &
    (df["pca_dim"] == TARGET_PCA)
].iloc[0]

# ── plot ─────────────────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3", "#a65628"]
GRADE_MARKERS  = {3: "o", 4: "^", 5: "s"}
GRADE_FILL     = {3: "#2c3e50", 4: "#7f8c8d", 5: "#bdc3c7"}
roman          = ["i", "ii", "iii", "iv", "v", "vi"]

fig = plt.figure(figsize=(22, 11))
fig.patch.set_facecolor("white")
gs = GridSpec(
    5, 8, figure=fig,
    height_ratios=[0.15, 1, 1, 1, 0.8],
    width_ratios=[1.8, 0.15, 1, 1, 1, 1, 1, 1],
    wspace=0.08, hspace=0.12,
    left=0.03, right=0.98, top=0.92, bottom=0.04
)

# t-SNE
ax_tsne = fig.add_subplot(gs[0:5, 0])
for c in range(1, K_CLUSTERS + 1):
    col = CLUSTER_COLORS[c - 1]
    for g in [3, 4, 5]:
        mask = (rep["clusters"] == c) & (rep["grades"] == g)
        if mask.sum() == 0:
            continue
        ax_tsne.scatter(
            Xtsne[mask, 0], Xtsne[mask, 1],
            c=col, marker=GRADE_MARKERS[g],
            s=30, alpha=0.85, linewidths=0.3,
            edgecolors="white", zorder=3
        )

for c in range(1, K_CLUSTERS + 1):
    mask = rep["clusters"] == c
    mx, my = Xtsne[mask, 0].mean(), Xtsne[mask, 1].mean()
    ax_tsne.text(
        mx, my, roman[c - 1],
        fontsize=11, fontweight="bold",
        ha="center", va="center",
        color=CLUSTER_COLORS[c - 1],
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        zorder=5
    )

ax_tsne.set_xticks([])
ax_tsne.set_yticks([])
ax_tsne.spines[["top", "right", "bottom", "left"]].set_visible(False)

grade_legend = [
    plt.Line2D([0], [0], marker="o", color="grey", ls="", ms=7, label="G3"),
    plt.Line2D([0], [0], marker="^", color="grey", ls="", ms=7, label="G4"),
    plt.Line2D([0], [0], marker="s", color="grey", ls="", ms=7, label="G5"),
]
cluster_legend = [
    plt.Line2D([0], [0], marker="o", color=CLUSTER_COLORS[c - 1], ls="", ms=7, label=roman[c - 1])
    for c in range(1, K_CLUSTERS + 1)
]
leg1 = ax_tsne.legend(handles=grade_legend, fontsize=7, loc="lower left",
                      framealpha=0.8, edgecolor="grey",
                      title="Grade", title_fontsize=7)
ax_tsne.add_artist(leg1)
ax_tsne.legend(handles=cluster_legend, fontsize=7, loc="upper right",
               framealpha=0.8, edgecolor="grey",
               title="Cluster", title_fontsize=7, ncol=2)
ax_tsne.set_title(f"t-SNE\n{TARGET_LABEL}", fontsize=10, fontweight="bold", pad=3)

# headers
for ci, c in enumerate(range(1, K_CLUSTERS + 1)):
    ax_h = fig.add_subplot(gs[0, ci + 2])
    ax_h.axis("off")
    ax_h.text(0.5, 0.3, roman[c - 1], transform=ax_h.transAxes,
              fontsize=14, fontweight="bold", ha="center", va="center",
              color=CLUSTER_COLORS[c - 1])

# ROI grid
for ri, g in enumerate([3, 4, 5]):
    for ci, c in enumerate(range(1, K_CLUSTERS + 1)):
        ax = fig.add_subplot(gs[ri + 1, ci + 2])
        img = rep_rois.get((c, g))
        if img is not None:
            ax.imshow(np.clip(img, 0, 1))
            for sp in ax.spines.values():
                sp.set_edgecolor(CLUSTER_COLORS[c - 1])
                sp.set_linewidth(2.5)
                sp.set_visible(True)
        else:
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, color="#aaaaaa", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

# row labels
for ri, g in enumerate([3, 4, 5]):
    ax_gl = fig.add_subplot(gs[ri + 1, 1])
    ax_gl.axis("off")
    ax_gl.text(0.5, 0.5, f"G{g}", transform=ax_gl.transAxes,
               fontsize=9, va="center", ha="center",
               color="#555555", style="italic", rotation=90)

# histograms
x = np.arange(3)
w = 0.55
for ci, c in enumerate(range(1, K_CLUSTERS + 1)):
    ax = fig.add_subplot(gs[4, ci + 2])
    ax.bar(
        x, cfrac_rep[c - 1], w,
        color=[GRADE_FILL[g] for g in [3, 4, 5]],
        yerr=cfrac_std[c - 1], capsize=2,
        error_kw={"elinewidth": 0.8, "ecolor": "#333"}
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["G3", "G4", "G5"], fontsize=6)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="y", labelsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    if ci == 0:
        ax.set_ylabel("Fraction", fontsize=7)
    else:
        ax.set_yticklabels([])
    n_c = (rep["clusters"] == c).sum()
    ax.text(0.5, 1.05, f"n={n_c}", transform=ax.transAxes,
            fontsize=6, ha="center", va="bottom", color="#444")

grade_counts = {g: int((grades_all == g).sum()) for g in [3, 4, 5]}
fig.suptitle(
    f"Best compact variant: {TARGET_LABEL}  |  silhouette={row['silhouette_mean']:.3f}  "
    f"G5-purity={row['g5_purity_mean']:.3f}  stability={row['stability']:.3f}\n"
    f"(559 ROIs  |  G3={grade_counts[3]}  G4={grade_counts[4]}  G5={grade_counts[5]}  |  "
    f"k={K_CLUSTERS} clusters  |  {N_BOOTSTRAP} bootstraps  |  top_k={TARGET_TOPK}  pca={TARGET_PCA})",
    fontsize=12, fontweight="bold", y=0.97
)

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved figure -> {OUT_PATH}")
print("\nRepresentative bootstrap cluster composition:")
print(f"{'Cluster':<8} {'G3':>6} {'G4':>6} {'G5':>6}")
for c in range(1, K_CLUSTERS + 1):
    frac = cfrac_rep[c - 1]
    print(f"{roman[c-1]:<8} {frac[0]:>6.3f} {frac[1]:>6.3f} {frac[2]:>6.3f}")