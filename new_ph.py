"""
novel_features.py
─────────────────
Computes three augmented feature matrices and compares them against the
baseline (paper's method) using silhouette score, agglomerative coefficient,
and grade purity.

Feature matrices built from the same 559 ROIs:
  X_baseline   — intensity filtration only, 64×64  (paper replication)
  X_spatial    — intensity + 4 directional filtrations at 0°,45°,90°,135°, 128×128
  X_multiscale — intensity filtration at 3 scales: 32×32, 64×64, 128×128
  X_combined   — spatial (all directions) + multiscale (all scales)

Pipeline per variant:
  PCA (6 components) → bootstrap Ward clustering (k=6, 50 bootstraps)
  → representative bootstrap → t-SNE → Figure-4-style plot + metrics

Run:
  python novel_features.py [--force]
  --force  recompute all feature matrices even if cached
"""

import os, sys, json, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from skimage import color, filters
from skimage.transform import resize as sk_resize
import gudhi
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR     = "cache"
OUT_DIR       = "."

# Cache files from baseline pipeline (must exist)
CACHE_IMAGES  = os.path.join(CACHE_DIR, "roi_images.npz")
CACHE_META    = os.path.join(CACHE_DIR, "roi_meta.npz")

# New cache files for novel features
CACHE_SPATIAL     = os.path.join(CACHE_DIR, "ph_spatial.npz")
CACHE_MULTISCALE  = os.path.join(CACHE_DIR, "ph_multiscale.npz")
CACHE_COMBINED    = os.path.join(CACHE_DIR, "ph_combined.npz")
CACHE_BASELINE_V2 = os.path.join(CACHE_DIR, "ph_vectors.npz")  # already exists

K_CLUSTERS   = 6
N_BOOTSTRAP  = 50
N_PCA        = 6
DIRECTIONS   = [0, 45, 90, 135]   # degrees for PHT
SCALES       = [32, 64, 128]       # pixels for multiscale

FORCE = "--force" in sys.argv[1:]

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def hema_from_thumb(thumb_f32):
    """thumb_f32: (H,W,3) float32 [0,1] → hematoxylin [0,1] float64"""
    img_u8 = (thumb_f32 * 255).astype(np.uint8)
    hed    = color.rgb2hed(img_u8)
    h      = hed[:, :, 0]
    hmin   = np.percentile(h, 1)
    hmax   = np.percentile(h, 99)
    return np.clip((h-hmin)/(hmax-hmin+1e-9), 0, 1).astype(np.float64)

def normalise(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

def ph_on_grid(grid_f64):
    """Run gudhi cubical PH on a 2D float64 array. Returns (dgm0, dgm1)."""
    cc = gudhi.CubicalComplex(top_dimensional_cells=grid_f64)
    cc.compute_persistence()
    def get_fin(dim):
        arr = np.array(cc.persistence_intervals_in_dimension(dim), dtype=float)
        if arr.size == 0: return np.zeros((0, 2))
        return arr[np.isfinite(arr).all(axis=1)]
    return get_fin(0), get_fin(1)

def ranked_vec(dgm0, dgm1):
    def sp(d):
        if len(d) == 0: return np.array([])
        return np.sort(d[:,1] - d[:,0])[::-1]
    return np.concatenate([sp(dgm0), sp(dgm1)])

# ── directional filtration ────────────────────────────────────────────────────
def directional_filtration(hema_norm, angle_deg, size=128):
    """
    Project pixel positions onto direction θ.
    f_θ(x,y) = x·cos(θ) + y·sin(θ)
    Normalise to [0,1] and run PH.
    hema_norm: already at target size.
    """
    H, W   = hema_norm.shape
    θ      = np.deg2rad(angle_deg)
    ys, xs = np.mgrid[0:H, 0:W]
    # projection value for each pixel
    proj   = xs * np.cos(θ) + ys * np.sin(θ)
    proj_n = normalise(proj).astype(np.float64)
    # Weight by hematoxylin so stroma (background) is suppressed:
    # pixels with low hematoxylin contribute at high filtration value (enter late)
    # This focuses the topology on nuclear structure in each direction
    weighted = normalise(proj_n * hema_norm).astype(np.float64)
    return ph_on_grid(weighted)

# ── feature extraction functions ──────────────────────────────────────────────
def extract_spatial(hema_128):
    """
    Intensity filtration + 4 directional filtrations at 128×128.
    Returns concatenated ranked persistence vector.
    """
    parts = []
    # Intensity (superlevel → pass as-is for sublevel by using 1-hema)
    dgm0, dgm1 = ph_on_grid(hema_128)
    parts.append(ranked_vec(dgm0, dgm1))
    # Directional
    for angle in DIRECTIONS:
        dgm0, dgm1 = directional_filtration(hema_128, angle, size=128)
        parts.append(ranked_vec(dgm0, dgm1))
    return np.concatenate(parts)

def extract_multiscale(hema_128):
    """
    Intensity filtration at 3 scales (32, 64, 128).
    Returns concatenated ranked persistence vector.
    """
    parts = []
    for size in SCALES:
        if size == 128:
            grid = hema_128
        else:
            grid = sk_resize(hema_128, (size, size), anti_aliasing=True)
        grid = normalise(grid).astype(np.float64)
        dgm0, dgm1 = ph_on_grid(grid)
        parts.append(ranked_vec(dgm0, dgm1))
    return np.concatenate(parts)

def extract_combined(hema_128):
    """All directions × all scales."""
    parts = []
    for size in SCALES:
        if size == 128:
            grid = hema_128
        else:
            grid = normalise(
                sk_resize(hema_128, (size, size), anti_aliasing=True)
            ).astype(np.float64)
        # Intensity
        dgm0, dgm1 = ph_on_grid(grid)
        parts.append(ranked_vec(dgm0, dgm1))
        # Directional
        for angle in DIRECTIONS:
            dgm0, dgm1 = directional_filtration(grid, angle, size=size)
            parts.append(ranked_vec(dgm0, dgm1))
    return np.concatenate(parts)

# ── load cached images ────────────────────────────────────────────────────────
ts("Loading cached ROI images …")
img_data     = np.load(CACHE_IMAGES)
meta_data    = np.load(CACHE_META, allow_pickle=True)
all_imgs     = img_data["thumbnails"]        # (N, 128, 128, 3) float32
grades_all   = meta_data["grades"]
N            = len(all_imgs)
ts(f"  {N} ROIs  grades: { {g: int((grades_all==g).sum()) for g in [3,4,5]} }")

# Precompute hematoxylin at 128×128 for all ROIs (used by all variants)
ts("Precomputing hematoxylin channels …")
hema_all = np.stack([hema_from_thumb(all_imgs[i]) for i in range(N)])  # (N,128,128)
ts(f"  hema_all shape: {hema_all.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# Compute feature matrices
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(extract_fn, cache_path, label):
    if not FORCE and os.path.exists(cache_path):
        ts(f"Loading cached {label} …")
        data = np.load(cache_path)
        X    = data["X"]
        ts(f"  Loaded {X.shape}")
        return X

    ts(f"Computing {label} features …")
    vecs = []
    for i in range(N):
        v = extract_fn(hema_all[i])
        vecs.append(v)
        if (i+1) % 50 == 0:
            ts(f"  {i+1}/{N}")

    min_len = min(len(v) for v in vecs)
    X       = np.stack([v[:min_len] for v in vecs]).astype(np.float32)
    ts(f"  {label} X shape: {X.shape}")
    np.savez_compressed(cache_path, X=X)
    ts(f"  Saved → {cache_path}")
    return X

# Baseline: reuse existing cache (different key name)
if not FORCE and os.path.exists(CACHE_BASELINE_V2):
    ts("Loading baseline (paper method) …")
    X_baseline = np.load(CACHE_BASELINE_V2)["X_raw"].astype(np.float32)
    ts(f"  X_baseline {X_baseline.shape}")
else:
    # Recompute at 64×64 to match original pipeline
    ts("Computing baseline features …")
    vecs = []
    for i in range(N):
        grid = normalise(
            sk_resize(hema_all[i], (64,64), anti_aliasing=True)
        ).astype(np.float64)
        dgm0, dgm1 = ph_on_grid(grid)
        vecs.append(ranked_vec(dgm0, dgm1))
        if (i+1) % 50 == 0: ts(f"  {i+1}/{N}")
    min_len    = min(len(v) for v in vecs)
    X_baseline = np.stack([v[:min_len] for v in vecs]).astype(np.float32)
    np.savez_compressed(CACHE_BASELINE_V2, X_raw=X_baseline)
    ts(f"  X_baseline {X_baseline.shape}")

X_spatial    = build_feature_matrix(extract_spatial,    CACHE_SPATIAL,    "spatial (PHT)")
X_multiscale = build_feature_matrix(extract_multiscale, CACHE_MULTISCALE, "multiscale")
X_combined   = build_feature_matrix(extract_combined,   CACHE_COMBINED,   "combined")

variants = {
    "Baseline\n(paper)"  : X_baseline,
    "Spatial\n(PHT)"     : X_spatial,
    "Multiscale"         : X_multiscale,
    "Combined\n(PHT+MS)" : X_combined,
}

ts(f"\nFeature matrix sizes:")
for name, X in variants.items():
    ts(f"  {name.replace(chr(10),' '):<25} {X.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# Clustering pipeline (same for all variants)
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_fracs(lbls, grades, k=K_CLUSTERS):
    out = np.zeros((k, 3))
    for ci, c in enumerate(range(1, k+1)):
        mask  = lbls == c
        total = mask.sum() + 1e-9
        for gi, g in enumerate([3,4,5]):
            out[ci, gi] = (grades[mask]==g).sum() / total
    return out

def agglomerative_coeff(Z, n):
    """
    AC = 1 - (1/n) * sum_i( merger_height(i) / max_height )
    Higher = stronger clustering structure.
    """
    heights  = Z[:, 2]
    max_h    = heights.max() + 1e-9
    # For each original observation, find the height at which it first merged
    # Approximate: mean of all merge heights weighted by cluster size
    ac = 1 - heights.mean() / max_h
    return float(ac)

def run_pipeline(X, grades, label):
    """
    Returns dict with:
      Xpca_rep, Xtsne, clusters_rep, grades_rep,
      silhouette_mean, silhouette_std,
      ac_mean, ac_std,
      g5_purity_mean, g5_purity_std
    """
    ts(f"  Pipeline: {label}")
    grades_u  = np.array(sorted(np.unique(grades)))
    min_count = min((grades==g).sum() for g in grades_u)

    rng          = np.random.default_rng(0)
    boot_results = []
    silhouettes  = []
    acs          = []

    for b in range(N_BOOTSTRAP):
        idx_boot = []
        for g in grades_u:
            g_idx  = np.where(grades == g)[0]
            chosen = rng.choice(g_idx, size=min_count, replace=False)
            idx_boot.extend(chosen.tolist())
        idx_boot = np.array(idx_boot)

        X_b  = X[idx_boot]
        gr_b = grades[idx_boot]

        pca  = PCA(n_components=min(N_PCA, X_b.shape[1], X_b.shape[0]-1))
        Xpca = pca.fit_transform(X_b)

        Z    = linkage(Xpca, method="ward")
        lbls = fcluster(Z, K_CLUSTERS, criterion="maxclust")

        # Silhouette score
        if len(np.unique(lbls)) > 1:
            sil = silhouette_score(Xpca, lbls)
        else:
            sil = 0.0
        silhouettes.append(sil)

        # Agglomerative coefficient
        acs.append(agglomerative_coeff(Z, len(Xpca)))

        boot_results.append((Xpca, lbls, gr_b, idx_boot))

    # Find representative bootstrap
    all_cfrac = np.stack([cluster_fracs(r[1], r[2]) for r in boot_results])
    meta_cent = all_cfrac.mean(axis=0)
    dists     = [np.linalg.norm(all_cfrac[b] - meta_cent) for b in range(N_BOOTSTRAP)]
    rep_b     = int(np.argmin(dists))

    Xpca_rep, lbls_rep, gr_rep, idx_rep = boot_results[rep_b]

    # Order clusters by aggressiveness
    cfrac_rep = cluster_fracs(lbls_rep, gr_rep)
    scores    = cfrac_rep @ np.array([3,4,5])
    order     = np.argsort(scores)
    remap     = {old+1: new+1 for new, old in enumerate(order)}
    clusters_ordered = np.array([remap[l] for l in lbls_rep])

    # G5 purity: fraction of G5 in the most G5-dominant cluster
    g5_purities = []
    for b in range(N_BOOTSTRAP):
        _, lbls_b, gr_b, _ = boot_results[b]
        cf = cluster_fracs(lbls_b, gr_b)
        g5_purities.append(cf[:, 2].max())   # max G5 fraction across clusters

    # t-SNE
    perp  = min(40, len(Xpca_rep)//4)
    tsne  = TSNE(n_components=2, perplexity=perp, random_state=7,
                 max_iter=2000, learning_rate="auto", init="pca")
    Xtsne = tsne.fit_transform(Xpca_rep)

    ts(f"    sil={np.mean(silhouettes):.3f}±{np.std(silhouettes):.3f}  "
       f"AC={np.mean(acs):.3f}±{np.std(acs):.3f}  "
       f"G5_purity={np.mean(g5_purities):.3f}±{np.std(g5_purities):.3f}")

    return {
        "Xpca_rep"       : Xpca_rep,
        "Xtsne"          : Xtsne,
        "clusters_rep"   : clusters_ordered,
        "grades_rep"     : gr_rep,
        "cfrac"          : cluster_fracs(clusters_ordered, gr_rep),
        "silhouette_mean": float(np.mean(silhouettes)),
        "silhouette_std" : float(np.std(silhouettes)),
        "ac_mean"        : float(np.mean(acs)),
        "ac_std"         : float(np.std(acs)),
        "g5_purity_mean" : float(np.mean(g5_purities)),
        "g5_purity_std"  : float(np.std(g5_purities)),
    }

ts("\nRunning clustering pipeline for all variants …")
results = {}
for name, X in variants.items():
    results[name] = run_pipeline(X, grades_all, name)

# ═══════════════════════════════════════════════════════════════════════════════
# Find representative ROIs per (cluster, grade) using all-ROI PCA
# ═══════════════════════════════════════════════════════════════════════════════
ts("Finding representative ROIs …")

def get_rep_rois(X, result):
    pca_all  = PCA(n_components=N_PCA)
    Xpca_all = pca_all.fit_transform(X)
    centroids = np.zeros((K_CLUSTERS, N_PCA))
    for c in range(1, K_CLUSTERS+1):
        mask = result["clusters_rep"] == c
        centroids[c-1] = result["Xpca_rep"][mask].mean(axis=0)
    rois = {}
    for c in range(1, K_CLUSTERS+1):
        for g in [3,4,5]:
            g_idx = np.where(grades_all == g)[0]
            if len(g_idx) == 0:
                rois[(c,g)] = None
                continue
            dists = np.linalg.norm(Xpca_all[g_idx] - centroids[c-1], axis=1)
            best  = g_idx[np.argmin(dists)]
            rois[(c,g)] = all_imgs[best]
    return rois

rep_rois_all = {name: get_rep_rois(X, results[name])
                for name, X in variants.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Metrics comparison bar chart
# ═══════════════════════════════════════════════════════════════════════════════
ts("Plotting metrics comparison …")

metric_names   = ["Silhouette Score", "Agglomerative\nCoefficient", "G5 Cluster\nPurity"]
metric_keys_m  = ["silhouette_mean", "ac_mean",  "g5_purity_mean"]
metric_keys_s  = ["silhouette_std",  "ac_std",   "g5_purity_std"]
var_names      = list(variants.keys())
var_colors     = ["#555555", "#e41a1c", "#377eb8", "#4daf4a"]

fig_m, axes_m = plt.subplots(1, 3, figsize=(14, 4))
fig_m.patch.set_facecolor("white")

for ai, (mname, mk, sk) in enumerate(zip(metric_names, metric_keys_m, metric_keys_s)):
    ax   = axes_m[ai]
    vals = [results[n][mk] for n in var_names]
    errs = [results[n][sk] for n in var_names]
    x    = np.arange(len(var_names))
    bars = ax.bar(x, vals, 0.6, color=var_colors,
                  yerr=errs, capsize=4,
                  error_kw={"elinewidth":1.2, "ecolor":"#333"})
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("\n"," ") for n in var_names],
                       fontsize=8, rotation=15, ha="right")
    ax.set_title(mname, fontsize=10, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, min(1.15, max(vals)*1.35))
    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

fig_m.suptitle("Clustering Quality: Baseline vs Novel Features",
               fontsize=12, fontweight="bold")
plt.tight_layout()
fig_m.savefig(os.path.join(OUT_DIR, "metrics_comparison.png"),
              dpi=150, bbox_inches="tight")
ts("  Saved metrics_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: 2×2 t-SNE grid (one panel per variant)
# ═══════════════════════════════════════════════════════════════════════════════
ts("Plotting t-SNE comparison …")

CLUSTER_COLORS = ["#e41a1c","#ff7f00","#4daf4a","#377eb8","#984ea3","#a65628"]
GRADE_MARKERS  = {3:"o", 4:"^", 5:"s"}
roman          = ["i","ii","iii","iv","v","vi"]

fig_t, axes_t = plt.subplots(2, 2, figsize=(14, 12))
fig_t.patch.set_facecolor("white")
axes_t = axes_t.flatten()

for ai, (name, res) in enumerate(results.items()):
    ax     = axes_t[ai]
    Xtsne  = res["Xtsne"]
    clust  = res["clusters_rep"]
    grades = res["grades_rep"]

    for c in range(1, K_CLUSTERS+1):
        col = CLUSTER_COLORS[c-1]
        for g in [3,4,5]:
            mask = (clust==c) & (grades==g)
            if mask.sum()==0: continue
            ax.scatter(Xtsne[mask,0], Xtsne[mask,1],
                       c=col, marker=GRADE_MARKERS[g],
                       s=35, alpha=0.85, linewidths=0.3,
                       edgecolors="white", zorder=3)

    for c in range(1, K_CLUSTERS+1):
        mask = clust == c
        if mask.sum() == 0: continue
        mx, my = Xtsne[mask,0].mean(), Xtsne[mask,1].mean()
        ax.text(mx, my, roman[c-1], fontsize=10, fontweight="bold",
                ha="center", va="center", color=CLUSTER_COLORS[c-1],
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7), zorder=5)

    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top","right","bottom","left"]].set_visible(False)

    sil = res["silhouette_mean"]
    g5p = res["g5_purity_mean"]
    ax.set_title(f"{name.replace(chr(10),' ')}   "
                 f"sil={sil:.3f}  G5-purity={g5p:.3f}",
                 fontsize=10, fontweight="bold")

# Shared legend
leg_elements = [
    plt.Line2D([0],[0],marker="o",color="grey",ls="",ms=8,label="G3"),
    plt.Line2D([0],[0],marker="^",color="grey",ls="",ms=8,label="G4"),
    plt.Line2D([0],[0],marker="s",color="grey",ls="",ms=8,label="G5"),
] + [
    plt.Line2D([0],[0],marker="o",color=CLUSTER_COLORS[c-1],ls="",ms=8,
               label=f"cluster {roman[c-1]}")
    for c in range(1, K_CLUSTERS+1)
]
fig_t.legend(handles=leg_elements, fontsize=8, loc="lower center",
             ncol=9, bbox_to_anchor=(0.5, -0.02), framealpha=0.8)

fig_t.suptitle("t-SNE: Baseline vs Novel Feature Variants",
               fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
fig_t.savefig(os.path.join(OUT_DIR, "tsne_comparison.png"),
              dpi=150, bbox_inches="tight")
ts("  Saved tsne_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Figure-4-style panel for best variant
# ═══════════════════════════════════════════════════════════════════════════════
ts("Plotting Figure-4-style panel for best variant …")

# Pick best variant by silhouette score
best_name = max(results, key=lambda n: results[n]["silhouette_mean"])
ts(f"  Best variant: {best_name.replace(chr(10),' ')}")

res       = results[best_name]
rep_rois  = rep_rois_all[best_name]
cfrac     = res["cfrac"]
clust     = res["clusters_rep"]
grades_r  = res["grades_rep"]

# Bootstrap error bars on cfrac
rng2    = np.random.default_rng(42)
boot_cf = []
for _ in range(200):
    idx = rng2.integers(0, len(grades_r), size=len(grades_r))
    boot_cf.append(cluster_fracs(clust[idx], grades_r[idx]))
cfrac_std = np.std(boot_cf, axis=0)

GRADE_FILL = {3:"#2c3e50", 4:"#7f8c8d", 5:"#bdc3c7"}

fig_best = plt.figure(figsize=(22, 11))
fig_best.patch.set_facecolor("white")
gs = GridSpec(5, 8, figure=fig_best,
              height_ratios=[0.15, 1, 1, 1, 0.8],
              width_ratios=[1.8, 0.15, 1,1,1,1,1,1],
              wspace=0.08, hspace=0.12,
              left=0.03, right=0.98, top=0.92, bottom=0.04)

ax_tsne = fig_best.add_subplot(gs[0:5, 0])
for c in range(1, K_CLUSTERS+1):
    col = CLUSTER_COLORS[c-1]
    for g in [3,4,5]:
        mask = (clust==c) & (grades_r==g)
        if mask.sum()==0: continue
        ax_tsne.scatter(res["Xtsne"][mask,0], res["Xtsne"][mask,1],
                        c=col, marker=GRADE_MARKERS[g],
                        s=30, alpha=0.85, linewidths=0.3,
                        edgecolors="white", zorder=3)
for c in range(1, K_CLUSTERS+1):
    mask = clust == c
    if mask.sum() == 0: continue
    mx = res["Xtsne"][mask,0].mean()
    my = res["Xtsne"][mask,1].mean()
    ax_tsne.text(mx, my, roman[c-1], fontsize=11, fontweight="bold",
                ha="center", va="center", color=CLUSTER_COLORS[c-1],
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.7), zorder=5)
ax_tsne.set_xticks([]); ax_tsne.set_yticks([])
ax_tsne.spines[["top","right","bottom","left"]].set_visible(False)
grade_leg = [plt.Line2D([0],[0],marker=GRADE_MARKERS[g],color="grey",
                        ls="",ms=7,label=f"G{g}") for g in [3,4,5]]
clust_leg = [plt.Line2D([0],[0],marker="o",color=CLUSTER_COLORS[c-1],
                        ls="",ms=7,label=roman[c-1])
             for c in range(1,K_CLUSTERS+1)]
leg1 = ax_tsne.legend(handles=grade_leg, fontsize=7, loc="lower left",
                      framealpha=0.8, edgecolor="grey",
                      title="Grade", title_fontsize=7)
ax_tsne.add_artist(leg1)
ax_tsne.legend(handles=clust_leg, fontsize=7, loc="lower right",
               framealpha=0.8, edgecolor="grey",
               title="Cluster", title_fontsize=7, ncol=2)
ax_tsne.set_title(f"t-SNE\n{best_name.replace(chr(10),' ')}",
                  fontsize=9, fontweight="bold", pad=3)

for ci, c in enumerate(range(1, K_CLUSTERS+1)):
    ax_h = fig_best.add_subplot(gs[0, ci+2])
    ax_h.axis("off")
    ax_h.text(0.5, 0.3, roman[c-1], transform=ax_h.transAxes,
              fontsize=14, fontweight="bold", ha="center", va="center",
              color=CLUSTER_COLORS[c-1])

for ri, g in enumerate([3,4,5]):
    for ci, c in enumerate(range(1, K_CLUSTERS+1)):
        ax  = fig_best.add_subplot(gs[ri+1, ci+2])
        img = rep_rois.get((c,g))
        if img is not None:
            ax.imshow(np.clip(img, 0, 1))
            for sp in ax.spines.values():
                sp.set_edgecolor(CLUSTER_COLORS[c-1])
                sp.set_linewidth(2.5); sp.set_visible(True)
        else:
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5,0.5,"—",ha="center",va="center",
                   transform=ax.transAxes,color="#aaaaaa",fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])

for ri, g in enumerate([3,4,5]):
    ax_gl = fig_best.add_subplot(gs[ri+1, 1])
    ax_gl.axis("off")
    ax_gl.text(0.5, 0.5, f"G{g}", transform=ax_gl.transAxes,
               fontsize=9, va="center", ha="center",
               color="#555555", style="italic", rotation=90)

x = np.arange(3); w = 0.55
for ci, c in enumerate(range(1, K_CLUSTERS+1)):
    ax = fig_best.add_subplot(gs[4, ci+2])
    ax.bar(x, cfrac[c-1], w,
           color=[GRADE_FILL[g] for g in [3,4,5]],
           yerr=cfrac_std[c-1], capsize=2,
           error_kw={"elinewidth":0.8, "ecolor":"#333"})
    ax.set_xticks(x)
    ax.set_xticklabels(["G3","G4","G5"], fontsize=6)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="y", labelsize=6)
    ax.spines[["top","right"]].set_visible(False)
    if ci == 0: ax.set_ylabel("Fraction", fontsize=7)
    else:       ax.set_yticklabels([])
    n_c = (clust==c).sum()
    ax.text(0.5, 1.05, f"n={n_c}", transform=ax.transAxes,
            fontsize=6, ha="center", va="bottom", color="#444")

grade_counts = {g: int((grades_all==g).sum()) for g in [3,4,5]}
fig_best.suptitle(
    f"Best variant: {best_name.replace(chr(10),' ')}  —  "
    f"silhouette={results[best_name]['silhouette_mean']:.3f}  "
    f"G5-purity={results[best_name]['g5_purity_mean']:.3f}\n"
    f"({len(grades_all)} ROIs  |  "
    f"G3={grade_counts[3]}  G4={grade_counts[4]}  G5={grade_counts[5]}  |  "
    f"k={K_CLUSTERS} clusters  |  {N_BOOTSTRAP} bootstraps)",
    fontsize=11, fontweight="bold", y=0.97
)
plt.savefig(os.path.join(OUT_DIR, "figure4_best_variant.png"),
            dpi=150, bbox_inches="tight")
ts("  Saved figure4_best_variant.png")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Variant':<25} {'Silhouette':>12} {'AC':>10} {'G5-Purity':>12}")
print("="*65)
for name, res in results.items():
    nm = name.replace("\n"," ")
    marker = " ◄" if name == best_name else ""
    print(f"{nm:<25} "
          f"{res['silhouette_mean']:>6.3f}±{res['silhouette_std']:.3f}  "
          f"{res['ac_mean']:>5.3f}±{res['ac_std']:.3f}  "
          f"{res['g5_purity_mean']:>6.3f}±{res['g5_purity_std']:.3f}"
          f"{marker}")
print("="*65)
print(f"\nOutputs saved to: {OUT_DIR}")
print("  metrics_comparison.png")
print("  tsne_comparison.png")
print("  figure4_best_variant.png")