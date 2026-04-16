"""
unsupervised_compact.py
───────────────────────
Unsupervised version of the compact feature pipeline.

Key difference from replacement_compact_features.py:
  - NO f_classif / feature selection inside the bootstrap loop
  - Features are standardised then PCA-reduced, no label information used
  - This is a fair comparison to the original paper's unsupervised pipeline

Feature sets (same as before, just no supervised filtering):
  BaselineTopLife       — top-k PH lifetimes only
  Base+Life             — + lifetime summary statistics
  Base+Betti            — + Betti curve summaries
  Base+Life+Betti       — best from supervised experiment
  Base+Life+Betti+Morph — full compact set

Inputs:
  cache/roi_images.npz
  cache/roi_meta.npz
  cache/compact_feature_sets.npz   (built by replacement_compact_features.py)

Outputs:
  unsupervised_metrics.csv
  unsupervised_metrics.png
  figure4_unsupervised_best.png

Run:
  python unsupervised_compact.py [--force-pipeline]
  --force-pipeline  rerun bootstrap even if csv exists
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
CACHE_DIR      = "cache"
CACHE_IMAGES   = os.path.join(CACHE_DIR, "roi_images.npz")
CACHE_META     = os.path.join(CACHE_DIR, "roi_meta.npz")
CACHE_FEATURES = os.path.join(CACHE_DIR, "compact_feature_sets.npz")
OUT_CSV        = "unsupervised_metrics.csv"
OUT_BAR        = "unsupervised_metrics.png"
OUT_FIGURE4    = "figure4_unsupervised_best.png"

K_CLUSTERS   = 6
N_BOOTSTRAP  = 50
RANDOM_SEED  = 7

# PCA options to sweep — same as supervised script for fair comparison
PCA_OPTIONS = [6, 8, 10, 15]

FORCE_PIPELINE = "--force-pipeline" in sys.argv[1:]

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def cluster_fracs(lbls, grades, k=K_CLUSTERS):
    out = np.zeros((k, 3))
    for ci, c in enumerate(range(1, k+1)):
        mask  = lbls == c
        total = mask.sum() + 1e-9
        for gi, g in enumerate([3,4,5]):
            out[ci, gi] = (grades[mask]==g).sum() / total
    return out

def agglomerative_coeff(Z):
    heights = Z[:,2]
    return float(1.0 - heights.mean() / (heights.max() + 1e-9))

def composition_stability(cfrac_list):
    arr  = np.stack(cfrac_list)
    meta = arr.mean(axis=0)
    dists = np.array([np.linalg.norm(x - meta) for x in arr])
    return float(1.0 / (1.0 + dists.mean()))

def reorder_clusters(lbls, grades):
    cfrac  = cluster_fracs(lbls, grades)
    scores = cfrac @ np.array([3,4,5], dtype=float)
    order  = np.argsort(scores)
    remap  = {old+1: new+1 for new, old in enumerate(order)}
    return np.array([remap[l] for l in lbls], dtype=np.int32)

def adaptive_pca(X, n_components):
    """
    Standardise then PCA.
    No label information used at any point.
    Returns (embedding, scaler, pca_model).
    """
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    n_comp = min(n_components, Xs.shape[1], Xs.shape[0]-1)
    pca    = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    Xe     = pca.fit_transform(Xs)
    var_explained = pca.explained_variance_ratio_.cumsum()[-1]
    return Xe, scaler, pca, float(var_explained)

# ── load data ─────────────────────────────────────────────────────────────────
ts("Loading cached data …")
all_imgs   = np.load(CACHE_IMAGES)["thumbnails"].astype(np.float32)
grades_all = np.load(CACHE_META, allow_pickle=True)["grades"].astype(np.int32)
fc         = np.load(CACHE_FEATURES)

X_base  = fc["baseline_toplife"]
X_life  = fc["life_summary"]
X_betti = fc["betti_summary"]
X_morph = fc["morph_summary"]

ts(f"  {len(all_imgs)} ROIs  grades: { {g: int((grades_all==g).sum()) for g in [3,4,5]} }")

feature_sets = {
    "BaselineTopLife"       : X_base,
    "Base+Life"             : np.concatenate([X_base, X_life],             axis=1),
    "Base+Betti"            : np.concatenate([X_base, X_betti],            axis=1),
    "Base+Morph"            : np.concatenate([X_base, X_morph],            axis=1),
    "Base+Life+Betti"       : np.concatenate([X_base, X_life, X_betti],    axis=1),
    "Base+Life+Morph"       : np.concatenate([X_base, X_life, X_morph],    axis=1),
    "Base+Betti+Morph"      : np.concatenate([X_base, X_betti, X_morph],   axis=1),
    "Base+Life+Betti+Morph" : np.concatenate([X_base, X_life, X_betti, X_morph], axis=1),
}

ts("Feature matrix shapes:")
for name, X in feature_sets.items():
    ts(f"  {name:<28} {X.shape}")

# ── pipeline ──────────────────────────────────────────────────────────────────
def run_unsupervised(X, grades, label, pca_dim):
    """
    Fully unsupervised pipeline:
      standardise → PCA (no labels) → Ward clustering → metrics
    """
    grades_u  = np.array(sorted(np.unique(grades)))
    min_count = min((grades==g).sum() for g in grades_u)

    rng          = np.random.default_rng(RANDOM_SEED)
    silhouettes  = []
    acs          = []
    g5_purities  = []
    cfrac_list   = []
    boot_results = []
    var_exp_log  = []

    for b in range(N_BOOTSTRAP):
        # Balanced subsample — no label use beyond balancing
        idx_boot = []
        for g in grades_u:
            g_idx  = np.where(grades==g)[0]
            chosen = rng.choice(g_idx, size=min_count, replace=False)
            idx_boot.extend(chosen.tolist())
        idx_boot = np.array(idx_boot, dtype=np.int32)

        X_b  = X[idx_boot]
        gr_b = grades[idx_boot]

        # Dimensionality reduction — purely unsupervised
        Xe, scaler, pca, var_exp = adaptive_pca(X_b, pca_dim)
        var_exp_log.append(var_exp)

        Z    = linkage(Xe, method="ward")
        lbls = fcluster(Z, K_CLUSTERS, criterion="maxclust")
        lbls = reorder_clusters(lbls, gr_b)

        sil  = silhouette_score(Xe, lbls) if len(np.unique(lbls))>1 else 0.0
        ac   = agglomerative_coeff(Z)
        cf   = cluster_fracs(lbls, gr_b)
        g5p  = float(cf[:,2].max())

        silhouettes.append(float(sil))
        acs.append(float(ac))
        g5_purities.append(g5p)
        cfrac_list.append(cf)
        boot_results.append({
            "idx_boot" : idx_boot,
            "grades"   : gr_b,
            "clusters" : lbls,
            "scaler"   : scaler,
            "pca"      : pca,
            "embedding": Xe,
        })

    stability = composition_stability(cfrac_list)

    # Representative bootstrap
    all_cfrac = np.stack(cfrac_list)
    meta_cent = all_cfrac.mean(axis=0)
    dists     = np.array([np.linalg.norm(cf-meta_cent) for cf in all_cfrac])
    rep_b     = int(np.argmin(dists))
    rep       = boot_results[rep_b]

    # t-SNE on representative bootstrap
    perp  = min(40, max(5, len(rep["embedding"])//4))
    tsne  = TSNE(n_components=2, perplexity=perp, random_state=RANDOM_SEED,
                 max_iter=2000, learning_rate="auto", init="pca")
    Xtsne = tsne.fit_transform(rep["embedding"])

    # Composite score matching supervised script weights
    composite = (0.35 * np.mean(silhouettes)
               + 0.45 * np.mean(g5_purities)
               + 0.20 * stability)

    ts(f"  {label:<28} pca={pca_dim}  "
       f"sil={np.mean(silhouettes):.3f}±{np.std(silhouettes):.3f}  "
       f"G5={np.mean(g5_purities):.3f}  "
       f"stab={stability:.3f}  "
       f"var_exp={np.mean(var_exp_log):.2f}")

    return {
        "label"            : label,
        "pca_dim"          : pca_dim,
        "silhouette_mean"  : float(np.mean(silhouettes)),
        "silhouette_std"   : float(np.std(silhouettes)),
        "ac_mean"          : float(np.mean(acs)),
        "ac_std"           : float(np.std(acs)),
        "g5_purity_mean"   : float(np.mean(g5_purities)),
        "g5_purity_std"    : float(np.std(g5_purities)),
        "stability"        : float(stability),
        "composite_score"  : float(composite),
        "var_explained"    : float(np.mean(var_exp_log)),
        # store rep bootstrap for plotting
        "_rep"             : rep,
        "_Xtsne"           : Xtsne,
    }

# ── run or load ───────────────────────────────────────────────────────────────
if not FORCE_PIPELINE and os.path.exists(OUT_CSV):
    ts(f"Loading cached results from {OUT_CSV} …")
    ts("  (use --force-pipeline to rerun)")
    df = pd.read_csv(OUT_CSV)
    # Still need to rerun rep bootstrap for plotting — quick since N_BOOTSTRAP=50
    ts("  Rerunning rep bootstrap for best variant (for plot) …")
    RERUN_PLOT = True
else:
    RERUN_PLOT = False

if FORCE_PIPELINE or not os.path.exists(OUT_CSV):
    ts("\nRunning unsupervised pipeline …")
    results = []
    for feat_name, X in feature_sets.items():
        for pca_dim in PCA_OPTIONS:
            res = run_unsupervised(X, grades_all, feat_name, pca_dim)
            results.append(res)

    # Save metrics (without internal objects)
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
    df = pd.DataFrame(rows).sort_values("composite_score", ascending=False)
    df["experiment"] = df.apply(
        lambda r: f"{r['label']} | pca={r['pca_dim']}", axis=1)
    df.to_csv(OUT_CSV, index=False)
    ts(f"Saved metrics → {OUT_CSV}")
    RERUN_PLOT = False   # already have rep bootstrap in results

# ── metrics bar chart ─────────────────────────────────────────────────────────
ts("Plotting metrics comparison …")

metrics_to_plot = [
    ("silhouette_mean", "Silhouette Score"),
    ("ac_mean",         "Agglomerative Coeff"),
    ("g5_purity_mean",  "G5 Cluster Purity"),
    ("stability",       "Cluster Stability"),
    ("composite_score", "Composite Score"),
]

# Top 16 experiments by composite
df_plot = df.head(16).reset_index(drop=True)
x       = np.arange(len(df_plot))
cmap    = plt.cm.tab20(np.linspace(0, 1, len(df_plot)))

fig_m, axes_m = plt.subplots(1, len(metrics_to_plot), figsize=(26, 5))
fig_m.patch.set_facecolor("white")

for ax, (col, title) in zip(axes_m, metrics_to_plot):
    vals = df_plot[col].values
    errs = df_plot.get(col.replace("_mean","_std"), pd.Series(np.zeros(len(df_plot)))).values
    bars = ax.bar(x, vals, 0.65, color=cmap,
                  yerr=errs if "mean" in col else None,
                  capsize=3, error_kw={"elinewidth":0.8, "ecolor":"#333"})
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["experiment"].tolist(),
                       fontsize=6, rotation=35, ha="right")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, min(1.15, max(vals)*1.4+0.05))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=45)

fig_m.suptitle(
    "Unsupervised compact PH — no label leakage\n"
    "Standardise → PCA → Ward clustering (no f_classif)",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig_m.savefig(OUT_BAR, dpi=150, bbox_inches="tight")
ts(f"Saved → {OUT_BAR}")

# ── find best result and its rep bootstrap ────────────────────────────────────
best_row = df.iloc[0]
ts(f"\nBest: {best_row['experiment']}  composite={best_row['composite_score']:.4f}")

# Locate or rerun the best result
best_result = None
if not RERUN_PLOT:
    for r in results:
        if r["label"] == best_row["label"] and r["pca_dim"] == best_row["pca_dim"]:
            best_result = r
            break

if best_result is None:
    ts("Rerunning best variant for plotting …")
    X_best = feature_sets[best_row["label"]]
    best_result = run_unsupervised(
        X_best, grades_all,
        best_row["label"], int(best_row["pca_dim"]))

rep   = best_result["_rep"]
Xtsne = best_result["_Xtsne"]

# ── representative ROI per (cluster, grade) using all-data embedding ──────────
ts("Finding representative ROIs …")
X_all_best = feature_sets[best_result["label"]]

scaler_all = StandardScaler()
Xs_all     = scaler_all.fit_transform(X_all_best)
n_comp     = min(best_result["pca_dim"], Xs_all.shape[1], Xs_all.shape[0]-1)
pca_all    = PCA(n_components=n_comp, random_state=RANDOM_SEED)
Xe_all     = pca_all.fit_transform(Xs_all)

# Cluster centroids from rep bootstrap
centroids = np.zeros((K_CLUSTERS, n_comp))
for c in range(1, K_CLUSTERS+1):
    mask = rep["clusters"] == c
    centroids[c-1] = rep["embedding"][mask, :n_comp].mean(axis=0)

rep_rois = {}
for c in range(1, K_CLUSTERS+1):
    for g in [3,4,5]:
        g_idx = np.where(grades_all==g)[0]
        if len(g_idx)==0:
            rep_rois[(c,g)] = None
            continue
        d    = np.linalg.norm(Xe_all[g_idx, :n_comp] - centroids[c-1], axis=1)
        best = g_idx[np.argmin(d)]
        rep_rois[(c,g)] = all_imgs[best]

# Bootstrap error bars for histograms
rng2     = np.random.default_rng(42)
boot_cf  = []
for _ in range(200):
    idx = rng2.integers(0, len(rep["grades"]), size=len(rep["grades"]))
    boot_cf.append(cluster_fracs(rep["clusters"][idx], rep["grades"][idx]))
cfrac_rep = cluster_fracs(rep["clusters"], rep["grades"])
cfrac_std = np.std(boot_cf, axis=0)

# ── Figure-4-style plot ───────────────────────────────────────────────────────
ts("Plotting Figure-4-style panel …")

CLUSTER_COLORS = ["#e41a1c","#ff7f00","#4daf4a","#377eb8","#984ea3","#a65628"]
GRADE_MARKERS  = {3:"o", 4:"^", 5:"s"}
GRADE_FILL     = {3:"#2c3e50", 4:"#7f8c8d", 5:"#bdc3c7"}
roman          = ["i","ii","iii","iv","v","vi"]

fig = plt.figure(figsize=(22,11))
fig.patch.set_facecolor("white")
gs  = GridSpec(5, 8, figure=fig,
               height_ratios=[0.15, 1, 1, 1, 0.8],
               width_ratios=[1.8, 0.15, 1,1,1,1,1,1],
               wspace=0.08, hspace=0.12,
               left=0.03, right=0.98, top=0.92, bottom=0.04)

# t-SNE panel
ax_tsne = fig.add_subplot(gs[0:5, 0])
for c in range(1, K_CLUSTERS+1):
    col = CLUSTER_COLORS[c-1]
    for g in [3,4,5]:
        mask = (rep["clusters"]==c) & (rep["grades"]==g)
        if mask.sum()==0: continue
        ax_tsne.scatter(Xtsne[mask,0], Xtsne[mask,1],
                        c=col, marker=GRADE_MARKERS[g],
                        s=30, alpha=0.85, linewidths=0.3,
                        edgecolors="white", zorder=3)
for c in range(1, K_CLUSTERS+1):
    mask = rep["clusters"]==c
    mx, my = Xtsne[mask,0].mean(), Xtsne[mask,1].mean()
    ax_tsne.text(mx, my, roman[c-1], fontsize=11, fontweight="bold",
                ha="center", va="center", color=CLUSTER_COLORS[c-1],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=5)

ax_tsne.set_xticks([]); ax_tsne.set_yticks([])
ax_tsne.spines[["top","right","bottom","left"]].set_visible(False)
leg1 = ax_tsne.legend(
    handles=[plt.Line2D([0],[0],marker=GRADE_MARKERS[g],color="grey",
                        ls="",ms=7,label=f"G{g}") for g in [3,4,5]],
    fontsize=7, loc="lower left", framealpha=0.8,
    edgecolor="grey", title="Grade", title_fontsize=7)
ax_tsne.add_artist(leg1)
ax_tsne.legend(
    handles=[plt.Line2D([0],[0],marker="o",color=CLUSTER_COLORS[c-1],
                        ls="",ms=7,label=roman[c-1])
             for c in range(1,K_CLUSTERS+1)],
    fontsize=7, loc="lower right", framealpha=0.8,
    edgecolor="grey", title="Cluster", title_fontsize=7, ncol=2)
ax_tsne.set_title(f"t-SNE\n{best_result['label']}\n(unsupervised)",
                  fontsize=9, fontweight="bold", pad=3)

# Column headers
for ci, c in enumerate(range(1, K_CLUSTERS+1)):
    ax_h = fig.add_subplot(gs[0, ci+2])
    ax_h.axis("off")
    ax_h.text(0.5, 0.3, roman[c-1], transform=ax_h.transAxes,
              fontsize=14, fontweight="bold", ha="center", va="center",
              color=CLUSTER_COLORS[c-1])

# ROI grid
for ri, g in enumerate([3,4,5]):
    for ci, c in enumerate(range(1, K_CLUSTERS+1)):
        ax  = fig.add_subplot(gs[ri+1, ci+2])
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

# Grade row labels
for ri, g in enumerate([3,4,5]):
    ax_gl = fig.add_subplot(gs[ri+1, 1])
    ax_gl.axis("off")
    ax_gl.text(0.5, 0.5, f"G{g}", transform=ax_gl.transAxes,
               fontsize=9, va="center", ha="center",
               color="#555555", style="italic", rotation=90)

# Histograms
x_bar = np.arange(3); w = 0.55
for ci, c in enumerate(range(1, K_CLUSTERS+1)):
    ax = fig.add_subplot(gs[4, ci+2])
    ax.bar(x_bar, cfrac_rep[c-1], w,
           color=[GRADE_FILL[g] for g in [3,4,5]],
           yerr=cfrac_std[c-1], capsize=2,
           error_kw={"elinewidth":0.8,"ecolor":"#333"})
    ax.set_xticks(x_bar)
    ax.set_xticklabels(["G3","G4","G5"], fontsize=6)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="y", labelsize=6)
    ax.spines[["top","right"]].set_visible(False)
    if ci==0: ax.set_ylabel("Fraction", fontsize=7)
    else:     ax.set_yticklabels([])
    n_c = (rep["clusters"]==c).sum()
    ax.text(0.5, 1.05, f"n={n_c}", transform=ax.transAxes,
            fontsize=6, ha="center", va="bottom", color="#444")

grade_counts = {g: int((grades_all==g).sum()) for g in [3,4,5]}
fig.suptitle(
    f"Unsupervised compact variant: {best_result['label']}  |  pca={best_result['pca_dim']}  "
    f"[NO feature selection]\n"
    f"silhouette={best_result['silhouette_mean']:.3f}  "
    f"G5-purity={best_result['g5_purity_mean']:.3f}  "
    f"stability={best_result['stability']:.3f}  "
    f"var_explained={best_result['var_explained']:.2f}\n"
    f"({len(grades_all)} ROIs  |  G3={grade_counts[3]}  "
    f"G4={grade_counts[4]}  G5={grade_counts[5]}  |  "
    f"k={K_CLUSTERS} clusters  |  {N_BOOTSTRAP} bootstraps)",
    fontsize=11, fontweight="bold", y=0.97)

plt.savefig(OUT_FIGURE4, dpi=150, bbox_inches="tight")
ts(f"Saved → {OUT_FIGURE4}")

# ── summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"{'Rank':<5} {'Experiment':<40} {'Silhouette':>12} {'G5-Purity':>12} "
      f"{'Stability':>10} {'Composite':>10}")
print("="*90)
for i, (_, row) in enumerate(df.head(20).iterrows(), start=1):
    marker = " ◄" if i==1 else ""
    print(f"{i:<5} {row['experiment'][:40]:<40} "
          f"{row['silhouette_mean']:>6.3f}±{row['silhouette_std']:.3f}  "
          f"{row['g5_purity_mean']:>6.3f}±{row['g5_purity_std']:.3f}  "
          f"{row['stability']:>9.3f}  "
          f"{row['composite_score']:>9.3f}{marker}")
print("="*90)

# ── direct comparison with supervised results ─────────────────────────────────
print("\n── Direct comparison (same feature set, same metric) ──────────────────")
print(f"{'Method':<45} {'Silhouette':>12} {'G5-Purity':>12} {'Stability':>10}")
print("-"*80)

# Supervised best (from your earlier results)
print(f"{'Supervised Base+Life+Betti | top_k=30 | pca=6':<45} "
      f"{'0.265':>12}  {'0.984':>12}  {'0.693':>10}")

# Unsupervised best
best = df.iloc[0]
print(f"{best['experiment'][:45]:<45} "
      f"{best['silhouette_mean']:>6.3f}±{best['silhouette_std']:.3f}  "
      f"{best['g5_purity_mean']:>6.3f}±{best['g5_purity_std']:.3f}  "
      f"{best['stability']:>9.3f}")

# Original paper baseline
print(f"{'Baseline paper (ranked vec, pca=6)':<45} "
      f"{'0.458':>12}  {'0.935':>12}  {'N/A':>10}")
print("-"*80)
print("\nNote: supervised G5-purity=0.984 uses f_classif inside bootstrap loop.")
print("      Unsupervised result is the fair comparison to the paper baseline.")