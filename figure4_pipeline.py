"""
figure4_cached.py
─────────────────
Full Figure 4 pipeline with persistent caching.

Cache files written to CACHE_DIR:
  roi_images.npz    — 128×128 RGB thumbnails for all ROIs
  roi_meta.npz      — grades, source (test_idx, ann_idx), centroid coords
  ph_vectors.npz    — ranked persistence vectors (X_raw) + per-ROI H0/H1 diagrams
  pipeline.npz      — PCA coords, cluster labels, t-SNE coords (rep bootstrap)

On subsequent runs:
  - If all cache files exist → skip directly to plotting (seconds)
  - If ph_vectors.npz exists but pipeline.npz missing → skip PH, redo clustering
  - If roi_images.npz exists but ph_vectors missing → skip image loading, redo PH

Run:
  python figure4_cached.py [--force] [--force-ph] [--force-pipeline]

Flags:
  --force          rebuild everything from scratch
  --force-ph       rebuild PH vectors (but reuse images if available)
  --force-pipeline rebuild PCA/cluster/tSNE (but reuse PH vectors)
"""

import json, os, sys, warnings, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from skimage import color, filters
from skimage.transform import resize as sk_resize
import tifffile
import gudhi
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
BASE_TIFF    = "magicScan_TDA_ML/Subset1/Subset1_Test_{}.tiff"
BASE_GEOJSON = "magicScan_TDA_ML/Subset1_annot/Subset1_Test_{}.geojson"
CACHE_DIR    = "cache"
OUT_PATH     = "figure4_cached.png"

ROI_SIZE     = 512
THUMB_SIZE   = 128    # size of saved thumbnail
PH_SIZE      = 64     # grid size for PH computation
N_TEST_FILES = 30
K_CLUSTERS   = 6
N_BOOTSTRAP  = 50
N_PCA        = 6
GAUSS_SIGMA  = 1.0

os.makedirs(CACHE_DIR, exist_ok=True)

# Cache file paths
CACHE_IMAGES   = os.path.join(CACHE_DIR, "roi_images.npz")
CACHE_META     = os.path.join(CACHE_DIR, "roi_meta.npz")
CACHE_PH       = os.path.join(CACHE_DIR, "ph_vectors.npz")
CACHE_PIPELINE = os.path.join(CACHE_DIR, "pipeline.npz")

# Parse flags
args         = set(sys.argv[1:])
FORCE_ALL      = "--force"          in args
FORCE_PH       = "--force-ph"       in args or FORCE_ALL
FORCE_PIPELINE = "--force-pipeline" in args or FORCE_PH or FORCE_ALL

def ts(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def load_and_crop(tiff_path, cx, cy, size=ROI_SIZE):
    raw = tifffile.imread(tiff_path)
    if raw.ndim == 4: raw = raw[0]
    if raw.ndim == 3 and raw.shape[0] in (1,3,4): raw = np.moveaxis(raw, 0, -1)
    if raw.shape[-1] == 4: raw = raw[..., :3]
    if raw.dtype != np.uint8: raw = (raw / raw.max() * 255).astype(np.uint8)
    H, W = raw.shape[:2]
    half  = size // 2
    cy_c  = max(half, min(H-half, int(cy)))
    cx_c  = max(half, min(W-half, int(cx)))
    crop  = raw[cy_c-half:cy_c+half, cx_c-half:cx_c+half]
    return crop, cy_c-half, cx_c-half

def get_label(props):
    return str((
        (props.get("classification") or {}).get("name")
        or props.get("label") or props.get("class")
        or props.get("objectType") or props.get("type") or "unknown"
    )).strip()

def label_to_grade(label):
    l = label.upper().replace(" ", "")
    for digit in ["3", "4", "5"]:
        if digit in l and any(k in l for k in ["G","GLEASON","GRADE",digit]):
            return int(digit)
    return None

def hematoxylin_norm(img_rgb):
    hed  = color.rgb2hed(img_rgb)
    h    = hed[:, :, 0]
    hmin = np.percentile(h, 1)
    hmax = np.percentile(h, 99)
    return np.clip((h-hmin)/(hmax-hmin+1e-9), 0, 1).astype(np.float64)

def compute_ph(hema_norm):
    small = sk_resize(hema_norm, (PH_SIZE, PH_SIZE), anti_aliasing=True).astype(np.float64)
    small = (small-small.min()) / (small.max()-small.min()+1e-9)
    cc    = gudhi.CubicalComplex(top_dimensional_cells=small)
    cc.compute_persistence()
    def get_fin(dim):
        arr = np.array(cc.persistence_intervals_in_dimension(dim), dtype=float)
        if arr.size == 0: return np.zeros((0, 2))
        return arr[np.isfinite(arr).all(axis=1)]
    return get_fin(0), get_fin(1)

def ranked_vec(dgm0, dgm1):
    def sp(dgm):
        if len(dgm) == 0: return np.array([])
        return np.sort(dgm[:,1]-dgm[:,0])[::-1]
    return np.concatenate([sp(dgm0), sp(dgm1)])

def cluster_fracs(lbls, grades, k=K_CLUSTERS):
    out = np.zeros((k, 3))
    for ci, c in enumerate(range(1, k+1)):
        mask  = lbls == c
        total = mask.sum() + 1e-9
        for gi, g in enumerate([3,4,5]):
            out[ci, gi] = (grades[mask]==g).sum() / total
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — ROI images + metadata
# ═══════════════════════════════════════════════════════════════════════════════
if not FORCE_ALL and os.path.exists(CACHE_IMAGES) and os.path.exists(CACHE_META):
    ts("STAGE 1: Loading cached ROI images …")
    img_data      = np.load(CACHE_IMAGES)
    meta_data     = np.load(CACHE_META, allow_pickle=True)
    all_imgs      = img_data["thumbnails"]           # (N, 128, 128, 3) float32
    grades_all    = meta_data["grades"]
    centroids_xy  = meta_data["centroids_xy"]        # (N, 2) [cx, cy]
    source_arr    = meta_data["source"]              # (N, 2) [test_idx, ann_idx]
    ts(f"  Loaded {len(all_imgs)} ROIs from cache")
else:
    ts("STAGE 1: Extracting ROI images from TIFFs …")
    all_imgs_list  = []
    grades_list    = []
    centroids_list = []
    source_list    = []

    for test_idx in range(1, N_TEST_FILES+1):
        tiff_path    = BASE_TIFF.format(test_idx)
        geojson_path = BASE_GEOJSON.format(test_idx)
        if not os.path.exists(tiff_path) or not os.path.exists(geojson_path):
            continue
        ts(f"  Test_{test_idx} …")

        with open(geojson_path) as fj:
            gj = json.load(fj)

        for ann_idx, feat in enumerate(gj.get("features", [])):
            props = feat.get("properties", {})
            label = get_label(props)
            grade = label_to_grade(label)
            if grade is None: continue

            geom  = feat.get("geometry", {})
            gtype = geom.get("type", "")
            rings = ([geom["coordinates"][0]] if gtype == "Polygon"
                     else [p[0] for p in geom["coordinates"]] if gtype == "MultiPolygon"
                     else [])
            if not rings: continue

            pts    = np.array(rings[0], dtype=float)
            cx_ann = pts[:, 0].mean()
            cy_ann = pts[:, 1].mean()

            try:
                img_rgb, r0, c0 = load_and_crop(tiff_path, cx_ann, cy_ann)
                if img_rgb.shape[:2] != (ROI_SIZE, ROI_SIZE): continue
                thumb = sk_resize(img_rgb, (THUMB_SIZE, THUMB_SIZE),
                                  anti_aliasing=True).astype(np.float32)
                all_imgs_list.append(thumb)
                grades_list.append(grade)
                centroids_list.append([cx_ann, cy_ann])
                source_list.append([test_idx, ann_idx])
            except Exception as e:
                continue

    all_imgs     = np.stack(all_imgs_list)               # (N, 128, 128, 3)
    grades_all   = np.array(grades_list, dtype=np.int32)
    centroids_xy = np.array(centroids_list, dtype=np.float64)
    source_arr   = np.array(source_list,   dtype=np.int32)

    np.savez_compressed(CACHE_IMAGES, thumbnails=all_imgs)
    np.savez(CACHE_META,
             grades=grades_all,
             centroids_xy=centroids_xy,
             source=source_arr)
    ts(f"  Saved {len(all_imgs)} ROI thumbnails → {CACHE_IMAGES}")
    ts(f"  Saved metadata → {CACHE_META}")

grade_counts = {g: int((grades_all==g).sum()) for g in [3,4,5]}
ts(f"  Grade counts: {grade_counts}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PH vectors
# ═══════════════════════════════════════════════════════════════════════════════
if not FORCE_PH and os.path.exists(CACHE_PH):
    ts("STAGE 2: Loading cached PH vectors …")
    ph_data = np.load(CACHE_PH)
    X_raw   = ph_data["X_raw"]
    ts(f"  Loaded X_raw {X_raw.shape}")
else:
    ts("STAGE 2: Computing PH for all ROIs …")
    vec_list = []
    n        = len(all_imgs)

    for i in range(n):
        # Reconstruct hema from thumbnail (sufficient for PH at PH_SIZE)
        img_uint8 = (all_imgs[i] * 255).astype(np.uint8)
        hema      = hematoxylin_norm(img_uint8)
        dgm0, dgm1 = compute_ph(hema)
        vec_list.append(ranked_vec(dgm0, dgm1))
        if (i+1) % 50 == 0:
            ts(f"  {i+1}/{n} done …")

    # Pad to uniform length
    min_len = min(len(v) for v in vec_list)
    X_raw   = np.stack([v[:min_len] for v in vec_list]).astype(np.float32)
    ts(f"  X_raw shape: {X_raw.shape}")

    np.savez_compressed(CACHE_PH, X_raw=X_raw)
    ts(f"  Saved PH vectors → {CACHE_PH}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — PCA + Bootstrap Ward clustering + t-SNE
# ═══════════════════════════════════════════════════════════════════════════════
if not FORCE_PIPELINE and os.path.exists(CACHE_PIPELINE):
    ts("STAGE 3: Loading cached pipeline results …")
    pipe      = np.load(CACHE_PIPELINE)
    Xpca_rep  = pipe["Xpca_rep"]
    Xtsne     = pipe["Xtsne"]
    clusters_rep = pipe["clusters_rep"]
    grades_rep   = pipe["grades_rep"]
    idx_rep      = pipe["idx_rep"]
    ts(f"  Loaded pipeline: {len(Xpca_rep)} ROIs in rep bootstrap")
else:
    ts("STAGE 3: Running bootstrap PCA + Ward clustering …")
    grades_u   = np.array(sorted(np.unique(grades_all)))
    min_count  = min((grades_all==g).sum() for g in grades_u)
    ts(f"  Grades: {grades_u}  min count per grade: {min_count}")

    rng          = np.random.default_rng(0)
    boot_results = []

    for b in range(N_BOOTSTRAP):
        idx_boot = []
        for g in grades_u:
            g_idx  = np.where(grades_all == g)[0]
            chosen = rng.choice(g_idx, size=min_count, replace=False)
            idx_boot.extend(chosen.tolist())
        idx_boot = np.array(idx_boot)

        X_b  = X_raw[idx_boot]
        gr_b = grades_all[idx_boot]

        pca  = PCA(n_components=min(N_PCA, X_b.shape[1], X_b.shape[0]-1))
        Xpca = pca.fit_transform(X_b)

        Z    = linkage(Xpca, method="ward")
        lbls = fcluster(Z, K_CLUSTERS, criterion="maxclust")

        boot_results.append((Xpca, lbls, gr_b, idx_boot))
        if (b+1) % 10 == 0:
            ts(f"  bootstrap {b+1}/{N_BOOTSTRAP}")

    ts("  Finding representative bootstrap …")
    all_cfrac = np.stack([cluster_fracs(r[1], r[2]) for r in boot_results])
    meta_cent = all_cfrac.mean(axis=0)
    dists     = [np.linalg.norm(all_cfrac[b] - meta_cent) for b in range(N_BOOTSTRAP)]
    rep_b     = int(np.argmin(dists))
    ts(f"  Representative bootstrap: {rep_b}")

    Xpca_rep, lbls_rep, gr_rep, idx_rep = boot_results[rep_b]

    # Order clusters by aggressiveness score
    cfrac_rep = cluster_fracs(lbls_rep, gr_rep)
    scores    = cfrac_rep @ np.array([3,4,5])
    order     = np.argsort(scores)
    remap     = {old+1: new+1 for new, old in enumerate(order)}
    clusters_rep = np.array([remap[l] for l in lbls_rep])
    grades_rep   = gr_rep

    ts("  Running t-SNE …")
    perp  = min(40, len(Xpca_rep)//4)
    tsne  = TSNE(n_components=2, perplexity=perp, random_state=7,
                 max_iter=2000, learning_rate="auto", init="pca")
    Xtsne = tsne.fit_transform(Xpca_rep)

    np.savez(CACHE_PIPELINE,
             Xpca_rep=Xpca_rep,
             Xtsne=Xtsne,
             clusters_rep=clusters_rep,
             grades_rep=grades_rep,
             idx_rep=idx_rep)
    ts(f"  Saved pipeline → {CACHE_PIPELINE}")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Find representative ROIs per (cluster, grade)
# ═══════════════════════════════════════════════════════════════════════════════
ts("STAGE 4: Finding representative ROIs …")
pca_all  = PCA(n_components=N_PCA)
Xpca_all = pca_all.fit_transform(X_raw)

centroids_pca = np.zeros((K_CLUSTERS, N_PCA))
for c in range(1, K_CLUSTERS+1):
    mask = clusters_rep == c
    centroids_pca[c-1] = Xpca_rep[mask].mean(axis=0)

rep_rois = {}
for c in range(1, K_CLUSTERS+1):
    for g in [3,4,5]:
        g_idx = np.where(grades_all == g)[0]
        if len(g_idx) == 0:
            rep_rois[(c,g)] = None
            continue
        dists = np.linalg.norm(Xpca_all[g_idx] - centroids_pca[c-1], axis=1)
        best  = g_idx[np.argmin(dists)]
        rep_rois[(c,g)] = all_imgs[best]   # float32 [0,1]

# Grade fractions + bootstrap error bars
cfrac = cluster_fracs(clusters_rep, grades_rep)
rng2  = np.random.default_rng(42)
boot_cf = []
for _ in range(200):
    idx = rng2.integers(0, len(grades_rep), size=len(grades_rep))
    boot_cf.append(cluster_fracs(clusters_rep[idx], grades_rep[idx]))
cfrac_std = np.std(boot_cf, axis=0)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Plot
# ═══════════════════════════════════════════════════════════════════════════════
ts("STAGE 5: Plotting …")

CLUSTER_COLORS = ["#e41a1c","#ff7f00","#4daf4a","#377eb8","#984ea3","#a65628"]
GRADE_MARKERS  = {3:"o", 4:"^", 5:"s"}
GRADE_FILL     = {3:"#2c3e50", 4:"#7f8c8d", 5:"#bdc3c7"}
roman          = ["i","ii","iii","iv","v","vi"]

fig = plt.figure(figsize=(22, 11))
fig.patch.set_facecolor("white")
gs  = GridSpec(5, 8, figure=fig,
               height_ratios=[0.15, 1, 1, 1, 0.8],
               width_ratios=[1.8, 0.15, 1,1,1,1,1,1],
               wspace=0.08, hspace=0.12,
               left=0.03, right=0.98, top=0.92, bottom=0.04)

# t-SNE
ax_tsne = fig.add_subplot(gs[0:5, 0])
for c in range(1, K_CLUSTERS+1):
    col = CLUSTER_COLORS[c-1]
    for g in [3,4,5]:
        mask = (clusters_rep==c) & (grades_rep==g)
        if mask.sum()==0: continue
        ax_tsne.scatter(Xtsne[mask,0], Xtsne[mask,1],
                        c=col, marker=GRADE_MARKERS[g],
                        s=30, alpha=0.85, linewidths=0.3,
                        edgecolors="white", zorder=3)
for c in range(1, K_CLUSTERS+1):
    mask = clusters_rep == c
    mx, my = Xtsne[mask,0].mean(), Xtsne[mask,1].mean()
    ax_tsne.text(mx, my, roman[c-1],
                fontsize=11, fontweight="bold", ha="center", va="center",
                color=CLUSTER_COLORS[c-1],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                zorder=5)
ax_tsne.set_xticks([]); ax_tsne.set_yticks([])
ax_tsne.spines[["top","right","bottom","left"]].set_visible(False)
ax_tsne.legend(handles=[
    plt.Line2D([0],[0],marker="o",color="grey",ls="",ms=7,label="G3"),
    plt.Line2D([0],[0],marker="^",color="grey",ls="",ms=7,label="G4"),
    plt.Line2D([0],[0],marker="s",color="grey",ls="",ms=7,label="G5"),
], fontsize=8, loc="lower left", framealpha=0.8, edgecolor="grey")
ax_tsne.set_title("t-SNE", fontsize=10, fontweight="bold", pad=3)

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
            ax.text(0.5, 0.5, "—", ha="center", va="center",
                   transform=ax.transAxes, color="#aaaaaa", fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])

# Grade row labels
for ri, g in enumerate([3,4,5]):
    ax_gl = fig.add_subplot(gs[ri+1, 1])
    ax_gl.axis("off")
    ax_gl.text(0.5, 0.5, f"G{g}", transform=ax_gl.transAxes,
               fontsize=9, va="center", ha="center",
               color="#555555", style="italic", rotation=90)

# Histograms
x = np.arange(3); w = 0.55
for ci, c in enumerate(range(1, K_CLUSTERS+1)):
    ax = fig.add_subplot(gs[4, ci+2])
    ax.bar(x, cfrac[c-1], w,
           color=[GRADE_FILL[g] for g in [3,4,5]],
           yerr=cfrac_std[c-1], capsize=2,
           error_kw={"elinewidth":0.8, "ecolor":"#333"})
    ax.set_xticks(x)
    ax.set_xticklabels(["G3","G4","G5"], fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelsize=6)
    ax.spines[["top","right"]].set_visible(False)
    if ci == 0: ax.set_ylabel("Fraction", fontsize=7)
    else:       ax.set_yticklabels([])

fig.suptitle(
    f"PH reveals architectural subpatterns — Subset1\n"
    f"({len(X_raw)} ROIs  |  "
    f"G3={grade_counts[3]}  G4={grade_counts[4]}  G5={grade_counts[5]}  |  "
    f"k={K_CLUSTERS} clusters  |  {N_BOOTSTRAP} bootstraps)",
    fontsize=12, fontweight="bold", y=0.97
)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
ts(f"Saved figure → {OUT_PATH}")

# Summary table
print("\nCluster composition (representative bootstrap):")
print(f"{'':5} {'N':>5}  {'G3%':>6}  {'G4%':>6}  {'G5%':>6}  Dominant")
print("─"*48)
for c in range(1, K_CLUSTERS+1):
    mask = clusters_rep == c
    n    = mask.sum()
    f    = cfrac[c-1]
    dom  = ["G3","G4","G5"][np.argmax(f)]
    print(f"  {roman[c-1]:<3}  {n:>5}  {f[0]*100:>5.0f}%  {f[1]*100:>5.0f}%  {f[2]*100:>5.0f}%  {dom}")

print(f"\nCache files in: {CACHE_DIR}")
for f in [CACHE_IMAGES, CACHE_META, CACHE_PH, CACHE_PIPELINE]:
    size_mb = os.path.getsize(f) / 1e6 if os.path.exists(f) else 0
    print(f"  {os.path.basename(f):<22} {size_mb:6.1f} MB")