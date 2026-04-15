"""
Replicate Figure 4 Dr.Wenk's paper

Pipeline:
  1. Load all TIFF+GeoJSON pairs
  2. For each annotated ROI: colour deconvolution → PH (gudhi cubical)
  3. Build ranked persistence vectors (H0 + H1 concatenated)
  4. PCA → 6 components
  5. Bootstrap sampling (balanced by Gleason grade)
  6. Hierarchical Ward clustering (k chosen by Gap statistic)
  7. t-SNE visualisation coloured by cluster, shaped by grade
  8. Save figure + cluster summary

Usage:
  python figure4_pipeline.py

Expects files:
  .../Subset1/Subset1_Test_{1..N}.tiff
  .../Subset1_annot/Subset1_Test_{1..N}.geojson
"""

import json, warnings, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import color, filters
from skimage.transform import resize as sk_resize
from skimage.filters import threshold_otsu
import tifffile
import gudhi
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
BASE_TIFF   = "magicScan_TDA_ML/Subset1/Subset1_Test_{}.tiff"
BASE_GEOJSON= "magicScan_TDA_ML/Subset1_annot/Subset1_Test_{}.geojson"
OUT_PATH    = "figure4_replicate.png"
OUT_DATA    = "figure4_data.npz"

ROI_SIZE    = 512       # pixels
PH_SIZE     = 64        # downsample for PH (faster; 64×64 sufficient for topology)
N_BOOTSTRAP = 30        # paper uses 100; use 30 for speed, increase if needed
N_PCA       = 6
K_CLUSTERS  = 6         # paper found k=6 optimal
GAUSS_SIGMA = 1.0
N_TEST_FILES= 30        # how many Subset1_Test_N.tiff files to try (stops at missing)

# ── helpers ───────────────────────────────────────────────────────────────────
def load_and_crop(tiff_path, cx, cy, size=ROI_SIZE):
    """Load a centre crop of size×size around (cx,cy) from a TIFF."""
    raw = tifffile.imread(tiff_path)
    if raw.ndim == 4: raw = raw[0]
    if raw.ndim == 3 and raw.shape[0] in (1,3,4): raw = np.moveaxis(raw, 0, -1)
    if raw.shape[-1] == 4: raw = raw[..., :3]
    if raw.dtype != np.uint8: raw = (raw / raw.max() * 255).astype(np.uint8)
    H, W = raw.shape[:2]
    half = size // 2
    cy_c = max(half, min(H-half, int(cy)))
    cx_c = max(half, min(W-half, int(cx)))
    r0,r1 = cy_c-half, cy_c+half
    c0,c1 = cx_c-half, cx_c+half
    return raw[r0:r1, c0:c1], r0, c0

def hematoxylin_norm(img_rgb):
    """Return normalised hematoxylin channel in [0,1]."""
    hed = color.rgb2hed(img_rgb)
    h   = hed[:,:,0]
    hmin, hmax = np.percentile(h,1), np.percentile(h,99)
    return np.clip((h-hmin)/(hmax-hmin+1e-9), 0, 1).astype(np.float64)

def compute_ph(hema_norm):
    """Compute H0+H1 persistence diagram via gudhi cubical complex."""
    small = sk_resize(hema_norm, (PH_SIZE,PH_SIZE), anti_aliasing=True).astype(np.float64)
    small = (small-small.min())/(small.max()-small.min()+1e-9)
    cc    = gudhi.CubicalComplex(top_dimensional_cells=small)
    cc.compute_persistence()
    def get_fin(dim):
        arr = np.array(cc.persistence_intervals_in_dimension(dim), dtype=float)
        if arr.size == 0: return np.zeros((0,2))
        return arr[np.isfinite(arr).all(axis=1)]
    return get_fin(0), get_fin(1)

def ranked_persistence_vector(dgm0, dgm1):
    """Sort each diagram by persistence descending, concatenate."""
    def sorted_pers(dgm):
        if len(dgm) == 0: return np.array([])
        p = dgm[:,1] - dgm[:,0]
        return np.sort(p)[::-1]
    return np.concatenate([sorted_pers(dgm0), sorted_pers(dgm1)])

def get_label(props):
    """Extract Gleason grade label from GeoJSON properties."""
    label = (
        (props.get("classification") or {}).get("name")
        or props.get("label") or props.get("class")
        or props.get("objectType") or props.get("type") or "unknown"
    )
    return str(label).strip()

def label_to_grade(label):
    """Map label string to integer grade (3,4,5) or None."""
    l = label.upper()
    for g in ["G3","G4","G5","GLEASON3","GLEASON4","GLEASON5",
              "GRADE3","GRADE4","GRADE5","3","4","5"]:
        if g in l.replace(" ",""):
            for digit in ["3","4","5"]:
                if digit in g:
                    return int(digit)
    return None

def gap_statistic(X, k_range, B=10):
    """Simple gap statistic to choose optimal k."""
    from scipy.cluster.hierarchy import linkage, fcluster
    gaps = []
    for k in k_range:
        Z   = linkage(X, method="ward")
        lbl = fcluster(Z, k, criterion="maxclust")
        # within-cluster dispersion
        wk = 0.0
        for c in range(1, k+1):
            pts = X[lbl==c]
            if len(pts) > 1:
                wk += np.sum(pdist(pts)**2) / (2*len(pts))
        # reference distribution
        ref_wks = []
        rng = np.random.default_rng(42)
        for _ in range(B):
            Xr  = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            Zr  = linkage(Xr, method="ward")
            lr  = fcluster(Zr, k, criterion="maxclust")
            wkr = 0.0
            for c in range(1, k+1):
                pts = Xr[lr==c]
                if len(pts) > 1:
                    wkr += np.sum(pdist(pts)**2) / (2*len(pts))
            ref_wks.append(np.log(wkr+1e-9))
        gaps.append(np.mean(ref_wks) - np.log(wk+1e-9))
    return gaps

# ── 1. Collect all ROIs ───────────────────────────────────────────────────────
print("="*60)
print("STEP 1: Collecting ROIs from all test files")
print("="*60)

all_vectors = []   # ranked persistence vectors
all_grades  = []   # integer grade (3,4,5)
all_labels  = []   # string label
all_imgs    = []   # small RGB crops for visualisation
all_source  = []   # (test_idx, annotation_idx)

for test_idx in range(1, N_TEST_FILES+1):
    tiff_path    = BASE_TIFF.format(test_idx)
    geojson_path = BASE_GEOJSON.format(test_idx)
    if not os.path.exists(tiff_path) or not os.path.exists(geojson_path):
        continue
    print(f"  Processing Test_{test_idx} …", end="", flush=True)

    with open(geojson_path) as fj:
        gj = json.load(fj)
    features_gj = gj.get("features", [])

    n_rois = 0
    for ann_idx, feat in enumerate(features_gj):
        geom  = feat.get("geometry", {})
        props = feat.get("properties", {})
        label = get_label(props)
        grade = label_to_grade(label)
        if grade is None:
            continue   # skip unannotated or unrecognised

        gtype = geom.get("type","")
        rings = ([geom["coordinates"][0]] if gtype=="Polygon"
                 else [p[0] for p in geom["coordinates"]] if gtype=="MultiPolygon"
                 else [])
        if not rings: continue

        # Centroid of annotation
        pts   = np.array(rings[0], dtype=float)
        cx_ann = pts[:,0].mean()
        cy_ann = pts[:,1].mean()

        try:
            img_rgb, r0, c0 = load_and_crop(tiff_path, cx_ann, cy_ann)
            if img_rgb.shape[:2] != (ROI_SIZE, ROI_SIZE): continue

            hema = hematoxylin_norm(img_rgb)
            dgm0, dgm1 = compute_ph(hema)
            vec = ranked_persistence_vector(dgm0, dgm1)
            if len(vec) == 0: continue

            all_vectors.append(vec)
            all_grades.append(grade)
            all_labels.append(label)
            all_imgs.append(sk_resize(img_rgb, (64,64), anti_aliasing=True))
            all_source.append((test_idx, ann_idx))
            n_rois += 1
        except Exception as e:
            continue

    print(f" {n_rois} ROIs")

print(f"\nTotal ROIs: {len(all_vectors)}")
grades_arr = np.array(all_grades)
for g in [3,4,5]:
    print(f"  G{g}: {(grades_arr==g).sum()}")

if len(all_vectors) == 0:
    print("ERROR: No ROIs collected. Check file paths and GeoJSON label format.")
    print("Printing sample labels from last file for debugging:")
    with open(geojson_path) as fj:
        gj2 = json.load(fj)
    for f in gj2.get("features",[])[:5]:
        print("  props:", f.get("properties",{}))
    exit(1)

# ── 2. Pad vectors to uniform length ─────────────────────────────────────────
print("\nSTEP 2: Building uniform ranked persistence matrix")
min_len = min(len(v) for v in all_vectors)
print(f"  min vector length: {min_len}")
X_raw = np.stack([v[:min_len] for v in all_vectors])  # (N, min_len)
print(f"  matrix shape: {X_raw.shape}")

# ── 3. Bootstrap + PCA + Ward + t-SNE ────────────────────────────────────────
print("\nSTEP 3: Bootstrap pipeline")

grades_u = np.unique(grades_arr)
min_count = min((grades_arr==g).sum() for g in grades_u)
print(f"  grades present: {grades_u}  min count: {min_count}")
if min_count < 5:
    print("  WARNING: very few ROIs per grade — results may not be meaningful")

# Run bootstraps
rng = np.random.default_rng(0)
boot_results = []   # list of (pca_coords, cluster_labels, grade_labels)

for b in range(N_BOOTSTRAP):
    # Balanced subsample
    idx_boot = []
    for g in grades_u:
        g_idx = np.where(grades_arr == g)[0]
        chosen = rng.choice(g_idx, size=min_count, replace=False)
        idx_boot.extend(chosen.tolist())
    idx_boot = np.array(idx_boot)

    X_boot  = X_raw[idx_boot]
    gr_boot = grades_arr[idx_boot]

    # PCA
    pca  = PCA(n_components=min(N_PCA, X_boot.shape[1], X_boot.shape[0]-1))
    Xpca = pca.fit_transform(X_boot)

    # Ward clustering
    Z    = linkage(Xpca, method="ward")
    lbls = fcluster(Z, K_CLUSTERS, criterion="maxclust")

    boot_results.append((Xpca, lbls, gr_boot, idx_boot))
    if (b+1) % 5 == 0:
        print(f"  bootstrap {b+1}/{N_BOOTSTRAP} done")

# ── 4. Meta-clustering to find representative bootstrap ──────────────────────
print("\nSTEP 4: Finding representative bootstrap")

# For each bootstrap, represent each cluster as (frac_G3, frac_G4, frac_G5)
def cluster_fracs(lbls, grades, k=K_CLUSTERS, grade_list=[3,4,5]):
    vecs = []
    for c in range(1, k+1):
        mask = lbls == c
        total = mask.sum() + 1e-9
        vecs.append([((grades[mask]==g).sum()/total) for g in grade_list])
    return np.array(vecs)   # (k, 3)

# Collect cluster fraction vectors across all bootstraps
all_cfrac = np.stack([cluster_fracs(r[1], r[2]) for r in boot_results])
# all_cfrac shape: (N_BOOTSTRAP, K_CLUSTERS, 3)

# Meta-centroid: mean across bootstraps
meta_centroid = all_cfrac.mean(axis=0)  # (K_CLUSTERS, 3)

# Find bootstrap closest to meta-centroid
dists = [np.linalg.norm(all_cfrac[b] - meta_centroid) for b in range(N_BOOTSTRAP)]
rep_b = int(np.argmin(dists))
print(f"  Representative bootstrap: {rep_b}")

Xpca_rep, lbls_rep, gr_rep, idx_rep = boot_results[rep_b]

# ── 5. t-SNE on representative bootstrap ─────────────────────────────────────
print("\nSTEP 5: t-SNE")
perp = min(47, len(Xpca_rep)//3)
tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
Xtsne = tsne.fit_transform(Xpca_rep)

# ── 6. Order clusters by Gleason grade composition ───────────────────────────
# Sort clusters so that cluster 1 = most G3-dominated ... cluster k = most G5
cfrac_rep = cluster_fracs(lbls_rep, gr_rep)
# Score = weighted sum 3*G3_frac + 4*G4_frac + 5*G5_frac
scores = cfrac_rep @ np.array([3,4,5])
order  = np.argsort(scores)          # ascending aggressiveness
remap  = {old+1: new+1 for new, old in enumerate(order)}
lbls_ordered = np.array([remap[l] for l in lbls_rep])

# ── 7. Figure ─────────────────────────────────────────────────────────────────
print("\nSTEP 6: Plotting")

CLUSTER_COLORS = plt.cm.tab10(np.linspace(0, 0.6, K_CLUSTERS))
GRADE_MARKERS  = {3:"o", 4:"^", 5:"s"}
GRADE_NAMES    = {3:"G3", 4:"G4", 5:"G5"}

# Find representative ROI image per cluster (closest to cluster centroid in PCA space)
rep_roi_imgs = {}
for c in range(1, K_CLUSTERS+1):
    mask = lbls_ordered == c
    if mask.sum() == 0: continue
    pts  = Xpca_rep[mask]
    cent = pts.mean(axis=0)
    dists_c = np.linalg.norm(pts - cent, axis=1)
    local_idx = np.argmin(dists_c)
    global_idx = idx_rep[np.where(mask)[0][local_idx]]
    rep_roi_imgs[c] = all_imgs[global_idx]

# Figure layout: left = t-SNE, right = grid of representative ROIs + histograms
n_row_roi = max(len(grades_u), 3)   # rows of example ROIs
fig = plt.figure(figsize=(20, 10))
fig.patch.set_facecolor("white")

from matplotlib.gridspec import GridSpec
gs = GridSpec(n_row_roi+2, K_CLUSTERS+1, figure=fig,
              wspace=0.15, hspace=0.25,
              left=0.04, right=0.98, top=0.93, bottom=0.05)

# t-SNE panel (left, spans all rows)
ax_tsne = fig.add_subplot(gs[:n_row_roi, 0])

for c in range(1, K_CLUSTERS+1):
    col = CLUSTER_COLORS[c-1]
    for g in grades_u:
        mask = (lbls_ordered==c) & (gr_rep==g)
        if mask.sum() == 0: continue
        ax_tsne.scatter(Xtsne[mask,0], Xtsne[mask,1],
                        c=[col], marker=GRADE_MARKERS[g],
                        s=25, alpha=0.8, linewidths=0.2,
                        edgecolors="white")
    # Label cluster
    mask_c = lbls_ordered == c
    cx_ = Xtsne[mask_c,0].mean()
    cy_ = Xtsne[mask_c,1].mean()
    ax_tsne.text(cx_, cy_, str(["i","ii","iii","iv","v","vi"][c-1]),
                fontsize=10, fontweight="bold", ha="center", va="center",
                color=CLUSTER_COLORS[c-1])

ax_tsne.set_xticks([]); ax_tsne.set_yticks([])
ax_tsne.set_title("t-SNE", fontsize=10, fontweight="bold")

# Legend for grades
grade_legend = [mpatches.Patch(facecolor="grey",
                               label=f"G{g} = {GRADE_MARKERS[g]}")
                for g in grades_u]
ax_tsne.legend(handles=[
    plt.Line2D([0],[0],marker="o",color="grey",ls="",ms=6,label="G3"),
    plt.Line2D([0],[0],marker="^",color="grey",ls="",ms=6,label="G4"),
    plt.Line2D([0],[0],marker="s",color="grey",ls="",ms=6,label="G5"),
], fontsize=7, loc="lower left")

# Column headers
roman = ["i","ii","iii","iv","v","vi"]
for c in range(1, K_CLUSTERS+1):
    ax_h = fig.add_subplot(gs[0, c])
    ax_h.axis("off")
    ax_h.text(0.5, 0.5, roman[c-1], transform=ax_h.transAxes,
              fontsize=14, fontweight="bold", ha="center", va="center",
              color=CLUSTER_COLORS[c-1])

# Representative ROI images — one row per grade
for row_idx, g in enumerate(sorted(grades_u)):
    for c in range(1, K_CLUSTERS+1):
        ax = fig.add_subplot(gs[row_idx, c])
        # Find a ROI of grade g in cluster c
        mask = (lbls_ordered==c) & (gr_rep==g)
        if mask.sum() > 0:
            pts   = Xpca_rep[mask]
            cent  = pts.mean(axis=0)
            dist  = np.linalg.norm(pts - cent, axis=1)
            li    = np.argmin(dist)
            gi    = idx_rep[np.where(mask)[0][li]]
            ax.imshow(all_imgs[gi])
            for sp in ax.spines.values():
                sp.set_edgecolor(CLUSTER_COLORS[c-1])
                sp.set_linewidth(2)
        else:
            ax.set_facecolor("#f0f0f0")
            ax.text(0.5,0.5,"—",ha="center",va="center",
                   transform=ax.transAxes, color="grey")
        ax.set_xticks([]); ax.set_yticks([])

# Grade labels on left of ROI rows
for row_idx, g in enumerate(sorted(grades_u)):
    ax_lbl = fig.add_subplot(gs[row_idx, 0])
    ax_lbl.axis("off")
    ax_lbl.text(0.85, 0.5, f"G{g}", transform=ax_lbl.transAxes,
                fontsize=9, va="center", ha="right",
                color="grey", style="italic")

# Histograms: grade distribution per cluster
ax_bar_row = n_row_roi
grade_colors = {3:"#2c3e50", 4:"#7f8c8d", 5:"#bdc3c7"}
for c in range(1, K_CLUSTERS+1):
    ax = fig.add_subplot(gs[ax_bar_row, c])
    fracs = cfrac_rep[c-1]   # [G3_frac, G4_frac, G5_frac]
    x     = np.arange(len(grades_u))
    bars  = ax.bar(x, fracs[:len(grades_u)],
                   color=[grade_colors.get(g,"grey") for g in sorted(grades_u)],
                   width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{g}" for g in sorted(grades_u)], fontsize=6)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", labelsize=6)
    if c == 1: ax.set_ylabel("Fraction", fontsize=7)

fig.suptitle("PH reveals architectural subpatterns — Subset1\n"
             f"({len(all_vectors)} ROIs, {len(grades_u)} grades, "
             f"k={K_CLUSTERS} clusters, {N_BOOTSTRAP} bootstraps)",
             fontsize=12, fontweight="bold")

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved figure → {OUT_PATH}")

# Save data for further analysis
np.savez(OUT_DATA,
         X_raw=X_raw, grades=grades_arr,
         Xpca=Xpca_rep, Xtsne=Xtsne,
         clusters=lbls_ordered, grades_rep=gr_rep)
print(f"Saved data   → {OUT_DATA}")

# Print cluster summary
print("\nCluster summary (representative bootstrap):")
print(f"{'Cluster':<10} {'N':<6} {'G3%':<8} {'G4%':<8} {'G5%':<8} {'Dominant'}")
print("-"*50)
for c in range(1, K_CLUSTERS+1):
    mask  = lbls_ordered == c
    n     = mask.sum()
    fracs = cfrac_rep[c-1]
    dom   = ["G3","G4","G5"][np.argmax(fracs[:len(grades_u)])]
    g3 = f"{fracs[0]*100:.0f}%" if 3 in grades_u else "—"
    g4 = f"{fracs[1]*100:.0f}%" if 4 in grades_u else "—"
    g5 = f"{fracs[2]*100:.0f}%" if 5 in grades_u else "—"
    print(f"{roman[c-1]:<10} {n:<6} {g3:<8} {g4:<8} {g5:<8} {dom}")