"""
Real-data example: two de-identified breast-tumour TMA cores (CosMX 6k panel),
fitted jointly with a pool of benign epithelial cells from other patients.

Each core holds an aneuploid tumour clone and a spatially coherent CN-neutral
clone that coincides with the patches annotated as benign epithelium.

The pool (patient 'normal_pool') anchors the per-gene baseline to normal
expression; it is fitted like any other sample, nothing is pinned.

The data file is not distributed with the repository; pass its path. The
reference result used K=4, 10000 iterations, lr 0.01 (~4 min on a GPU):

    python examples/03_real_data.py examples/data/madrid_example.h5ad --num_iterations 10000 --lr 0.01
"""
import argparse
import contextlib
import io
import warnings
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from matplotlib.colors import LinearSegmentedColormap

import cosmx_cnv as cc

warnings.filterwarnings("ignore", message="Found plate statements in guide but not model")
# blue = loss, red = gain; neutral is grey (not white) so CN-neutral clones stay visible
CN_CMAP = LinearSegmentedColormap.from_list("cn", ["#2166ac", "#bdbdbd", "#b2182b"])

p = argparse.ArgumentParser()
p.add_argument("h5ad", nargs="?", default="examples/data/madrid_example.h5ad")
p.add_argument("--num_components", type=int, default=4)
p.add_argument("--num_iterations", type=int, default=2000)
p.add_argument("--lr", type=float, default=2e-2)
args = p.parse_args()
pyro.set_rng_seed(0)
out = Path("outputs")
out.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data ---------------------------------------------------------------------
adata = ad.read_h5ad(args.h5ad)
gene_coords = (adata.var[["chromosome", "start", "end"]].rename_axis("gene_symbol")
               .reset_index().astype({"chromosome": str}))
with contextlib.redirect_stdout(io.StringIO()):  # silence per-chromosome logging
    data_dict = {c: cc.prepare_chromosome_data(c, {"Cancer": adata}, gene_coords)
                 for c in gene_coords["chromosome"].unique()}
    md = cc.prepare_data(data_dict, device=device)
md["y_counts"] = md["y_counts"].round().clamp(min=0)

# --- Fit (penalties as in the cohort-wide runs) --------------------------------
config = cc.CNVModelConfig(
    num_components=args.num_components, spatial_penalty=1.0, gene_proximity_penalty=80.0,
    spatial_decay_scale=400.0, gene_decay_scale=3e6, device=device,
)
model = cc.CNVModel(config)
model.fit(
    md["y_counts"], num_iterations=args.num_iterations, lr=args.lr,
    r_init=md["y_counts"].log1p().mean(dim=0, keepdim=True),
    cell_neighbours_adj=md["cell_adjacency"], cell_neighbours_dist=md["cell_distances"],
    gene_neighbours_adj=md["gene_adjacency"], gene_neighbours_dist=md["gene_distances"],
    core_ids=md["cell_core_ids"], patient_ids=md["cell_patient_ids"],
)

# --- Results ------------------------------------------------------------------
clone = model.get_membership_probs().argmax(1).cpu().numpy()
s = model.get_parameters()["s_loc"].cpu().numpy() / np.log(2)   # patients x K x genes, log2
obs = adata.obs.loc[md["cell_names"]]                             # model (core-sorted) order
pidx = {p: i for i, p in enumerate(obs["patient.id"].cat.categories)}
benign = obs["annotation"].eq("benign epithelium").to_numpy()
chrom = np.array(md["chr_ids"])
patients = [p for p in pidx if p != "normal_pool"]

for pid in patients:
    m = (obs["patient.id"] == pid).to_numpy()
    frac = np.bincount(clone[m], minlength=config.num_components) / m.sum()
    print(f"{pid}: {m.sum()} cells")
    for k in np.flatnonzero(frac >= 0.03):
        print(f"  component {k}: {frac[k]:5.1%} of cells, mean |log2 CN| {np.abs(s[pidx[pid], k]).mean():.2f}, "
              f"{benign[m & (clone == k)].mean():4.0%} annotated benign")

# --- Plot: clones in space, and each clone's CN profile (one dot per gene) -----
fig = plt.figure(figsize=(15, 3.4 * len(patients)))
for sf, pid in zip(np.atleast_1d(fig.subfigures(len(patients), 1)), patients):
    m = (obs["patient.id"] == pid).to_numpy()
    frac = np.bincount(clone[m], minlength=config.num_components) / m.sum()
    rows = list(np.flatnonzero(frac >= 0.03))
    gs = sf.add_gridspec(len(rows), 3, width_ratios=[1, 1, 4], hspace=0.15)

    xy = obs.loc[m, ["CenterX_global_px", "CenterY_global_px"]].to_numpy()
    ax = sf.add_subplot(gs[:, 0])
    for k in rows:
        ax.scatter(*xy[clone[m] == k].T, s=1, color=f"C{k}", label=f"clone {k} ({frac[k]:.0%})")
    ax.legend(fontsize=7, markerscale=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0))
    ax.set_title(f"{pid}: inferred clones", fontsize=9)
    ax2 = sf.add_subplot(gs[:, 1])
    ax2.scatter(*xy.T, s=1, c=np.where(benign[m], "tab:green", "lightgrey"))
    ax2.set_title("annotated benign epithelium", fontsize=9)
    for a in (ax, ax2):
        a.set_aspect("equal")
        a.axis("off")

    edges = np.r_[0, np.flatnonzero(chrom[1:] != chrom[:-1]) + 1, len(chrom)]
    for i, k in enumerate(rows):
        a = sf.add_subplot(gs[i, 2])
        y = s[pidx[pid], k]
        a.scatter(np.arange(len(y)), y, c=y, cmap=CN_CMAP, vmin=-1, vmax=1, s=2, linewidths=0)
        a.axhline(0, color="grey", lw=0.4)
        for e in edges[1:-1]:
            a.axvline(e - 0.5, color="grey", lw=0.3)
        a.set_ylim(-1.6, 1.6)
        a.set_xlim(0, len(y))
        a.set_ylabel(f"clone {k}", fontsize=8)
        a.tick_params(labelsize=7)
        a.spines[["right", "top"]].set_visible(False)
        a.set_xticks([] if i < len(rows) - 1 else (edges[:-1] + edges[1:]) / 2)
        if i == len(rows) - 1:
            a.set_xticklabels(chrom[edges[:-1]], fontsize=6)
            a.set_xlabel("chromosome (log2 CN: red = gain, blue = loss)", fontsize=8)
fig.savefig(out / "real_data.png", dpi=150, bbox_inches="tight")
print(f"Wrote {out}/real_data.png")
