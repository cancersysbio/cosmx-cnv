"""
Step 2 - infer subclones and their CN profiles from single-cell counts.

Fits the Pyro mixture model with spatial (cell-cell) and genomic (gene-gene)
smoothness penalties, then compares the result with the simulated truth.

Runs on simulated data in ~30 s on a laptop CPU:  python examples/02_cnv_inference.py
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

import cosmx_cnv as cc

warnings.filterwarnings("ignore", message="Found plate statements in guide but not model")
pyro.set_rng_seed(0)
out = Path("outputs")
out.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data ---------------------------------------------------------------------
# adata.obs needs: 'patient.id', 'core_id' (categoricals), 'CenterX_global_px',
# 'CenterY_global_px'. gene_coords: gene_symbol, chromosome (str), start, end.
sim = cc.simulate_dataset(n_patients=3, n_cores_per_patient=2, n_cells_per_core=300, n_genes=300, seed=0)
adata, gene_coords = sim["adata"], sim["gene_coords"]

# --- Prepare per-chromosome data, then build count matrix and neighbour graphs -
chromosomes = sorted(gene_coords["chromosome"].unique(), key=int)
data_dict = {c: cc.prepare_chromosome_data(c, {"Cancer": adata}, gene_coords) for c in chromosomes}
md = cc.prepare_data(data_dict, device=device, cell_distance_threshold=2000, gene_distance_threshold=20e6)

# --- Fit ----------------------------------------------------------------------
config = cc.CNVModelConfig(
    num_components=3,
    spatial_penalty=1.0,
    gene_proximity_penalty=80,
    spatial_decay_scale=400.0,  # pixels
    gene_decay_scale=3e6,       # base pairs
    device=device,
)
model = cc.CNVModel(config)
model.fit(
    md["y_counts"], num_iterations=1000, lr=0.05,
    r_init=md["y_counts"].log1p().mean(dim=0, keepdim=True),
    cell_neighbours_adj=md["cell_adjacency"], cell_neighbours_dist=md["cell_distances"],
    gene_neighbours_adj=md["gene_adjacency"], gene_neighbours_dist=md["gene_distances"],
    core_ids=md["cell_core_ids"], patient_ids=md["cell_patient_ids"],
)

# --- Results ------------------------------------------------------------------
params = model.get_parameters()
membership = model.get_membership_probs().cpu().numpy()  # cells x K, core-sorted order
clone_pred = membership.argmax(1)
s_loc = params["s_loc"].cpu().numpy()                    # patients x K x genes (log scale)

obs = adata.obs.loc[md["cell_names"]]                    # align obs to model cell order
patients = list(obs["patient.id"].cat.categories)
for pid in patients:
    m = (obs["patient.id"] == pid).to_numpy()
    print(f"{pid}: ARI vs true clones = {adjusted_rand_score(obs['true_clone'][m], clone_pred[m]):.2f}")

cc.create_combined_clone_state_table(params, data_dict, gene_coords).to_csv(out / "clone_cn_profiles.csv", index=False)

# --- Plot patient 1: spatial clones and CN profiles vs truth -------------------
p = 0
m = (obs["patient.id"] == patients[p]).to_numpy()
true_k, pred_k = obs["true_clone"].to_numpy()[m], clone_pred[m]
conf = np.array([[np.sum((true_k == i) & (pred_k == j)) for j in range(config.num_components)]
                 for i in range(config.num_components)])
_, match = linear_sum_assignment(-conf)                  # true clone i <-> component match[i]

K = config.num_components
clone_colors = plt.get_cmap("Set2")(np.arange(K))
fig = plt.figure(figsize=(12, 4.8))
outer = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.98], wspace=0.18)
left, right = outer[0].subgridspec(2, 1, hspace=0.25), outer[1].subgridspec(K, 1, hspace=0.15)
xy = obs.loc[m, ["CenterX_global_px", "CenterY_global_px"]].to_numpy()
for i, (labels, title) in enumerate([(true_k, "True clones"), (np.argsort(match)[pred_k], "Inferred clones")]):
    ax = fig.add_subplot(left[i])
    ax.scatter(xy[:, 0], xy[:, 1], c=clone_colors[labels], s=4)
    ax.set_title(f"{title} ({patients[p]}, 2 cores)", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

# One track per clone: inferred log2 CN per gene (dots, blue = loss, red = gain), truth in black
true_cn = sim["true_cn"][patients[p]]
gene_order = [list(gene_coords["gene_symbol"]).index(g) for g in md["gene_names"]]
chrom = gene_coords["chromosome"].to_numpy()[gene_order]
bounds = np.flatnonzero(chrom[1:] != chrom[:-1]) + 1
x = np.arange(len(gene_order))
for k in range(K):
    ax = fig.add_subplot(right[k], sharex=ax if k else None)
    inferred = s_loc[p, match[k]] / np.log(2)      # model works in natural log
    truth = true_cn[k, gene_order]
    ax.scatter(x, truth, s=1, color="k", zorder=1)
    ax.scatter(x, inferred, s=3, c=inferred, cmap="RdBu_r", vmin=-1, vmax=1, zorder=2)
    lim = max(0.6, 1.15 * np.abs(np.r_[truth, inferred]).max())
    ax.set_ylim(-lim, lim)
    for b in bounds:
        ax.axvline(b - 0.5, color="0.85", lw=0.5, zorder=0)
    ax.set_ylabel(f"clone {k}" + (" (neutral)" if k == 0 else ""), color=clone_colors[k] * 0.75, fontsize=9, fontweight="bold")
    ax.spines[["right", "top"]].set_visible(False)
    if k == 0:
        ax.set_title("log2 CN ratio per gene: true (black) vs inferred")
    if k < K - 1:
        ax.tick_params(labelbottom=False)
ax.set_xticks((np.r_[0, bounds] + np.r_[bounds, len(x)]) / 2, [f"chr{c}" for c in chrom[np.r_[0, bounds]]], fontsize=8)
ax.tick_params(axis="x", length=0)
fig.savefig(out / "cnv_inference.png", dpi=150, bbox_inches="tight")
print(f"Wrote {out}/clone_cn_profiles.csv, cnv_inference.png")
