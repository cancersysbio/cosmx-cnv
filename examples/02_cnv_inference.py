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

fig = plt.figure(figsize=(11, 3.6))
xy = obs.loc[m, ["CenterX_global_px", "CenterY_global_px"]].to_numpy()
for i, (labels, title) in enumerate([(true_k, "True clones"), (np.argsort(match)[pred_k], "Inferred clones")]):
    ax = fig.add_subplot(1, 3, i + 1)
    ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="Set2", s=4, vmin=0, vmax=7)
    ax.set_title(f"{title} ({patients[p]})")
    ax.set_aspect("equal")
    ax.axis("off")

ax = fig.add_subplot(1, 3, 3)
true_cn = sim["true_cn"][patients[p]]
gene_order = [list(gene_coords["gene_symbol"]).index(g) for g in md["gene_names"]]
for k in range(1, config.num_components):
    ax.plot(true_cn[k, gene_order] + 2 * k, color="k", lw=1)
    ax.plot(s_loc[p, match[k]] / np.log(2) + 2 * k, color=f"C{k}", lw=1, alpha=0.8)
ax.set_xlabel("Genes (genome order)")
ax.set_ylabel("log2 CN (offset per clone)")
ax.set_title("CN profiles: true (black) vs inferred")
ax.spines[["right", "top"]].set_visible(False)
fig.tight_layout()
fig.savefig(out / "cnv_inference.png", dpi=150)
print(f"Wrote {out}/clone_cn_profiles.csv, cnv_inference.png")
