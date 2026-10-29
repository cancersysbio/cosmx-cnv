"""
Step 1 - select genes whose expression tracks copy number.

Uses paired bulk CN (sWGS segments) and CosMX expression across patients:
for every gene, fit pseudobulk expression as a log-linear (Poisson) function of
log2 CN and keep genes whose slope is clearly positive and whose fit explains
enough variance (R^2 > 0.2, the threshold used for the shipped breast-cancer list,
cc.load_breast_cancer_genes()).

Runs on simulated data in a few seconds:  python examples/01_gene_selection.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import cosmx_cnv as cc

out = Path("outputs")
out.mkdir(exist_ok=True)

# --- Data: 30 patients, 40% of genes ignore copy number -----------------------
# With real data: gene_coords from your panel (see cc.lookup_ensembl_coordinates),
# segments from QDNAseq/ichorCNA etc., adata restricted to cancer cells.
sim = cc.simulate_dataset(
    n_patients=30, n_cores_per_patient=1, n_cells_per_core=150,
    n_genes=200, non_responsive_frac=0.4, n_truncal_events=6, seed=1,
)
gene_coords, segments, adata = sim["gene_coords"], sim["segments"], sim["adata"]

# --- Per-patient CN and expression tables (patients x genes) ------------------
gene_cn = cc.get_gene_cnv_data(gene_coords, segments)
cn_table = gene_cn.pivot_table(values="SegmentMean", index="gene_symbol", columns="Patient_ID").T
expr_table = cc.pseudobulk_expression(adata, groupby="patient.id")

# --- CN-expression fits and filter --------------------------------------------
res_df = cc.fit_cn_expression_trends(cn_table, expr_table)
good_genes = cc.filter_genes_by_trend(res_df, gene_coords, r2_threshold=0.2, max_slope=5)

truth = pd.Series(sim["responsive"], index=gene_coords["gene_symbol"])
selected = truth.index.isin(good_genes["gene_symbol"])
print(f"Selected {selected.sum()}/{len(truth)} genes; "
      f"precision {truth[selected].mean():.2f}, recall {truth[selected].sum() / truth.sum():.2f}")

res_df.to_csv(out / "cn_expression_fits.csv", index=False)
good_genes.to_csv(out / "good_gene_coords.csv", index=False)

# --- Plot: variance explained vs slope, coloured by ground truth --------------
is_resp = res_df["gene"].map(truth)
fig, ax = plt.subplots(figsize=(4.5, 3.5))
for flag, label, color in [(True, "CN-responsive", "C0"), (False, "non-responsive", "C1")]:
    d = res_df[is_resp == flag]
    ax.scatter(d["slope"], d["r2"], s=10, alpha=0.7, color=color, label=label)
ax.axhline(0.2, color="k", lw=0.8, ls="--")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(-1, 5)  # genes with almost no CN variation give unstable slopes; max_slope drops them
ax.set_xlabel("Slope (log expression per log2 CN)")
ax.set_ylabel("Variance explained (R$^2$)")
ax.spines[["right", "top"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "gene_selection.png", dpi=150)
print(f"Wrote {out}/cn_expression_fits.csv, good_gene_coords.csv, gene_selection.png")
