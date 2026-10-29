import warnings

import numpy as np
import pyro
import pytest

import cosmx_cnv as cc

warnings.filterwarnings("ignore", message="Found plate statements in guide but not model")


@pytest.fixture(scope="module")
def sim():
    return cc.simulate_dataset(n_patients=2, n_cores_per_patient=1, n_cells_per_core=60,
                               n_genes=60, chromosomes=("1", "2", "3"), seed=0)


@pytest.fixture(scope="module")
def model_data(sim):
    gc = sim["gene_coords"]
    data_dict = {c: cc.prepare_chromosome_data(c, {"Cancer": sim["adata"]}, gc) for c in ("1", "2", "3")}
    return data_dict, cc.prepare_data(data_dict)


def test_prepare_data(sim, model_data):
    _, md = model_data
    assert md["y_counts"].shape == (120, 60)
    # each gene is labelled with its own chromosome
    chrom = sim["gene_coords"].set_index("gene_symbol").loc[md["gene_names"], "chromosome"]
    assert list(chrom) == md["chr_ids"]
    # gene edges never cross chromosomes
    chr_ids = np.array(md["chr_ids"])
    adj = md["gene_adjacency"].numpy()
    assert (chr_ids[adj[:, 0]] == chr_ids[adj[:, 1]]).all()


def test_fit_and_results(sim, model_data):
    data_dict, md = model_data
    pyro.set_rng_seed(0)
    model = cc.CNVModel(cc.CNVModelConfig(num_components=3))
    model.fit(
        md["y_counts"], num_iterations=20, lr=0.05,
        r_init=md["y_counts"].log1p().mean(0, keepdim=True),
        cell_neighbours_adj=md["cell_adjacency"], cell_neighbours_dist=md["cell_distances"],
        gene_neighbours_adj=md["gene_adjacency"], gene_neighbours_dist=md["gene_distances"],
        core_ids=md["cell_core_ids"], patient_ids=md["cell_patient_ids"],
    )
    assert len(model.losses) == 20 and np.isfinite(model.losses).all()
    probs = model.get_membership_probs()
    assert probs.shape == (120, 3)
    assert np.allclose(probs.sum(1).numpy(), 1, atol=1e-5)
    params = model.get_parameters()
    assert params["s_loc"].shape == (2, 3, 60)
    table = cc.create_combined_clone_state_table(params, data_dict, sim["gene_coords"])
    assert len(table) == 2 * 60


def test_reference_requires_config(model_data):
    _, md = model_data
    model = cc.CNVModel(cc.CNVModelConfig(num_components=3, use_reference=True))
    with pytest.raises(ValueError):
        model.fit(md["y_counts"], num_iterations=1)


def test_gene_selection():
    sim = cc.simulate_dataset(n_patients=12, n_cores_per_patient=1, n_cells_per_core=50,
                              n_genes=30, non_responsive_frac=0.5, n_truncal_events=6, seed=1)
    gc = sim["gene_coords"]
    cn = cc.get_gene_cnv_data(gc, sim["segments"]).pivot_table(
        values="SegmentMean", index="gene_symbol", columns="Patient_ID").T
    expr = cc.pseudobulk_expression(sim["adata"])
    assert expr.shape == (12, 30)
    res, _ = cc.bootstrap_all_genes(cn, expr, n_boot=10, fillna_with_zero=False)
    assert {"gene", "support", "unexplained_mean", "mean_delta"} <= set(res.columns)
    good = cc.filter_genes_by_bootstrap(res, gc)
    assert set(good["gene_symbol"]) <= set(gc["gene_symbol"])
