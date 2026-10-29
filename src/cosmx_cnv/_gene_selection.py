"""Utilities for selecting CN-responsive genes using sWGS ground truth."""

from importlib.resources import files

import anndata
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.optimize import curve_fit
from scipy.special import gammaln
from scipy.stats import t as t_dist


def pseudobulk_expression(
    adata: anndata.AnnData,
    groupby: str = 'patient.id',
    L: float = 1e4,
    y0: float = 1.0,
) -> pd.DataFrame:
    """
    Per-sample average of size-factor-normalised log expression.

    Each cell is normalised as ``log(x / s + y0)`` with size factor
    ``s = total_counts / L``, then averaged over the cells of each group.

    Parameters
    ----------
    adata : AnnData
        Raw counts (cells x genes), typically restricted to cancer cells.
    groupby : str
        obs column defining samples (patients).
    L : float
        Target library size of the normalisation.
    y0 : float
        Pseudocount.

    Returns
    -------
    DataFrame
        Samples x genes, suitable as ``expr_table`` for ``fit_cn_expression_trends``.
    """
    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
    s = X.sum(axis=1, keepdims=True) / L
    logx = np.log(X / np.maximum(s, 1e-12) + y0)
    df = pd.DataFrame(logx, index=adata.obs_names, columns=adata.var_names)
    return df.groupby(adata.obs[groupby].astype(str).values).mean()


def get_gene_cnv_data(
    gene_coords: pd.DataFrame,
    segment_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Map gene coordinates to copy number values from sWGS segments.

    Joins gene_coords with segment_table by chromosome, filtering for full
    containment (gene start >= segment start AND gene end <= segment end).
    Creates all gene-patient combinations and fills missing with NaN.

    Parameters
    ----------
    gene_coords : DataFrame
        Must have columns: gene_symbol, chromosome, start, end.
    segment_table : DataFrame
        Must have columns: Chromosome, Start_Position, End_Position,
        Patient_ID, SegmentMean.

    Returns
    -------
    DataFrame
        Columns: gene_symbol, Patient_ID, SegmentMean. One row per
        gene-patient pair; SegmentMean is None where no segment covers the gene.
    """
    merged = pd.merge(
        gene_coords,
        segment_table,
        left_on='chromosome',
        right_on='Chromosome',
        how='left'
    )

    # Filter for full containment of gene inside segment
    mask = (
        (merged['start'] >= merged['Start_Position']) &
        (merged['end']   <= merged['End_Position'])
    )
    merged = merged[mask].copy()

    # Keep just the relevant columns
    merged = merged[['gene_symbol', 'Patient_ID', 'SegmentMean']]

    # Create a row for each gene-patient pair, even if unmatched
    unique_genes = gene_coords['gene_symbol'].unique()
    unique_patients = segment_table['Patient_ID'].unique()
    all_combinations = pd.MultiIndex.from_product(
        [unique_genes, unique_patients],
        names=['gene_symbol', 'Patient_ID']
    ).to_frame(index=False)

    # Merge full combinations with filtered results
    final = pd.merge(
        all_combinations,
        merged,
        on=['gene_symbol', 'Patient_ID'],
        how='left'
    )

    # SegmentMean is NaN where there was no match
    final['SegmentMean'] = final['SegmentMean'].where(
        pd.notnull(final['SegmentMean']),
        None
    )
    return final


def _pois_E_log1p(mu: np.ndarray) -> np.ndarray:
    """E[log(1 + Y)] for Y ~ Poisson(mu), summed exactly for mu <= 50, second-order delta method above."""
    mu = np.atleast_1d(np.asarray(mu, float))
    out = np.log1p(mu) - mu / (2 * (1 + mu) ** 2)
    small = mu <= 50
    if small.any():
        m = mu[small]
        k = np.arange(int(m.max() + 12 * np.sqrt(m.max()) + 30))[:, None]
        logp = k * np.log(np.maximum(m, 1e-300)) - m - gammaln(k + 1)
        out[small] = (np.log1p(k) * np.exp(logp)).sum(0)
    return out


def _f_poisson_log1p(x, b0, b1):
    return _pois_E_log1p(np.exp(np.clip(b0 + b1 * x, -30, 12)))


def fit_cn_expression_trends(
    cn_table: pd.DataFrame,
    expr_table: pd.DataFrame,
    fillna_with_zero: bool = False,
    min_samples: int = 4,
) -> pd.DataFrame:
    """
    Per-gene log-linear fit of pseudobulk expression on bulk copy number.

    For each gene, fits ``y_i = E[log(1 + Y_i)]`` with ``Y_i ~ Poisson(exp(b0 + b1 * x_i))``
    by nonlinear least squares, where ``x_i`` is the sWGS log2 CN ratio of sample i and
    ``y_i`` its pseudobulk expression (``pseudobulk_expression``). The slope ``b1`` is the
    log-expression change per log2 CN unit; ``r2`` is the fraction of across-sample
    expression variance the CN fit explains.

    Parameters
    ----------
    cn_table : DataFrame
        Samples x genes, log2 CN ratio.
    expr_table : DataFrame
        Samples x genes, pseudobulk expression.
    fillna_with_zero : bool
        Treat missing CN as neutral (0). Use this when the CN table lists only called
        segments; otherwise samples with missing CN are left out of that gene's fit.
    min_samples : int
        Skip genes with fewer usable samples.

    Returns
    -------
    DataFrame
        One row per gene: gene, slope, slope_ci_low, slope_ci_high (95%), pval, r2,
        intercept, n_samples. Genes where the fit fails are omitted.
    """
    samples = cn_table.index.astype(str).intersection(expr_table.index.astype(str))
    genes = cn_table.columns.intersection(expr_table.columns)
    cn = cn_table.set_axis(cn_table.index.astype(str)).loc[samples, genes].astype(float)
    expr = expr_table.set_axis(expr_table.index.astype(str)).loc[samples, genes].astype(float)
    if fillna_with_zero:
        cn = cn.fillna(0.0)

    rows = []
    for gene in genes:
        x, y = cn[gene].to_numpy(), expr[gene].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < min_samples or np.ptp(x) == 0:
            continue
        slope0 = np.polyfit(x, y, 1)[0]
        try:
            popt, pcov = curve_fit(_f_poisson_log1p, x, y, p0=[np.log(np.expm1(max(y.mean(), 1e-3))), slope0],
                                   maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        dof = max(x.size - 2, 1)
        se = np.sqrt(np.diag(pcov))
        tcrit = t_dist.ppf(0.975, dof)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - _f_poisson_log1p(x, *popt)) ** 2) / ss_tot if ss_tot > 0 else np.nan
        rows.append({
            "gene": gene, "slope": popt[1],
            "slope_ci_low": popt[1] - tcrit * se[1], "slope_ci_high": popt[1] + tcrit * se[1],
            "pval": 2 * t_dist.sf(abs(popt[1] / se[1]), dof) if se[1] > 0 else np.nan,
            "r2": r2, "intercept": popt[0], "n_samples": x.size,
        })
    return pd.DataFrame(rows)


def filter_genes_by_trend(
    trend_df: pd.DataFrame,
    gene_coords: pd.DataFrame,
    r2_threshold: float = 0.2,
    max_slope: float = 5.0,
) -> pd.DataFrame:
    """
    Keep genes whose expression rises with copy number and is well explained by it.

    A gene passes when the 95% CI of its slope lies within (0, ``max_slope``) and
    ``r2 > r2_threshold``. The defaults are the thresholds used to build
    ``load_breast_cancer_genes()``. The upper slope bound drops unstable fits.

    Parameters
    ----------
    trend_df : DataFrame
        Output of ``fit_cn_expression_trends``.
    gene_coords : DataFrame
        Gene coordinate table. Must have column: gene_symbol.

    Returns
    -------
    DataFrame
        Filtered copy of gene_coords.
    """
    good = trend_df[
        (trend_df['slope_ci_low'] > 0) &
        (trend_df['slope_ci_high'] < max_slope) &
        (trend_df['r2'] > r2_threshold)
    ]
    return gene_coords[gene_coords['gene_symbol'].isin(set(good['gene']))].copy()


def load_breast_cancer_genes() -> pd.DataFrame:
    """
    The 1443 CN-responsive genes of the CosMX 6k panel selected in breast cancer.

    Selected with ``fit_cn_expression_trends`` + ``filter_genes_by_trend`` (default
    thresholds) on paired bulk sWGS and CosMX 6k data from 117 breast tumours of the
    Madrid TMA cohort. Columns: gene_symbol, ensembl_id, chromosome (str), start, end
    (GRCh38), ready to pass as ``gene_coords``.
    """
    path = files("cosmx_cnv") / "data" / "breast_cancer_cn_genes.csv"
    with path.open() as fh:
        return pd.read_csv(fh, dtype={"chromosome": str})
