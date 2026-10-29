"""Utilities for selecting CN-responsive genes using sWGS ground truth."""

import anndata
import numpy as np
import pandas as pd
import scipy.sparse


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
        Samples x genes, suitable as ``expr_table`` for ``bootstrap_all_genes``.
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


def filter_genes_by_bootstrap(
    res_df: pd.DataFrame,
    gene_coords: pd.DataFrame,
    support_threshold: float = 0.8,
    unexplained_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Filter gene coordinate table to genes that show CN-expression dependency.

    Uses bootstrap results to keep genes where:
    - support >= support_threshold (bootstrap confidence in directionality)
    - unexplained_mean <= unexplained_threshold (model fits reasonably)

    Parameters
    ----------
    res_df : DataFrame
        Bootstrap results from ``bootstrap_all_genes``. Must have columns:
        gene, support, unexplained_mean.
    gene_coords : DataFrame
        Gene coordinate table. Must have column: gene_symbol.
    support_threshold : float
        Minimum support value (default 0.8).
    unexplained_threshold : float
        Maximum unexplained variance (default 0.8).

    Returns
    -------
    DataFrame
        Filtered copy of gene_coords containing only CN-responsive genes.
    """
    good = res_df[
        (res_df['support'] >= support_threshold) &
        (res_df['unexplained_mean'] <= unexplained_threshold)
    ]
    good_genes = set(good['gene'])
    return gene_coords[gene_coords['gene_symbol'].isin(good_genes)].copy()
