"""Post-inference utilities for extracting clone state tables."""

from typing import Dict

import pandas as pd


def create_clone_state_table(
    params: Dict,
    data_dict: Dict[str, Dict],
    gene_coords: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Create a table of inferred CN states for each patient's clones.

    Parameters
    ----------
    params : dict
        Dictionary returned by ``CNVModel.get_parameters()``.
        Must contain ``s_loc`` (n_patients x K x n_genes tensor).
    data_dict : dict
        Per-chromosome data dictionaries (from ``prepare_chromosome_data``).
        Each value must contain ``gene_names`` and ``adata``.
    gene_coords : DataFrame
        Gene coordinate table with columns: gene_symbol, chromosome, start, end.

    Returns
    -------
    dict
        Mapping patient_id -> DataFrame with columns: gene_symbol, chromosome,
        start, end, state_0, state_1, ...
    """
    # Get s_loc tensor (clone states) and convert to numpy
    s_loc = params['s_loc'].detach().cpu().numpy()

    # Collect genes and their coordinates across chromosomes
    all_genes = []
    gene_chr = []
    gene_starts = []
    gene_ends = []

    for chr_id in data_dict.keys():
        chr_data = data_dict[chr_id]
        if chr_data is None:
            continue
        chr_genes = chr_data['gene_names']
        all_genes.extend(chr_genes)

        # Get coordinates for these genes
        for gene in chr_genes:
            gene_info = gene_coords[gene_coords['gene_symbol'] == gene].iloc[0]
            gene_chr.append(gene_info['chromosome'])
            gene_starts.append(gene_info['start'])
            gene_ends.append(gene_info['end'])

    # Create tables for each patient
    patient_tables = {}

    # Get patient IDs from first chromosome's data
    first_chr = next(
        k for k, v in data_dict.items() if v is not None
    )
    patients = data_dict[first_chr]['adata'].obs['patient.id'].cat.categories.values

    for patient_idx, patient_id in enumerate(patients):
        # Get clone states for this patient
        patient_states = s_loc[patient_idx]

        # Create DataFrame
        df = pd.DataFrame({
            'gene_symbol': all_genes,
            'chromosome': gene_chr,
            'start': gene_starts,
            'end': gene_ends
        })

        # Add state columns
        for state_idx in range(patient_states.shape[0]):
            df[f'state_{state_idx}'] = patient_states[state_idx]

        patient_tables[patient_id] = df

    return patient_tables


def create_combined_clone_state_table(
    params: Dict,
    data_dict: Dict[str, Dict],
    gene_coords: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a combined table of inferred CN states for all patients' clones.

    Convenience wrapper around :func:`create_clone_state_table` that
    concatenates all patients into a single DataFrame with a ``patient_id``
    column.

    Parameters
    ----------
    params : dict
        Dictionary returned by ``CNVModel.get_parameters()``.
    data_dict : dict
        Per-chromosome data dictionaries.
    gene_coords : DataFrame
        Gene coordinate table.

    Returns
    -------
    DataFrame
        Combined table with columns: patient_id, gene_symbol, chromosome,
        start, end, state_0, state_1, ...
    """
    patient_tables = create_clone_state_table(params, data_dict, gene_coords)

    dfs = []
    for patient_id, df in patient_tables.items():
        df = df.copy()
        df['patient_id'] = patient_id
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Reorder columns to put patient_id first
    cols = ['patient_id', 'gene_symbol', 'chromosome', 'start', 'end'] + \
           [col for col in combined_df.columns if col.startswith('state_')]
    combined_df = combined_df[cols]

    return combined_df
