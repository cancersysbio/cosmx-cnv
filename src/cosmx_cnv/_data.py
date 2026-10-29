import logging
from typing import Dict, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import anndata
import scipy

logger = logging.getLogger(__name__)


def align_panel_genes(
    adata: anndata.AnnData,
    gene_coords: pd.DataFrame,
    gene_col: str = "gene_symbol",
) -> pd.DataFrame:
    """
    Reconcile gene-coord symbols with a panel's var_names.

    Two mismatches are handled, in order:

    1. Slash/hyphen aliases. Some CosMX panel symbols write a hyphenated gene
       with a slash ('NKX6/1' for 'NKX6-1'). The slash is overloaded in this
       panel though: it ALSO marks genuine multi-target probes ('HBA1/2',
       'CCL3/L1/L3', 'HLA-DQB1/2'), which are not hyphenated genes and have no
       single coordinate. So a slash is only rewritten to a hyphen when the
       hyphenated form is itself present in gene_coords; everything else is
       left alone. adata.var_names is renamed in place.

    2. Genes still absent from var_names are dropped from gene_coords, because
       prepare_chromosome_data indexes adata[:, gene_names] and raises a
       KeyError on any missing symbol.

    Args:
        adata: Panel AnnData; var_names are renamed in place for recovered aliases
        gene_coords: Gene coordinate table to restrict
        gene_col: Column in gene_coords holding the gene symbol

    Returns:
        gene_coords restricted to symbols present in adata.var_names
    """
    coord_genes = set(gene_coords[gene_col].astype(str))
    var_names = list(adata.var_names.astype(str))

    # 1. Recover slash-written hyphenated genes, but only where unambiguous.
    alias = {
        v: v.replace("/", "-")
        for v in var_names
        if "/" in v and v.replace("/", "-") in coord_genes
    }
    if alias:
        logger.info(
            "Recovered %d slash-written gene symbols in var_names (e.g. %s)",
            len(alias),
            ", ".join(f"{k}->{v}" for k, v in sorted(alias.items())[:5]),
        )
        adata.var_names = pd.Index([alias.get(v, v) for v in var_names])

    # 2. Drop coord genes the panel simply does not carry.
    panel_genes = set(adata.var_names.astype(str))
    n_before = len(gene_coords)
    missing_mask = ~gene_coords[gene_col].astype(str).isin(panel_genes)
    if missing_mask.any():
        missing = sorted(gene_coords.loc[missing_mask, gene_col].astype(str))
        logger.warning(
            "%d/%d gene-coord genes are not in the panel var_names and will be "
            "dropped: %s",
            len(missing), n_before, missing,
        )
        gene_coords = gene_coords.loc[~missing_mask].reset_index(drop=True)

    logger.info("Using %d/%d genes present in the panel", len(gene_coords), n_before)
    return gene_coords


def get_nearest_neighbours(
    coords: np.ndarray,
    n_neighbours: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find nearest neighbours using KDTree for memory efficient computation.

    Args:
        coords: Array of shape (n_points, n_dims) containing spatial coordinates
        n_neighbours: Number of nearest neighbours to find

    Returns:
        Tuple of (neighbour_indices, neighbour_distances) arrays

    Raises:
        ValueError: If coords is empty or n_neighbours > n_points
    """
    if len(coords) == 0:
        raise ValueError("Empty coordinates array provided")
    if n_neighbours >= len(coords):
        raise ValueError(f"n_neighbours ({n_neighbours}) must be less than number of points ({len(coords)})")

    tree = KDTree(coords)
    distances, indices = tree.query(coords, k=n_neighbours+1)

    return indices[:, 1:], distances[:, 1:]


def get_adjacency_from_knn(
    neighbour_indices: np.ndarray,
    n_neighbours_mask: np.ndarray,
    neighbour_distances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert KNN results to unique adjacency pairs with distances.

    Args:
        neighbour_indices: Array of shape (n_points, n_neighbours)
        n_neighbours_mask: Boolean mask of valid neighbours
        neighbour_distances: Array of distances

    Returns:
        Tuple of (adjacency_pairs, distances) arrays
    """
    pairs = {}
    for i in range(len(neighbour_indices)):
        for j in range(len(neighbour_indices[0])):
            if n_neighbours_mask[i, j]:
                i2 = neighbour_indices[i, j]
                dist = neighbour_distances[i, j]
                if i < i2:
                    pairs[(i, i2)] = dist
                else:
                    pairs[(i2, i)] = dist

    adjacency = []
    distances = []
    for (i, j), dist in sorted(pairs.items()):
        adjacency.append([i, j])
        distances.append(dist)

    return np.array(adjacency), np.array(distances)


def _grouped_cell_adjacency(
    coords: np.ndarray,
    group_ids: np.ndarray,
    n_neighbours: int,
    distance_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cell KNN adjacency computed WITHIN each group (e.g. TMA/slide).

    In a joint multi-TMA run the per-slide CenterX/Y_global_px coordinate
    systems overlap, so a single global KDTree links cells on different slides
    (different patients) as spurious spatial neighbours and corrupts the
    spatial-smoothness penalty. Building the graph per group prevents any edge
    from crossing a slide boundary. Returns adjacency in the coordinate array's
    own index space (0..len(coords)-1) with matching distances.
    """
    adj_list = []
    dist_list = []
    for g in np.unique(group_ids):
        members = np.where(group_ids == g)[0]
        if len(members) < 2:
            continue
        k = min(n_neighbours, len(members) - 1)
        idx, dists = get_nearest_neighbours(coords[members], n_neighbours=k)
        mask = dists < distance_threshold
        adj, adj_dist = get_adjacency_from_knn(idx, mask, dists)
        if adj.size == 0:
            continue
        # Map within-group indices back to the passed array's index space
        adj_list.append(members[adj])
        dist_list.append(adj_dist)

    if adj_list:
        return np.concatenate(adj_list), np.concatenate(dist_list)
    return np.zeros((0, 2), dtype=np.int64), np.zeros(0)


def prepare_data(
    chr_data_dict: Dict[str, Dict],
    device: str = "cpu",
    n_cell_neighbors: int = 10,
    n_gene_neighbors: int = 10,
    cell_distance_threshold: float = 2000,
    gene_distance_threshold: float = 5e6,
) -> Dict:
    """
    Prepare data for CNV model fitting.

    Args:
        chr_data_dict: Dictionary mapping chromosome IDs to data dictionaries
        device: PyTorch device string
        n_cell_neighbors: Number of cell neighbors to consider
        n_gene_neighbors: Number of gene neighbors to consider
        cell_distance_threshold: Maximum distance between neighboring cells (in pixels)
        gene_distance_threshold: Maximum distance between neighboring genes (in base pairs)

    Returns:
        Dictionary containing prepared data
    """
    # Convert dictionary to list of valid chromosome data
    chr_data_list = []
    chr_id_list = []
    for chr_id, chr_data in chr_data_dict.items():
        if chr_data is None:
            print(f"Skipping empty chromosome {chr_id}")
            continue
        if len(chr_data['gene_names']) == 0:
            print(f"Skipping chromosome {chr_id} with no genes")
            continue
        chr_data_list.append(chr_data)
        chr_id_list.append(chr_id)

    if not chr_data_list:
        raise ValueError("No valid chromosome data found")

    # Get gene names and offsets
    gene_names_list = []
    gene_offsets = [0]  # Keep track of gene indices offsets
    chr_ids_list = []  # Keep track of which chromosome each gene belongs to

    # First pass to collect valid chromosomes and build offsets
    for chr_id, chr_data in zip(chr_id_list, chr_data_list):
        gene_names_list.extend(chr_data['gene_names'])
        chr_ids_list.extend([chr_id] * len(chr_data['gene_names']))
        gene_offsets.append(gene_offsets[-1] + len(chr_data['gene_names']))

    # Create adjacency matrix considering chromosome boundaries
    gene_adjacency_list = []
    gene_distances_list = []

    for i, chr_data in enumerate(chr_data_list):
        # Get gene coordinates for this chromosome
        gene_coords = chr_data['sorted_good_genes_df']['mean_position'].values.reshape(-1, 1)

        # Calculate neighbors for this chromosome
        n_genes = len(gene_coords)
        if n_genes > 1:  # Only calculate neighbors if more than one gene
            n_neighbours = min(n_gene_neighbors, n_genes-1)
            gene_neihbours_indices, gene_neihbours_distances = get_nearest_neighbours(
                gene_coords,
                n_neighbours=n_neighbours
            )

            # Apply distance threshold
            gene_neihbours_mask = gene_neihbours_distances < gene_distance_threshold

            # Get adjacency and distances
            chr_adjacency, chr_distances = get_adjacency_from_knn(
                gene_neihbours_indices,
                gene_neihbours_mask,
                gene_neihbours_distances
            )

            # Offset the indices by the cumulative number of genes
            chr_adjacency = chr_adjacency + gene_offsets[i]

            gene_adjacency_list.append(chr_adjacency)
            gene_distances_list.append(chr_distances)

    # Stack data across chromosomes
    y_counts_list = []
    for chr_data in chr_data_list:
        y_counts_list.append(chr_data['y_counts'])

    # Get cell data from first chromosome since dimensions match
    first_chr_data = chr_data_list[0]
    coords_stacked = first_chr_data['adata'].obs[['CenterX_global_px', 'CenterY_global_px']].values
    patient_ids_stacked = first_chr_data['adata'].obs['patient.id'].cat.codes.values
    core_ids_stacked = first_chr_data['adata'].obs['core_id'].cat.codes.values
    tma_ids_stacked = (
        first_chr_data['adata'].obs['tma_id'].cat.codes.values
        if 'tma_id' in first_chr_data['adata'].obs
        else None
    )
    reference_mask = first_chr_data['reference_mask'] if 'reference_mask' in first_chr_data else None

    # Stack all data
    y_counts_stacked = np.hstack(y_counts_list)  # Stack genes horizontally

    # Cell spatial adjacency, built WITHIN each TMA/slide (group_ids) so KNN
    # edges never cross slides. Falls back to a single group when tma_id is
    # absent (reproduces the original single-TMA behaviour).
    group_ids_stacked = (
        tma_ids_stacked if tma_ids_stacked is not None
        else np.zeros(len(coords_stacked), dtype=np.int64)
    )
    if reference_mask is not None:
        target_mask = ~reference_mask
        target_indices = np.where(target_mask)[0]
        cell_adjacency, cell_distances = _grouped_cell_adjacency(
            coords_stacked[target_mask],
            group_ids_stacked[target_mask],
            n_cell_neighbors,
            cell_distance_threshold,
        )
        # Map target-local indices back to the full cell set
        if cell_adjacency.size:
            cell_adjacency = np.column_stack([
                target_indices[cell_adjacency[:, 0]],
                target_indices[cell_adjacency[:, 1]]
            ])
    else:
        cell_adjacency, cell_distances = _grouped_cell_adjacency(
            coords_stacked,
            group_ids_stacked,
            n_cell_neighbors,
            cell_distance_threshold,
        )

    # Concatenate all adjacency matrices and distances
    gene_adjacency_list = [arr for arr in gene_adjacency_list if arr.size > 0]
    gene_distances_list = [arr for arr in gene_distances_list if arr.size > 0]

    # Reshape 1D arrays to 2D if needed and concatenate
    gene_adjacency_stacked = np.concatenate([
        arr if len(arr.shape) == 2 and arr.shape[1] == 2
        else arr.reshape(-1, 2)  # reshape to have 2 columns (source, target)
        for arr in gene_adjacency_list if arr.size > 0
    ]) if gene_adjacency_list else np.array([])

    gene_distances_stacked = np.concatenate([
        arr.flatten()  # ensure flat 1D array
        for arr in gene_distances_list if arr.size > 0
    ]) if gene_distances_list else np.array([])

    # Create a mapping from original gene names to their new indices after concatenation
    gene_name_to_idx = {gene: idx for idx, gene in enumerate(gene_names_list)}

    # Handle reference configuration
    processed_ref_config = None
    if any('reference_config' in chr_data and chr_data['reference_config'] is not None
           for chr_data in chr_data_dict.values()):
        processed_ref_config = {
            'masks': {},
            'profiles': {}
        }

        # Combine masks and reorder profiles across chromosomes
        for chr_id, chr_data in chr_data_dict.items():
            if 'reference_config' not in chr_data or chr_data['reference_config'] is None:
                continue

            ref_config = chr_data['reference_config']

            # Handle masks
            for comp_idx, mask in ref_config['masks'].items():
                if comp_idx not in processed_ref_config['masks']:
                    processed_ref_config['masks'][comp_idx] = torch.tensor(mask, dtype=torch.bool, device=device)
                else:
                    processed_ref_config['masks'][comp_idx] |= torch.tensor(mask, dtype=torch.bool, device=device)

            # Handle profiles
            if 'profiles' in ref_config:
                for comp_idx, patient_profiles in ref_config['profiles'].items():
                    if patient_profiles is not None:
                        if comp_idx not in processed_ref_config['profiles']:
                            processed_ref_config['profiles'][comp_idx] = {}

                        for patient_idx, profile in patient_profiles.items():
                            if patient_idx not in processed_ref_config['profiles'][comp_idx]:
                                reindexed_profile = profile.reindex(
                                    gene_names_list,
                                    fill_value=0
                                )
                                processed_ref_config['profiles'][comp_idx][patient_idx] = torch.tensor(
                                    reindexed_profile.values,
                                    dtype=torch.float32,
                                    device=device
                                )
    return {
        'y_counts': torch.tensor(y_counts_stacked, dtype=torch.float32, device=device),
        'cell_adjacency': torch.tensor(cell_adjacency, dtype=torch.int64, device=device),
        'cell_distances': torch.tensor(cell_distances, dtype=torch.float32, device=device),
        'gene_adjacency': torch.tensor(gene_adjacency_stacked, dtype=torch.int64, device=device),
        'gene_distances': torch.tensor(gene_distances_stacked, dtype=torch.float32, device=device),
        'cell_core_ids': torch.tensor(core_ids_stacked, dtype=torch.int64, device=device),
        'cell_patient_ids': torch.tensor(patient_ids_stacked, dtype=torch.int64, device=device),
        'cell_tma_ids': (
            torch.tensor(tma_ids_stacked, dtype=torch.int64, device=device)
            if tma_ids_stacked is not None else None
        ),
        'reference_mask': torch.tensor(reference_mask, dtype=torch.bool, device=device) if reference_mask is not None else None,
        'reference_config': processed_ref_config,  # Now includes properly ordered profiles
        'gene_names': gene_names_list,
        'chr_ids': chr_ids_list,
        'gene_offsets': gene_offsets,
        # Barcodes in the same (core-sorted) order as y_counts / z_logits, so
        # callers can align per-cell outputs to the fitted membership.
        'cell_names': first_chr_data['adata'].obs.index.values,
    }


def prepare_chromosome_data(
    chr_id: str,
    cell_type_dict: Dict[str, anndata.AnnData],
    gene_coords_good_6k: pd.DataFrame,
    reference_config: Optional[Dict[str, Dict]] = None,
    patient_gene_cn_6k: Optional[pd.DataFrame] = None,
    target_key: str = 'Cancer',
) -> Dict:
    """
    Prepare data for a single chromosome.

    Args:
        chr_id: Chromosome identifier
        cell_type_dict: Dictionary mapping cell types to AnnData objects
        gene_coords_good_6k: DataFrame of gene coordinates
        reference_config: Optional dictionary of reference configuration
        patient_gene_cn_6k: Optional DataFrame of patient gene copy numbers
        target_key: Key for target cells in cell_type_dict
    """
    # preparing gene
    sorted_good_genes_df = gene_coords_good_6k[gene_coords_good_6k['chromosome'] == chr_id]
    sorted_good_genes_df = sorted_good_genes_df.sort_values(by='start')
    sorted_good_genes_df = sorted_good_genes_df.assign(mean_position=((sorted_good_genes_df['start'] + sorted_good_genes_df['end']) / 2).astype(int))
    gene_names = sorted_good_genes_df['gene_symbol'].values

    if hasattr(gene_names, "to_numpy"):
        gene_names = gene_names.to_numpy()
    gene_names = [str(g) for g in list(gene_names)]


    # Group genes into clusters based on 1MB proximity
    current_cluster = 1
    cluster_assignments = [current_cluster]

    for i in range(1, len(gene_names)):
        # Check if current gene is within 1MB of previous gene
        if sorted_good_genes_df.iloc[i]['mean_position'] - sorted_good_genes_df.iloc[i-1]['mean_position'] <= 2e6:
            cluster_assignments.append(current_cluster)
        else:
            # Start new cluster
            current_cluster += 1
            cluster_assignments.append(current_cluster)

    # Add cluster assignments as a new column
    sorted_good_genes_df['cluster'] = cluster_assignments

    print(f"Found {current_cluster} clusters of genes")

    cluster_sizes = [len(sorted_good_genes_df[sorted_good_genes_df['cluster'] == i]) for i in range(1, current_cluster + 1)]
    cluster_borders = np.cumsum(cluster_sizes)

    # Get target cells data only
    target_adata = cell_type_dict[target_key][:, gene_names].copy()

    # Store original order indices before sorting
    original_indices = np.arange(len(target_adata))

    # Sort by core_id and get the sorting indices
    sort_idx = target_adata.obs['core_id'].argsort()
    adata = target_adata[sort_idx]
    y_counts = np.array(adata.X.toarray()) if scipy.sparse.issparse(adata.X) else np.array(adata.X)

    # number of cells per core
    core_ids = adata.obs['core_id'].cat.categories.values
    cell_number_per_core = [np.sum(adata.obs['core_id'] == core_id) for core_id in core_ids]
    core_borders = np.cumsum(cell_number_per_core)

    coords = adata.obs[['CenterX_global_px', 'CenterY_global_px']].values
    cell_core_ids = adata.obs['core_id'].cat.codes.values
    patients = adata.obs['patient.id'].cat.categories.values
    cell_patient_ids = adata.obs['patient.id'].cat.codes.values

    # Convert core_id and patient.id to strings directly from categorical values
    core_patient_df = adata.obs[['core_id', 'patient.id']].drop_duplicates()
    coreid_to_patientid = dict(zip(
        core_patient_df['core_id'].astype(str),
        core_patient_df['patient.id'].astype(str)
    ))

    if patient_gene_cn_6k is not None:
        patient_gene_cn_by_core = patient_gene_cn_6k.loc[gene_names].reindex(columns=coreid_to_patientid.values())
        patient_gene_cn_by_core.columns = list(coreid_to_patientid.keys())
    else:
        patient_gene_cn_by_core = None

    # Get patient IDs and create mapping
    patient_ids = sorted(list(set(adata.obs['patient.id'])))
    patient_id_to_idx = {pid: idx for idx, pid in enumerate(patient_ids)}

    # Handle reference configuration
    processed_ref_config = None
    reference_mask = None
    if reference_config is not None:
        reference_mask = np.zeros(len(adata), dtype=bool)
        processed_ref_config = {
            'masks': {},
            'profiles': {}
        }

        for comp_idx, ref_info in reference_config['masks'].items():
            # Reorder the mask according to the core_id sorting
            reordered_mask = ref_info[sort_idx]
            processed_ref_config['masks'][comp_idx] = reordered_mask
            reference_mask |= reordered_mask  # Combine all reference masks

        if 'profiles' in reference_config:
            for comp_idx, patient_profiles in reference_config['profiles'].items():
                if patient_profiles is not None:
                    processed_ref_config['profiles'][comp_idx] = {
                        patient_id_to_idx[p_id]: profile
                        for p_id, profile in patient_profiles.items()
                    }

    return {
        'gene_names': gene_names,
        'core_ids': core_ids,
        'core_borders': core_borders,
        'cluster_borders': cluster_borders,
        'y_counts': y_counts,
        'patient_gene_cn_by_core': patient_gene_cn_by_core,
        'adata': adata,
        'coreid_to_patientid': coreid_to_patientid,
        'sorted_good_genes_df': sorted_good_genes_df,
        'reference_mask': reference_mask,
        'reference_config': processed_ref_config,
        'patient_ids': patient_ids,
        'patient_id_to_idx': patient_id_to_idx
    }
