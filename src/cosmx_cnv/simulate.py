"""Synthetic CosMX-like data with known clonal CN structure, for examples and tests."""

from typing import Dict, Sequence

import anndata
import numpy as np
import pandas as pd
import scipy.sparse

from .tools.genome_info import CHROMOSOME_LENGTHS

# log2 copy-number ratios relative to diploid: 1 copy, 3 copies, 4 copies
_CN_LEVELS = np.array([-1.0, np.log2(3 / 2), 1.0])


def _random_events(rng, chromosomes, n_events):
    """Random arm-scale CN events as (chromosome, start, end, log2_level)."""
    events = []
    for _ in range(n_events):
        chrom = str(rng.choice(chromosomes))
        length = CHROMOSOME_LENGTHS[chrom]
        width = rng.uniform(0.15, 0.6) * length
        start = rng.uniform(0, length - width)
        events.append((chrom, start, start + width, float(rng.choice(_CN_LEVELS))))
    return events


def _cn_at(events, chrom, pos):
    """log2 CN ratio at a set of positions on one chromosome (events add up)."""
    cn = np.zeros(len(pos))
    for e_chrom, start, end, level in events:
        if e_chrom == chrom:
            cn += level * ((pos >= start) & (pos <= end))
    return cn


def simulate_dataset(
    n_patients: int = 3,
    n_cores_per_patient: int = 2,
    n_cells_per_core: int = 300,
    n_clones: int = 3,
    n_genes: int = 300,
    chromosomes: Sequence[str] = ("1", "2", "3", "4", "5", "6"),
    non_responsive_frac: float = 0.0,
    n_truncal_events: int = 3,
    n_private_events: int = 2,
    seed: int = 0,
) -> Dict:
    """
    Simulate a small CosMX-like cohort with spatially coherent subclones.

    Each patient has ``n_clones`` clones: clone 0 is copy-number neutral, the
    others share ``n_truncal_events`` patient-level events plus
    ``n_private_events`` of their own. Within each core, clones occupy
    contiguous Voronoi patches. Counts are Poisson with
    ``rate = exp(ln2 * cn * responsive + r_gene + core_bias + cell_sensitivity)``,
    i.e. expression of responsive genes scales linearly with copy number and
    a ``non_responsive_frac`` of genes ignores copy number entirely.

    Parameters
    ----------
    n_patients, n_cores_per_patient, n_cells_per_core : int
        Cohort size.
    n_clones : int
        Number of clones per patient (including the neutral clone 0).
    n_genes : int
        Number of genes, spread over ``chromosomes`` proportionally to length.
    chromosomes : sequence of str
        Chromosomes to place genes on.
    non_responsive_frac : float
        Fraction of genes whose expression does not depend on copy number.
    n_truncal_events, n_private_events : int
        CN events shared by all tumour clones of a patient / unique to a clone.
    seed : int
        RNG seed.

    Returns
    -------
    dict
        ``adata``: AnnData (cells x genes) with the obs columns the model
        expects (``patient.id``, ``core_id``, ``CenterX_global_px``,
        ``CenterY_global_px``, ``cell_type``) plus ``true_clone``.
        ``gene_coords``: DataFrame (gene_symbol, chromosome, start, end).
        ``segments``: bulk sWGS-like segment table per patient (Chromosome,
        Start_Position, End_Position, Patient_ID, SegmentMean).
        ``true_cn``: dict patient_id -> array (n_clones x n_genes) of log2 CN.
        ``responsive``: bool array (n_genes,) of CN-responsive genes.
    """
    rng = np.random.default_rng(seed)
    chromosomes = [str(c) for c in chromosomes]

    # --- genes ---
    lengths = np.array([CHROMOSOME_LENGTHS[c] for c in chromosomes], dtype=float)
    gene_chr = rng.choice(chromosomes, size=n_genes, p=lengths / lengths.sum())
    gene_start = np.array([rng.uniform(0, CHROMOSOME_LENGTHS[c] - 2e5) for c in gene_chr])
    gene_end = gene_start + rng.uniform(5e3, 1e5, size=n_genes)
    order = np.lexsort((gene_start, [chromosomes.index(c) for c in gene_chr]))
    gene_coords = pd.DataFrame({
        "gene_symbol": [f"G{i:04d}" for i in range(n_genes)],
        "chromosome": gene_chr[order],
        "start": gene_start[order].astype(int),
        "end": gene_end[order].astype(int),
    })
    gene_mid = ((gene_coords["start"] + gene_coords["end"]) / 2).to_numpy()

    responsive = rng.random(n_genes) >= non_responsive_frac
    r_gene = rng.normal(0.5, 0.6, size=n_genes)

    # --- patients, clones, cells ---
    obs_rows, counts, true_cn, segments = [], [], {}, []
    core_spacing, core_radius = 5000.0, 1500.0
    core_counter = 0
    for p in range(n_patients):
        pid = f"P{p + 1:02d}"
        truncal = _random_events(rng, chromosomes, n_truncal_events)
        clone_events = [[]] + [
            truncal + _random_events(rng, chromosomes, n_private_events)
            for _ in range(n_clones - 1)
        ]
        cn = np.zeros((n_clones, n_genes))
        for k, events in enumerate(clone_events):
            for chrom in chromosomes:
                m = (gene_coords["chromosome"] == chrom).to_numpy()
                cn[k, m] = _cn_at(events, chrom, gene_mid[m])
        true_cn[pid] = cn

        clone_fracs = np.zeros(n_clones)
        for c in range(n_cores_per_patient):
            core_id = f"{pid}_C{c + 1}"
            center = np.array([
                (core_counter % 6) * core_spacing,
                (core_counter // 6) * core_spacing,
            ])
            core_counter += 1

            # Cells uniform in a disc; clones are Voronoi patches around seeds
            radius = core_radius * np.sqrt(rng.random(n_cells_per_core))
            angle = rng.uniform(0, 2 * np.pi, n_cells_per_core)
            xy = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
            seeds = rng.uniform(-core_radius, core_radius, size=(n_clones, 2)) * 0.7
            clone = np.argmin(((xy[:, None, :] - seeds[None]) ** 2).sum(-1), axis=1)
            clone_fracs += np.bincount(clone, minlength=n_clones)

            core_bias = rng.normal(0, 0.05)
            sensitivity = rng.normal(0, 0.3, size=n_cells_per_core)
            log_rate = (
                np.log(2) * cn[clone] * responsive
                + r_gene + core_bias + sensitivity[:, None]
            )
            counts.append(rng.poisson(np.exp(log_rate)))
            for i in range(n_cells_per_core):
                obs_rows.append({
                    "patient.id": pid,
                    "core_id": core_id,
                    "CenterX_global_px": center[0] + xy[i, 0],
                    "CenterY_global_px": center[1] + xy[i, 1],
                    "cell_type": "Cancer",
                    "true_clone": int(clone[i]),
                })

        segments.append(_bulk_segments(pid, clone_events, clone_fracs / clone_fracs.sum(), chromosomes))

    obs = pd.DataFrame(obs_rows)
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    for col in ("patient.id", "core_id", "cell_type"):
        obs[col] = obs[col].astype("category")
    adata = anndata.AnnData(
        X=scipy.sparse.csr_matrix(np.vstack(counts).astype(np.float32)),
        obs=obs,
        var=pd.DataFrame(index=gene_coords["gene_symbol"].to_list()),
    )

    return {
        "adata": adata,
        "gene_coords": gene_coords,
        "segments": pd.concat(segments, ignore_index=True),
        "true_cn": true_cn,
        "responsive": responsive,
    }


def _bulk_segments(pid, clone_events, clone_fracs, chromosomes):
    """Piecewise-constant bulk log2 ratio of a clone mixture, as a segment table."""
    rows = []
    for chrom in chromosomes:
        breaks = {0.0, float(CHROMOSOME_LENGTHS[chrom])}
        for events in clone_events:
            for e_chrom, start, end, _ in events:
                if e_chrom == chrom:
                    breaks.update((start, end))
        breaks = np.array(sorted(breaks))
        mids = (breaks[:-1] + breaks[1:]) / 2
        bulk = np.zeros(len(mids))
        for events, frac in zip(clone_events, clone_fracs):
            bulk += frac * 2.0 ** _cn_at(events, chrom, mids)
        for start, end, value in zip(breaks[:-1], breaks[1:], np.log2(bulk)):
            rows.append({
                "Chromosome": chrom,
                "Start_Position": int(start),
                "End_Position": int(end),
                "Patient_ID": pid,
                "SegmentMean": float(value),
            })
    return pd.DataFrame(rows)
