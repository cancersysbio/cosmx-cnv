"""Bayesian inference of clonal copy number variation from CosMX spatial transcriptomics."""

__version__ = "0.1.0"

# Core model
from ._config import CNVModelConfig, px_to_um, um_to_px
from ._model import CNVModel
from ._data import prepare_data, prepare_chromosome_data, get_nearest_neighbours, get_adjacency_from_knn, align_panel_genes

# Gene selection
from ._gene_selection import get_gene_cnv_data, filter_genes_by_bootstrap, pseudobulk_expression

# Post-inference results
from ._results import create_clone_state_table, create_combined_clone_state_table

# Bootstrap
from .bootstrap import model_cn, fit_cn_bootstrap, bootstrap_all_genes

# Synthetic data
from .simulate import simulate_dataset

# Tools
from .tools.math_tools import FixedBandwidthKDE
from .tools.gene_tools import lookup_ensembl_coordinates
from .tools.genome_info import CHROMOSOME_LENGTHS, CENTROMERE_POSITIONS, CYTOBAND_COORDINATES

__all__ = [
    # Config
    'CNVModelConfig', 'px_to_um', 'um_to_px',
    # Model
    'CNVModel',
    # Data preparation
    'prepare_data', 'prepare_chromosome_data', 'align_panel_genes',
    'get_nearest_neighbours', 'get_adjacency_from_knn',
    # Gene selection
    'get_gene_cnv_data', 'filter_genes_by_bootstrap', 'pseudobulk_expression',
    # Results
    'create_clone_state_table', 'create_combined_clone_state_table',
    # Bootstrap
    'model_cn', 'fit_cn_bootstrap', 'bootstrap_all_genes',
    # Synthetic data
    'simulate_dataset',
    # Tools
    'FixedBandwidthKDE', 'lookup_ensembl_coordinates',
    'CHROMOSOME_LENGTHS', 'CENTROMERE_POSITIONS', 'CYTOBAND_COORDINATES',
]
