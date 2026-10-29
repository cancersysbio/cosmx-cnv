from dataclasses import dataclass


@dataclass
class CNVModelConfig:
    """Configuration for CNV Model parameters."""
    num_components: int = 6
    spatial_penalty: float = 0.1
    gene_proximity_penalty: float = 0.1
    spatial_decay_scale: float = 1000.0
    gene_decay_scale: float = 1e6
    device: str = 'cpu'
    likelihood: str = 'poisson'  # 'poisson' or 'negative_binomial'
    use_reference: bool = False  # Whether to use reference cells


def px_to_um(px: float) -> float:
    """Convert pixels to micrometers."""
    return px / 4256 * 0.51 * 1e3


def um_to_px(um: float) -> float:
    """Convert micrometers to pixels."""
    return um / (0.51 * 1e3) * 4256
