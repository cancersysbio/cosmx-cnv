import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.distributions import constraints
from torch.distributions import transform_to

# Upper bound on the Dirichlet concentration alpha_q. Its posterior mass is
# bounded by the number of cells per patient (order 1e3 in these cohorts), so
# larger values are a runaway rather than signal -- and once alpha_q overflows
# to inf the Dirichlet constraint check aborts the whole run.
ALPHA_Q_MAX = 1e4

import matplotlib.pyplot as plt

from ._config import CNVModelConfig, px_to_um

# Set up logging
logger = logging.getLogger(__name__)


class CNVModel:
    """
    Copy Number Variation model using variational inference.

    Attributes:
        config: CNVModelConfig object containing model parameters
        losses: List of ELBO losses during training
    """

    def __init__(self, config: Optional[CNVModelConfig] = None):
        """
        Initialize CNV model.

        Args:
            config: Optional configuration object. If None, uses defaults.
        """
        self.config = config or CNVModelConfig()
        self.losses: List[float] = []

    def spatial_decay(self, distances: torch.Tensor) -> torch.Tensor:
        """Calculate spatial decay weights using Matern kernel."""
        sqrt3 = torch.sqrt(torch.tensor(3.0, device=self.config.device))
        d = sqrt3 * distances / self.config.spatial_decay_scale
        return (1.0 + d) * torch.exp(-d)

    def genomic_decay(self, distances: torch.Tensor) -> torch.Tensor:
        """Calculate genomic decay weights using Matern kernel."""
        sqrt3 = torch.sqrt(torch.tensor(3.0, device=self.config.device))
        d = sqrt3 * distances / self.config.gene_decay_scale
        return (1.0 + d) * torch.exp(-d)

    @config_enumerate
    def model(self, data: torch.Tensor, **kwargs):
        """
        Pyro model specification.

        Args:
            data: Tensor of shape (n_cells, n_genes)
            **kwargs: Additional model parameters
        """
        C, G = data.shape

        if "core_ids" in kwargs:
            num_cores = len(torch.unique(kwargs["core_ids"]))
        else:
            num_cores = 1

        if "tma_ids" in kwargs:
            num_tmas = len(torch.unique(kwargs["tma_ids"]))
        else:
            num_tmas = 1

        if "patient_ids" in kwargs:
            num_patients = len(torch.unique(kwargs["patient_ids"]))
        else:
            num_patients = 1
            kwargs["patient_ids"] = torch.zeros(C, dtype=torch.int64, device=self.config.device)

        # --- Priors ---
        # Handle reference cells in mixing proportions
        if self.config.use_reference and "reference_config" in kwargs:
            ref_config = kwargs["reference_config"]

            # Initialize patient-specific mixing proportions
            alpha = torch.ones(num_patients, self.config.num_components, dtype=data.dtype, device=self.config.device)
            # Increase prior for reference component
            alpha[:, list(ref_config['masks'].keys())] = 10.0
            pi = pyro.sample("pi", dist.Dirichlet(alpha).to_event(1))
        else:
            alpha = torch.ones(num_patients, self.config.num_components, dtype=data.dtype, device=self.config.device)
            pi = pyro.sample("pi", dist.Dirichlet(alpha).to_event(1))

        if "r_init" not in kwargs:
            kwargs["r_init"] = torch.zeros((1,G), device=self.config.device)
        r = pyro.sample("r", dist.Normal(
            kwargs["r_init"].unsqueeze(0),
            torch.ones((1,G), device=self.config.device) * 0.5
        ).to_event(1))

        slope = pyro.sample("slope", dist.LogNormal(
            torch.ones((1,G), device=self.config.device),
            torch.ones((1,G), device=self.config.device) * 0.01
        ).to_event(1))
        slope = 1.0


        # TMA-specific global bias — coarser slide/staining batch (looser prior, absorbs the dominant effect)
        if "tma_ids" in kwargs:
            tma_bias = pyro.sample(
                "tma_bias",
                dist.Normal(
                    torch.zeros(num_tmas, device=self.config.device),
                    0.05 * torch.ones(num_tmas, device=self.config.device)
                ).to_event(1)
            )

        # Core-specific residual bias within a TMA (tighter prior than tma)
        if "core_ids" in kwargs:
            core_bias = pyro.sample(
                "core_bias",
                dist.Normal(
                    torch.zeros(num_cores, device=self.config.device),
                    0.02 * torch.ones(num_cores, device=self.config.device)
                ).to_event(1)
            )

        cell_sensitivity = pyro.sample(
        "cell_sensitivity",
        dist.Normal(
            torch.zeros(C, device=self.config.device),
            torch.ones(C, device=self.config.device)
        ).to_event(1)
    )


        # Patient-specific components
        with pyro.plate("patients", num_patients, dim=-2):
            with pyro.plate("components", self.config.num_components, dim=-1) as k:

                sigma_rate = torch.where(
                        torch.arange(self.config.num_components, device=self.config.device) == 0,
                        torch.tensor(30., device=self.config.device),
                        torch.tensor(10., device=self.config.device)
                    )


                sigma = pyro.sample(
                    "sigma",
                    dist.Exponential(sigma_rate)
                )

                if self.config.use_reference:
                    # Create a full tensor for s
                    s_full = torch.zeros(
                        (num_patients, self.config.num_components, G),
                        device=self.config.device
                    )

                    # Create mask for components and patients that need to be sampled
                    sample_mask = torch.ones((num_patients, self.config.num_components), dtype=torch.bool, device=self.config.device)

                    # Fill in reference profiles where provided and update sampling mask
                    for comp_idx, patient_profiles in ref_config['profiles'].items():
                        if patient_profiles is not None:  # Skip neutral component
                            for patient_idx, profile in patient_profiles.items():
                                s_full[patient_idx, comp_idx] = profile
                                sample_mask[patient_idx, comp_idx] = False

                    # Sample s only for non-reference components and patients
                    s_nonref = pyro.sample(
                        "s_nonzero",
                        dist.Laplace(
                            torch.zeros_like(s_full),
                            sigma.unsqueeze(-1)
                        )
                        .mask(sample_mask.unsqueeze(-1))
                        .to_event(1)
                    )

                    # Fill the non-reference entries
                    s_full[sample_mask] = s_nonref[sample_mask]
                    s = s_full
                else:
                    # Original unconstrained sampling for all components
                    s = pyro.sample(
                        "s_nonzero",
                        dist.Laplace(
                            torch.zeros(num_patients, self.config.num_components, G, device=self.config.device),
                            sigma.unsqueeze(-1)
                        ).to_event(1)
                    )

        # --- Likelihood ---
        if self.config.likelihood == 'negative_binomial':
            concentration = pyro.sample(
                "concentration",
                dist.Gamma(
                    torch.tensor(15., device=self.config.device).expand(G),
                    torch.tensor(0.5, device=self.config.device).expand(G)
                ).to_event(1)
            )

        # Split cells into reference and non-reference
        if self.config.use_reference and ("reference_config" in kwargs):
            # Convert dictionary of masks to a single combined mask
            ref_mask = torch.zeros(C, dtype=torch.bool, device=self.config.device)
            for comp_idx, mask in kwargs["reference_config"]["masks"].items():
                ref_mask |= mask.bool().to(self.config.device)

            non_ref_mask = ~ref_mask

            # Handle non-reference cells with enumeration
            with pyro.plate("non_ref_cells", non_ref_mask.sum()) as non_ref_idx:
                z_cancer = pyro.sample(
                    "z_cancer",
                    dist.Categorical(pi[kwargs["patient_ids"][non_ref_mask]]),
                    infer={"enumerate": "parallel"}
                )
                s_z_cancer = s[kwargs["patient_ids"][non_ref_mask], z_cancer]

                # Add core + tma bias if present
                batch_effect_cancer = torch.zeros_like(cell_sensitivity[non_ref_mask].unsqueeze(-1))
                if "core_ids" in kwargs:
                    batch_effect_cancer = batch_effect_cancer + core_bias[kwargs["core_ids"][non_ref_mask]].unsqueeze(-1)
                if "tma_ids" in kwargs:
                    batch_effect_cancer = batch_effect_cancer + tma_bias[kwargs["tma_ids"][non_ref_mask]].unsqueeze(-1)
                rate_cancer = torch.exp(
                    s_z_cancer * slope + r + batch_effect_cancer +
                    cell_sensitivity[non_ref_mask].unsqueeze(-1)
                )

                # Sample observations for non-reference cells
                if self.config.likelihood == 'negative_binomial':
                    pyro.sample(
                        "obs_cancer",
                        dist.GammaPoisson(
                            concentration=concentration.expand(non_ref_mask.sum(), G),
                            rate=concentration / rate_cancer
                        ).to_event(1),
                        obs=data[non_ref_mask]
                    )
                else:
                    pyro.sample("obs_cancer", dist.Poisson(rate_cancer).to_event(1), obs=data[non_ref_mask])

            # Handle reference cells without enumeration
            with pyro.plate("ref_cells", ref_mask.sum()) as ref_idx:
                # For each reference component, handle its cells separately
                for comp_idx, comp_mask in kwargs["reference_config"]["masks"].items():
                    comp_mask = comp_mask.bool().to(self.config.device)
                    if comp_mask.any():
                        # Use the specified component for these reference cells
                        s_z_ref = s[kwargs["patient_ids"][comp_mask], comp_idx]

                        # Add core + tma bias if present
                        batch_effect_ref = torch.zeros_like(cell_sensitivity[comp_mask].unsqueeze(-1))
                        if "core_ids" in kwargs:
                            batch_effect_ref = batch_effect_ref + core_bias[kwargs["core_ids"][comp_mask]].unsqueeze(-1)
                        if "tma_ids" in kwargs:
                            batch_effect_ref = batch_effect_ref + tma_bias[kwargs["tma_ids"][comp_mask]].unsqueeze(-1)
                        rate_ref = torch.exp(
                            s_z_ref * slope + r + batch_effect_ref +
                            cell_sensitivity[comp_mask].unsqueeze(-1)
                        )

                        # Sample observations for reference cells
                        if self.config.likelihood == 'negative_binomial':
                            pyro.sample(
                                f"obs_ref_{comp_idx}",
                                dist.GammaPoisson(
                                    concentration=concentration.expand(comp_mask.sum(), G),
                                    rate=concentration / rate_ref
                                ).to_event(1),
                                obs=data[comp_mask]
                            )
                        else:
                            pyro.sample(f"obs_ref_{comp_idx}", dist.Poisson(rate_ref).to_event(1), obs=data[comp_mask])
        else:
            # Original behavior when no reference cells
            with pyro.plate("cells", C) as cell_idx:
                z = pyro.sample(
                    "z",
                    dist.Categorical(pi[kwargs["patient_ids"]]),
                    infer={"enumerate": "parallel"}
                )
                s_z = s[kwargs["patient_ids"], z]

                batch_effect = torch.zeros_like(cell_sensitivity.unsqueeze(-1))
                if "core_ids" in kwargs:
                    batch_effect = batch_effect + core_bias[kwargs["core_ids"]].unsqueeze(-1)
                if "tma_ids" in kwargs:
                    batch_effect = batch_effect + tma_bias[kwargs["tma_ids"]].unsqueeze(-1)
                rate = torch.exp(
                    s_z * slope + r + batch_effect +
                    cell_sensitivity.unsqueeze(-1)
                )


                if self.config.likelihood == 'negative_binomial':
                    pyro.sample(
                        "obs",
                        dist.GammaPoisson(
                            concentration=concentration.expand(C, G),
                            rate=concentration / rate
                        ).to_event(1),
                        obs=data
                    )
                else:
                    pyro.sample("obs", dist.Poisson(rate).to_event(1), obs=data)

        if "gene_neighbours_adj" in kwargs:
            with pyro.plate("pairs_gene", kwargs["gene_neighbours_adj"].size(0)):
                i, j = kwargs["gene_neighbours_adj"][:, 0], kwargs["gene_neighbours_adj"][:, 1]
                # Sum over patients and components
                diff = s[:, :, i] - s[:, :, j]  # shape (P, K, num_pairs)

                dist_weights = self.genomic_decay(kwargs["gene_neighbours_dist"])

                penalty_per_pair = -self.config.gene_proximity_penalty * dist_weights * (diff**2).sum(dim=(0,1))
                pyro.factor("gene_proximity_penalty", penalty_per_pair, has_rsample=True)




    @config_enumerate
    def guide(self, data: torch.Tensor, **kwargs):
        """
        Variational guide (approximate posterior).

        Args:
            data: Tensor of shape (n_cells, n_genes)
            **kwargs: Additional guide parameters
        """
        C, G = data.shape

        if "core_ids" in kwargs:
            num_cores = len(torch.unique(kwargs["core_ids"]))
        else:
            num_cores = 1

        if "tma_ids" in kwargs:
            num_tmas = len(torch.unique(kwargs["tma_ids"]))
        else:
            num_tmas = 1

        if "patient_ids" in kwargs:
            num_patients = len(torch.unique(kwargs["patient_ids"]))
        else:
            num_patients = 1

        # cell sensitivity parameters
        cell_sensitivity_loc = pyro.param(
            "cell_sensitivity_loc",
            torch.zeros(C, device=self.config.device)
        )
        cell_sensitivity_scale = pyro.param(
            "cell_sensitivity_scale",
            0.1 * torch.ones(C, device=self.config.device),
            constraint=constraints.positive
        )
        cell_sensitivity = pyro.sample(
            "cell_sensitivity",
            dist.Normal(cell_sensitivity_loc, cell_sensitivity_scale).to_event(1)
        )


        # q(pi): Patient-specific Dirichlet
        alpha_q = pyro.param(
            "alpha_q",
            10. * torch.ones(num_patients, self.config.num_components, device=self.config.device),
            constraint=constraints.positive
        )
        pi_q = pyro.sample("pi", dist.Dirichlet(alpha_q).to_event(1))

        # q(r): Normal with learnable mean, scale per gene
        if "r_init" not in kwargs:
            kwargs["r_init"] = torch.zeros((1,G), device=self.config.device)
        r_loc = pyro.param("r_loc", kwargs["r_init"])
        r_scale = pyro.param(
            "r_scale",
            0.1 * torch.ones((1,G), device=self.config.device),
            constraint=constraints.positive
        )
        r_q = pyro.sample("r", dist.Normal(r_loc, r_scale).to_event(1))


        slope_loc = pyro.param("slope_loc", torch.ones((1,G), device=self.config.device))
        slope_scale = pyro.param(
            "slope_scale",
            0.01 * torch.ones((1,G), device=self.config.device),
            constraint=constraints.positive
        )
        slope_q = pyro.sample("slope", dist.LogNormal(slope_loc, slope_scale).to_event(1))


        # q(tma_bias): Normal with learnable mean, scale per TMA
        if "tma_ids" in kwargs:
            tma_bias_loc = pyro.param("tma_bias_loc",
                torch.zeros(num_tmas, device=self.config.device))
            tma_bias_scale = pyro.param("tma_bias_scale",
                0.1 * torch.ones(num_tmas, device=self.config.device),
                constraint=constraints.positive)
            pyro.sample("tma_bias", dist.Normal(tma_bias_loc, tma_bias_scale).to_event(1))

        # q(core_bias): Normal with learnable mean, scale per core
        if "core_ids" in kwargs:
            core_bias_loc = pyro.param("core_bias_loc",
                torch.zeros(num_cores, device=self.config.device))
            core_bias_scale = pyro.param("core_bias_scale",
                0.1 * torch.ones(num_cores, device=self.config.device),
                constraint=constraints.positive)
            pyro.sample("core_bias", dist.Normal(core_bias_loc, core_bias_scale).to_event(1))

        # Patient-specific components
        with pyro.plate("patients", num_patients, dim=-2):
            with pyro.plate("components", self.config.num_components, dim=-1) as k:

                sigma_rate = pyro.param(
                    "sigma_rate",
                    torch.cat([
                        torch.full((num_patients, 1), -5.0, device=self.config.device),
                        torch.zeros(num_patients, self.config.num_components-1, device=self.config.device)
                    ], dim=1)
                )


                sigma_q = pyro.sample(
                    "sigma",
                    dist.LogNormal(sigma_rate, 0.1)
                )

                if self.config.use_reference:
                    ref_config = kwargs["reference_config"]

                    # Create mask for components and patients that need to be sampled
                    sample_mask = torch.ones((num_patients, self.config.num_components), dtype=torch.bool, device=self.config.device)

                    # Update sampling mask based on provided reference profiles
                    for comp_idx, patient_profiles in ref_config['profiles'].items():
                        if patient_profiles is not None:  # Skip neutral component
                            for patient_idx in patient_profiles:
                                sample_mask[patient_idx, comp_idx] = False

                    # Create parameters only for components/patients that need to be learned
                    s_loc = pyro.param(
                        "s_loc",
                        torch.zeros(num_patients, self.config.num_components, G, device=self.config.device)
                    )
                    s_scale = pyro.param(
                        "s_scale",
                        0.1 * torch.ones(num_patients, self.config.num_components, G, device=self.config.device),
                        constraint=constraints.positive
                    )

                    # Sample only for non-reference entries
                    s_nonref = pyro.sample(
                        "s_nonzero",
                        dist.Normal(s_loc, s_scale)
                        .mask(sample_mask.unsqueeze(-1))
                        .to_event(1)
                    )
                else:
                    # Unconstrained sampling
                    s_loc = pyro.param(
                        "s_loc",
                        torch.zeros(num_patients, self.config.num_components, G, device=self.config.device)
                    )
                    s_scale = pyro.param(
                        "s_scale",
                        0.1 * torch.ones(num_patients, self.config.num_components, G, device=self.config.device),
                        constraint=constraints.positive
                    )
                    s_full = pyro.sample(
                        "s_nonzero",
                        dist.Normal(s_loc, s_scale).to_event(1)
                    )
        # q(z_c): For each non-reference cell, a categorical distribution
        if self.config.use_reference and ("reference_config" in kwargs):
            # Convert dictionary of masks to a single combined mask
            ref_mask = torch.zeros(C, dtype=torch.bool, device=self.config.device)
            for comp_idx, mask in kwargs["reference_config"]["masks"].items():
                ref_mask |= mask.bool().to(self.config.device)

            non_ref_mask = ~ref_mask

            # Only need logits for non-reference cells
            z_logits = pyro.param(
                "z_logits",
                torch.zeros(non_ref_mask.sum(), self.config.num_components, device=self.config.device)
            )

            # Handle spatial penalty if present
            if "cell_neighbours_adj" in kwargs:
                # Need to adjust adjacency indices to account for reference cells
                adj_mask = (
                    non_ref_mask[kwargs["cell_neighbours_adj"][:, 0]] &
                    non_ref_mask[kwargs["cell_neighbours_adj"][:, 1]]
                )
                if adj_mask.any():
                    # Create mapping from original indices to non-ref indices
                    non_ref_idx_map = torch.cumsum(non_ref_mask, 0) - 1

                    with pyro.plate("pairs_cell", adj_mask.sum()):
                        i = non_ref_idx_map[kwargs["cell_neighbours_adj"][adj_mask, 0]]
                        j = non_ref_idx_map[kwargs["cell_neighbours_adj"][adj_mask, 1]]
                        diff = z_logits[i, :] - z_logits[j, :]

                        dist_weights = self.spatial_decay(kwargs["cell_neighbours_dist"][adj_mask])

                        penalty_per_pair = -self.config.spatial_penalty * dist_weights * (diff**2).sum(dim=-1)
                        pyro.factor("spatial_penalty", penalty_per_pair, has_rsample=False)

            # Sample z only for non-reference cells
            with pyro.plate("non_ref_cells", non_ref_mask.sum()):
                pyro.sample(
                    "z_cancer",
                    dist.Categorical(logits=z_logits)
                )

            # No need to sample for reference cells as they're fixed to their respective components
        else:
            # Original behavior when no reference cells
            z_logits = pyro.param(
                "z_logits",
                torch.zeros(C, self.config.num_components, device=self.config.device)
            )

            if "cell_neighbours_adj" in kwargs:
                with pyro.plate("pairs_cell", kwargs["cell_neighbours_adj"].size(0)):
                    i, j = kwargs["cell_neighbours_adj"][:, 0], kwargs["cell_neighbours_adj"][:, 1]
                    diff = z_logits[i, :] - z_logits[j, :]

                    dist_weights = self.spatial_decay(kwargs["cell_neighbours_dist"])

                    penalty_per_pair = -self.config.spatial_penalty * dist_weights * (diff**2).sum(dim=-1)
                    pyro.factor("spatial_penalty", penalty_per_pair, has_rsample=False)

            with pyro.plate("cells", C):
                pyro.sample(
                    "z",
                    dist.Categorical(logits=z_logits)
                )

    def fit(
        self,
        data: torch.Tensor,
        num_iterations: int = 2000,
        lr: float = 1e-2,
        clip_norm: float = 10.0,
        **kwargs
    ) -> None:
        """
        Run variational inference to fit the model.

        Args:
            data: Input data tensor
            num_iterations: Number of optimization iterations
            lr: Learning rate
            clip_norm: Max gradient L2-norm per step (ClippedAdam). Prevents
                gradient spikes from blowing up parameters (e.g. the Dirichlet
                concentration alpha_q) into inf/NaN. Set None/<=0 to disable.
            **kwargs: Additional training parameters

        Raises:
            RuntimeError: If optimization fails
        """
        if self.config.use_reference and kwargs.get("reference_config") is None:
            raise ValueError("use_reference=True requires a reference_config")

        try:
            # Move data to device
            data = data.to(self.config.device)
            new_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    new_kwargs[k] = v.to(self.config.device)
                else:
                    new_kwargs[k] = v

            kwargs = new_kwargs

            # Initialize optimizer. ClippedAdam bounds the per-step gradient
            # norm, which stops rare gradient spikes from pushing parameters
            # (notably the Dirichlet concentration alpha_q) to inf/NaN.
            pyro.clear_param_store()
            if clip_norm and clip_norm > 0:
                optimizer = optim.ClippedAdam({"lr": lr, "clip_norm": clip_norm})
            else:
                optimizer = optim.Adam({"lr": lr})
            svi = SVI(self.model, self.guide, optimizer,
                     loss=TraceEnum_ELBO())

            # Store kwargs for parameter reconstruction
            self.last_kwargs = kwargs

            # Training loop
            self.losses = []
            pbar = tqdm(range(num_iterations), desc="Training")

            # NaN-guard: keep a detached snapshot of the last finite param
            # state. If a step yields a non-finite loss or raises because a
            # parameter went inf/NaN, roll back to the snapshot and skip the
            # step instead of aborting the whole (multi-hour) run.
            param_store = pyro.get_param_store()
            alpha_q_max_unc = transform_to(constraints.positive).inv(
                torch.tensor(ALPHA_Q_MAX, device=self.config.device)
            ).item()
            last_good = None
            n_skipped = 0
            max_consecutive_skips = 50
            consecutive_skips = 0

            # Snapshot/restore operate on the UNCONSTRAINED tensors in
            # param_store._params, not on param_store[name]. For a constrained
            # param (alpha_q, every *_scale, sigma_rate) __getitem__ returns a
            # freshly transform_to()'d tensor, so writing to its .data lands in
            # a temporary that is discarded -- the rollback silently did nothing
            # for exactly the params that blow up.
            def _snapshot():
                return {name: p.detach().clone()
                        for name, p in param_store._params.items()}

            def _restore(snap):
                with torch.no_grad():
                    for name, val in snap.items():
                        param_store._params[name].data.copy_(val)
                # Restoring params is not enough: a NaN gradient passes through
                # ClippedAdam's element-wise clamp untouched and poisons
                # exp_avg/exp_avg_sq forever, so every later step re-emits NaN
                # regardless of the param values. Drop non-finite moments.
                for p, opt in optimizer.optim_objs.items():
                    state = getattr(opt, "state", {}).get(p)
                    if not state:
                        continue
                    for key in ("exp_avg", "exp_avg_sq"):
                        buf = state.get(key)
                        if buf is not None and not torch.isfinite(buf).all():
                            buf.zero_()

            for step in pbar:
                try:
                    loss = svi.step(data, **kwargs)
                    bad_step = (not np.isfinite(loss)) or any(
                        not torch.isfinite(param_store[name]).all()
                        for name in param_store.keys()
                    )
                except Exception as e:
                    # A step raised (typically a constraint failure once a
                    # parameter has already gone NaN). Treat as a bad step.
                    logger.warning(f"Step {step} raised ({e}); rolling back")
                    bad_step = True
                    loss = float('nan')

                if bad_step:
                    if last_good is None:
                        logger.error(
                            f"Non-finite loss/params at step {step} with no "
                            f"prior good state to roll back to")
                        raise RuntimeError("Training failed: diverged before "
                                           "any finite step")
                    _restore(last_good)
                    n_skipped += 1
                    consecutive_skips += 1
                    if consecutive_skips >= max_consecutive_skips:
                        logger.error(
                            f"Aborting: {consecutive_skips} consecutive "
                            f"non-finite steps at step {step}; not recovering")
                        raise RuntimeError("Training failed: persistent divergence")
                    pbar.set_postfix({'ELBO': 'skip(NaN)', 'skipped': n_skipped})
                    continue

                # Keep alpha_q below ALPHA_Q_MAX so it can never reach the
                # overflow regime that aborts the run. Clamp in unconstrained
                # space, deriving the bound through the same transform pyro
                # uses rather than assuming it is exp.
                with torch.no_grad():
                    alpha_unc = param_store._params.get("alpha_q")
                    if alpha_unc is not None:
                        alpha_unc.clamp_(max=alpha_q_max_unc)

                consecutive_skips = 0
                last_good = _snapshot()
                self.losses.append(loss)

                if len(self.losses) >= 2:
                    loss_diff = self.losses[-2] - loss
                    pbar.set_postfix({
                        'ELBO': f'{loss:.2f}',
                        'ΔELBO': f'{loss_diff:.2e}',
                        'skipped': n_skipped,
                    })
                else:
                    pbar.set_postfix({'ELBO': f'{loss:.2f}'})

            if n_skipped:
                logger.warning(
                    f"Training finished with {n_skipped} skipped (non-finite) "
                    f"steps out of {num_iterations}")

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Extract learned variational parameters, reconstructing full tensors including reference cells.

        Returns:
            Dictionary of parameter names to tensors
        """
        try:
            params = {}

            # Get Dirichlet parameters
            alpha_q = pyro.param("alpha_q").detach()
            params["pi"] = alpha_q / alpha_q.sum(dim=-1, keepdim=True)

            # Get basic parameters that don't need reconstruction
            for name in ["r_loc", "r_scale", "slope_loc", "slope_scale", "sigma_rate", "s_loc",
                        "s_scale", "cell_sensitivity_loc", "cell_sensitivity_scale"]:
                params[name] = pyro.param(name).detach()

            # Get optional core bias parameters
            if "core_bias_loc" in pyro.get_param_store():
                params["core_bias_loc"] = pyro.param("core_bias_loc").detach()
                params["core_bias_scale"] = pyro.param("core_bias_scale").detach()

            # Get optional tma bias parameters
            if "tma_bias_loc" in pyro.get_param_store():
                params["tma_bias_loc"] = pyro.param("tma_bias_loc").detach()
                params["tma_bias_scale"] = pyro.param("tma_bias_scale").detach()

            # Reconstruct full z_logits including reference cells
            z_logits = pyro.param("z_logits").detach()

            # Get total number of cells from cell_sensitivity parameters
            C = params["cell_sensitivity_loc"].shape[0]
            K = self.config.num_components

            # Initialize full z_logits with zeros
            z_logits_full = torch.zeros((C, K), device=self.config.device)

            if self.config.use_reference:
                # Get reference configuration from last run
                ref_config = self.last_kwargs.get("reference_config")
                if ref_config:
                    # Create reference mask
                    ref_mask = torch.zeros(C, dtype=torch.bool, device=self.config.device)
                    for comp_idx, mask in ref_config["masks"].items():
                        ref_mask |= mask.bool().to(self.config.device)
                        # Set very high logit for the correct component in reference cells
                        z_logits_full[mask.bool(), comp_idx] = 100.0  # Large positive value

                    # Fill in learned logits for non-reference cells
                    z_logits_full[~ref_mask] = z_logits
            else:
                z_logits_full = z_logits

            params["z_logits"] = z_logits_full

            # Reconstruct full s_loc including reference profiles
            if self.config.use_reference and "reference_config" in self.last_kwargs:
                s_loc = params["s_loc"]
                ref_config = self.last_kwargs["reference_config"]

                # Create full s_loc tensor including reference profiles
                s_loc_full = s_loc.clone()

                # Fill in reference profiles
                if 'profiles' in ref_config:
                    for comp_idx, patient_profiles in ref_config['profiles'].items():
                        if patient_profiles is not None:
                            for patient_idx, profile in patient_profiles.items():
                                s_loc_full[patient_idx, comp_idx] = profile

                params["s_loc_full"] = s_loc_full

            return params

        except Exception as e:
            logger.error(f"Error getting parameters: {str(e)}")
            raise

    def get_membership_probs(self) -> torch.Tensor:
        """
        Get cell membership probabilities.

        Returns:
            Tensor of membership probabilities
        """
        z_logits = pyro.param("z_logits").detach()
        return F.softmax(z_logits, dim=-1)

    def plot_decay_functions(self, figsize=(9, 4)):
        """
        Plot spatial and genomic decay functions.

        Args:
            figsize: Tuple of figure dimensions (width, height)
        """

        plt.figure(figsize=figsize)

        # Spatial decay
        plt.subplot(1, 2, 1)
        spatial_dist = torch.linspace(0, 3000, 1000, device=self.config.device)
        spatial_decay = self.spatial_decay(spatial_dist).detach().cpu().numpy()
        plt.plot(px_to_um(spatial_dist.cpu().numpy()), spatial_decay)

        # Add reference lines
        for x in [20, 40, 60]:
            plt.axvline(x, color='r', lw=1, ls='--')

        plt.title('Spatial decay')
        plt.ylabel('Decay weight')
        plt.xlabel('Distance (um)')
        plt.gca().spines[['right', 'top']].set_visible(False)

        # Genomic decay
        plt.subplot(1, 2, 2)
        genomic_dist = torch.linspace(0, 5e6, 1000, device=self.config.device)
        genomic_decay = self.genomic_decay(genomic_dist).detach().cpu().numpy()
        plt.plot(genomic_dist.cpu().numpy(), genomic_decay)

        # Add reference line
        plt.axvline(1e6, color='r', lw=1, ls='--')

        plt.xlabel('Distance (bp)')
        plt.title('Genomic decay')
        plt.ylabel('Decay weight')
        plt.gca().spines[['right', 'top']].set_visible(False)

        plt.tight_layout()

    def plot_training_progress(self, figsize=(10, 5), start=None, end=None):
        """
        Plot training loss over iterations.

        Args:
            figsize: Tuple of figure dimensions (width, height)
            start: Starting iteration index (optional)
            end: Ending iteration index (optional)
        """
        plt.figure(figsize=figsize)
        losses = self.losses[start:end]
        iterations = range(len(losses))
        if start is not None:
            iterations = range(start, start + len(losses))
        plt.plot(iterations, losses)
        plt.xlabel('Iteration')
        plt.ylabel('ELBO Loss')
        plt.title('Training Progress')
        plt.gca().spines[['right', 'top']].set_visible(False)
        plt.tight_layout()

    def get_overdispersion(self) -> Optional[torch.Tensor]:
        """
        Get learned overdispersion parameters for negative binomial model.

        Returns:
            Tensor of overdispersion parameters if using negative binomial,
            None if using Poisson
        """
        if self.config.likelihood == 'negative_binomial':
            concentration_loc = pyro.param("concentration_loc").detach()
            return 1.0 / concentration_loc  # Convert concentration to overdispersion
        return None
