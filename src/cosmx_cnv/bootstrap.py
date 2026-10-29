"""
Bootstrap CN-expression model per gene.

Quantifies how strongly each gene's expression depends on its copy number
across patients, using a nonlinear dose-response model with bootstrap
resampling.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tqdm import tqdm


def model_cn(x, b, alpha, gamma):
    """mu = log(1 + b + alpha * 2^(gamma*x))"""
    return np.log1p(b + alpha * (2.0 ** (gamma * x)))


def fit_cn_bootstrap(
    x, y,
    n_boot=500,
    x_low=-0.4, x_high=0.4,
    x_grid=None,
    bounds=([0.0, 0.0, -5.0], [np.inf, 20.0, 5.0]),
    fillna_with_zero=True,
    random_state=None,
    maxfev=5000,
):
    """
    Bootstrap a nonlinear CN-expression model for a single gene.

    Parameters
    ----------
    x : array_like
        Copy number values (log-scale, from sWGS).
    y : array_like
        Expression values (log-scale).
    n_boot : int
        Number of bootstrap iterations.
    x_low, x_high : float
        CN values at which to evaluate the fitted model for computing delta.
    x_grid : array_like or None
        Grid of CN values for prediction curves.
    bounds : tuple of lists
        Parameter bounds for curve_fit: ([b_lo, alpha_lo, gamma_lo], [b_hi, alpha_hi, gamma_hi]).
    fillna_with_zero : bool
        If True, replace NaN with 0; otherwise drop NaN entries.
    random_state : int or None
        Seed for the bootstrap RNG.
    maxfev : int
        Maximum function evaluations for curve_fit.

    Returns
    -------
    bootout : dict or None
        Per-bootstrap arrays (params, predictions, delta, grid curves).
    summary : dict or None
        Aggregate statistics (n_bootstrap, support_positive, mean_delta, CIs, unexplained variance).
    """
    rng = np.random.default_rng(random_state)

    # Handle NaNs
    if fillna_with_zero:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

    n = len(x)
    if n < 5:
        return None, None

    if x_grid is None:
        x_grid = np.linspace(min(-1.0, np.nanmin(x)), max(1.0, np.nanmax(x)), 200)

    params = np.full((n_boot, 3), np.nan)
    y_low_s = np.full(n_boot, np.nan)
    y_high_s = np.full(n_boot, np.nan)
    deltas = np.full(n_boot, np.nan)
    y_grid_s = np.full((n_boot, x_grid.size), np.nan)
    unexplained_s = np.full(n_boot, np.nan)

    # start from a reasonable point, update as we go
    p0 = [max(1e-3, np.nanpercentile(y, 25)), 1.0, 0.5]

    for b_idx in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        try:
            popt, _ = curve_fit(
                model_cn, xb, yb,
                p0=p0,
                bounds=bounds,
                maxfev=maxfev
            )
        except Exception:
            try:
                popt, _ = curve_fit(
                    model_cn, xb, yb,
                    p0=[0.1, 1.0, 0.0],
                    bounds=bounds,
                    maxfev=maxfev
                )
            except Exception:
                continue

        params[b_idx, :] = popt
        p0 = popt  # warm start

        # In-sample fit quality
        y_pred = model_cn(xb, *popt)
        rss = np.sum((yb - y_pred) ** 2)
        tss = np.sum((yb - yb.mean()) ** 2)
        unexplained_s[b_idx] = rss / tss if tss > 0 else np.nan

        # Effect and grid predictions
        y_low_s[b_idx]  = model_cn(np.array([x_low]), *popt)[0]
        y_high_s[b_idx] = model_cn(np.array([x_high]), *popt)[0]
        deltas[b_idx]   = y_high_s[b_idx] - y_low_s[b_idx]
        y_grid_s[b_idx, :] = model_cn(x_grid, *popt)

    ok = np.isfinite(params).all(axis=1)
    params = params[ok]
    y_low_s = y_low_s[ok]
    y_high_s = y_high_s[ok]
    deltas = deltas[ok]
    y_grid_s = y_grid_s[ok]
    unexplained_s = unexplained_s[ok]

    if params.shape[0] == 0:
        return None, None

    def ci(a, lo=2.5, hi=97.5):
        return np.percentile(a, [lo, hi])

    grid_mean = np.nanmean(y_grid_s, axis=0)
    grid_lo, grid_hi = np.nanpercentile(y_grid_s, [2.5, 97.5], axis=0)

    summary = {
        "n_bootstrap": int(params.shape[0]),
        "support_positive": float(np.mean(deltas > 0.0)),
        "mean_delta": float(np.nanmean(deltas)),
        "ci_delta_lo": float(ci(deltas)[0]),
        "ci_delta_hi": float(ci(deltas)[1]),
        "unexplained_mean": float(np.nanmean(unexplained_s)),
        "unexplained_ci_lo": float(ci(unexplained_s)[0]),
        "unexplained_ci_hi": float(ci(unexplained_s)[1]),
    }

    bootout = {
        "params": params,
        "y_low": y_low_s,
        "y_high": y_high_s,
        "delta": deltas,
        "unexplained": unexplained_s,
        "x_grid": x_grid,
        "y_grid": y_grid_s,
        "grid_mean": grid_mean,
        "grid_lo": grid_lo,
        "grid_hi": grid_hi,
    }
    return bootout, summary


def bootstrap_all_genes(
    cn_table, expr_table,
    n_boot=500, x_low=-0.4, x_high=0.4, x_grid=None,
    bounds=([0.0, 0.0, -5.0], [np.inf, 20.0, 5.0]),
    fillna_with_zero=True, random_state=0, maxfev=5000,
    save_boots_dir=None
):
    """
    Run bootstrap CN-expression analysis for all common genes.

    Parameters
    ----------
    cn_table : DataFrame
        Copy number table (samples x genes).
    expr_table : DataFrame
        Expression table (samples x genes).
    n_boot : int
        Number of bootstrap iterations per gene.
    x_low, x_high : float
        CN values for delta computation.
    x_grid : array_like or None
        Grid for prediction curves.
    bounds : tuple
        Parameter bounds for curve_fit.
    fillna_with_zero : bool
        NaN handling strategy.
    random_state : int
        RNG seed.
    maxfev : int
        Max function evaluations for curve_fit.
    save_boots_dir : str or None
        If provided, save per-gene .npz files to this directory.

    Returns
    -------
    res_df : DataFrame
        Per-gene summary statistics.
    common_genes : list
        List of genes that were analyzed.
    """
    common_genes = cn_table.columns.intersection(expr_table.columns)
    common_genes = list(common_genes)
    results = []

    if save_boots_dir is not None:
        Path(save_boots_dir).mkdir(parents=True, exist_ok=True)

    for gene in tqdm(common_genes, desc="Bootstrapping genes"):
        x = cn_table[gene].to_numpy(float)
        y = expr_table[gene].to_numpy(float)
        bootout, summ = fit_cn_bootstrap(
            x, y,
            n_boot=n_boot,
            x_low=x_low, x_high=x_high,
            x_grid=x_grid,
            bounds=bounds,
            fillna_with_zero=fillna_with_zero,
            random_state=random_state,
            maxfev=maxfev,
        )
        if summ is None:
            continue
        summ["gene"] = gene
        results.append(summ)

        # Save per-gene boots to disk to control memory
        if save_boots_dir is not None and bootout is not None:
            np.savez_compressed(
                os.path.join(save_boots_dir, f"{gene}.npz"),
                **bootout
            )

    res_df = pd.DataFrame(results).sort_values("mean_delta", ascending=False).reset_index(drop=True)

    # Compute symmetric support from support_positive
    # If below 0.5, flip as 1 - p
    if "support_positive" in res_df.columns:
        sp = res_df["support_positive"].values.astype(float)
        res_df["support"] = np.where(sp >= 0.5, sp, 1.0 - sp)

    return res_df, common_genes
