"""Command-line interface for cosmx-cnv bootstrap analysis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_all_genes


def main():
    ap = argparse.ArgumentParser(description="Bootstrap CN-expression model per gene.")
    ap.add_argument("--expr_csv", required=True, help="Expression CSV with samples as rows and genes as columns. Index is sample ID.")
    ap.add_argument("--cn_csv",   required=True, help="CN CSV with genes as rows and samples as columns, or vice versa. Will align by genes and samples.")
    ap.add_argument("--outdir",   required=True, help="Output directory.")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--x_low", type=float, default=-0.4)
    ap.add_argument("--x_high", type=float, default=0.4)
    ap.add_argument("--grid_min", type=float, default=-1.0)
    ap.add_argument("--grid_max", type=float, default=1.0)
    ap.add_argument("--grid_n", type=int, default=200)
    ap.add_argument("--fillna_with_zero", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--maxfev", type=int, default=5000)
    ap.add_argument("--save_boots_dir", type=str, default="boots", help="Directory under outdir to store per-gene .npz boot files.")
    ap.add_argument("--cn_has_genes_in_index", action="store_true", help="Set if CN CSV uses genes as index and samples as columns.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    boots_dir = outdir / args.save_boots_dir
    boots_dir.mkdir(parents=True, exist_ok=True)

    # Load expression, ensure index is string to match sample IDs
    expr_df = pd.read_csv(args.expr_csv, index_col=0, dtype={0: str})
    expr_df = expr_df[~expr_df.index.isna()]

    # Load CN and normalize orientation
    cn_df = pd.read_csv(args.cn_csv, index_col=0)

    # Determine common samples
    expr_samples = set(map(str, np.unique(expr_df.index)))
    cn_samples = set(map(str, cn_df.columns.astype(str)))
    common_samples = np.array(sorted(expr_samples & cn_samples, key=lambda s: (len(s), s)))
    if common_samples.size == 0:
        raise ValueError("No overlapping samples between expression and CN tables.")

    # Determine common genes
    expr_genes = set(map(str, np.unique(expr_df.columns)))
    cn_genes = set(map(str, np.unique(cn_df.index)))
    common_genes = np.array(sorted(expr_genes & cn_genes))
    if common_genes.size == 0:
        raise ValueError("No overlapping genes between expression and CN tables.")

    # Align CN: genes by rows, samples by columns
    cn_formated_table = cn_df.loc[common_genes, common_samples].T

    # Align expression: samples by rows, genes by columns
    expr_formatted_table = expr_df.loc[common_samples, common_genes]
    expr_formatted_table = expr_formatted_table[~expr_formatted_table.index.duplicated(keep="last")]

    x_grid = np.linspace(args.grid_min, args.grid_max, args.grid_n)
    fillna_flag = True if args.fillna_with_zero.lower() == "true" else False

    res_df, gene_list = bootstrap_all_genes(
        cn_table=cn_formated_table,
        expr_table=expr_formatted_table,
        n_boot=args.n_boot,
        x_low=args.x_low,
        x_high=args.x_high,
        x_grid=x_grid,
        bounds=([0.0, 0.0, -5.0], [np.inf, 20.0, 5.0]),
        fillna_with_zero=fillna_flag,
        random_state=args.random_state,
        maxfev=args.maxfev,
        save_boots_dir=str(boots_dir),
    )

    # Save tabular results
    res_path = outdir / "res_df.csv"
    res_df.to_csv(res_path, index=False)

    # Save metadata about boots on disk
    with open(outdir / "boots_index.json", "w") as f:
        json.dump({
            "boots_dir": str(boots_dir),
            "genes": gene_list,
            "x_low": args.x_low,
            "x_high": args.x_high,
            "grid_min": args.grid_min,
            "grid_max": args.grid_max,
            "grid_n": args.grid_n,
            "n_boot": args.n_boot,
            "fillna_with_zero": fillna_flag,
            "random_state": args.random_state,
        }, f, indent=2)

    print(f"Saved res_df to: {res_path}")
    print(f"Saved per-gene boot files to: {boots_dir}")
    print(f"Saved boots index to: {outdir / 'boots_index.json'}")


if __name__ == "__main__":
    main()
