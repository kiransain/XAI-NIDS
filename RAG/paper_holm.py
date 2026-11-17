"""Holm-Bonferroni correction.

Usage:
  python holm.py 0.0047 0.007 0.03 0.2          # p-values on the command line
  python holm.py --csv stats_tests.csv          # CSV with a p-value column (default: p_raw)
  python holm.py --csv stats_tests.csv --group comparison
      (corrects within each group, e.g. once per comparison across its metrics)
"""
import argparse

import numpy as np
import pandas as pd


def holm(p):
    """Holm-Bonferroni adjusted p-values (NaN values are ignored and stay NaN)."""
    p = np.asarray(p, dtype=float)
    adj = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])  # keeps adjusted values monotone
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser(description="Holm-Bonferroni correction")
    ap.add_argument("pvalues", nargs="*", type=float, help="raw p-values")
    ap.add_argument("--csv", help="CSV file with a column of raw p-values")
    ap.add_argument("--column", default="p_raw", help="name of the p-value column (default: p_raw)")
    ap.add_argument("--group", help="column to correct within (e.g. comparison)")
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level (default: 0.05)")
    ap.add_argument("--out", help="output CSV (default: <input>_holm.csv)")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        if args.group:
            df["p_holm"] = df.groupby(args.group)[args.column].transform(lambda s: pd.Series(holm(s), index=s.index))
        else:
            df["p_holm"] = holm(df[args.column].values)
        df["significant_holm"] = df["p_holm"] < args.alpha
        out = args.out or args.csv.rsplit(".", 1)[0] + "_holm.csv"
        df.to_csv(out, index=False)
        print(f"Saved {out}")
        print(df.to_string(index=False))
    elif args.pvalues:
        df = pd.DataFrame({"p_raw": args.pvalues})
        df["p_holm"] = holm(df["p_raw"].values)
        df["significant_holm"] = df["p_holm"] < args.alpha
        print(df.round(4).to_string(index=False))
    else:
        ap.error("give p-values or --csv")


if __name__ == "__main__":
    main()