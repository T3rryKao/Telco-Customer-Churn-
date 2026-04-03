#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Churn Behavior Analysis (Telco Customer Churn)
Objective: Identify high-risk customer segments

Outputs:
- Overall churn rate (printed)
- Segment-level churn rate tables (printed + CSV)
- Bar charts for churn rate by segment (saved as PNG)

Usage:
  python eda_churn_behavior.py --csv ./Telco-Customer-Churn.csv --outdir ./outputs

Notes:
- Assumes columns similar to Kaggle Telco dataset:
  'Churn', 'Contract', 'tenure', 'PaymentMethod'
- Robust to minor column name variations and whitespace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace in column names
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace in object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Find a column in df matching any candidate (case-insensitive).
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"Could not find any of these columns: {candidates}. Available: {list(df.columns)}")


def _to_churn_binary(series: pd.Series) -> pd.Series:
    """
    Convert churn labels to binary: Yes/True/1 => 1, No/False/0 => 0
    """
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "0": 0
    }
    out = s.map(mapping)
    if out.isna().any():
        bad = series[out.isna()].unique()[:10]
        raise ValueError(f"Unrecognized churn labels (sample): {bad}. Please clean/standardize Churn column.")
    return out.astype(int)


def churn_rate_table(df: pd.DataFrame, segment_col: str, churn_col_bin: str = "churn_bin") -> pd.DataFrame:
    """
    Return a table with:
      - customers
      - churners
      - churn_rate
    by segment_col
    """
    g = df.groupby(segment_col, dropna=False)[churn_col_bin]
    out = pd.DataFrame({
        "customers": g.size(),
        "churners": g.sum(),
        "churn_rate": g.mean()
    }).reset_index()

    out = out.sort_values("churn_rate", ascending=False).reset_index(drop=True)
    return out


def plot_churn_rate_bar(table: pd.DataFrame, segment_col: str, outpath: Path, top_n: int | None = None) -> None:
    """
    Bar chart for churn_rate by segment.
    If many categories, keep top_n by churn_rate.
    """
    t = table.copy()
    if top_n is not None and len(t) > top_n:
        t = t.head(top_n)

    # Convert segment values to string for plotting
    t[segment_col] = t[segment_col].astype(str)

    plt.figure(figsize=(10, 5))
    plt.bar(t[segment_col], t["churn_rate"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Churn rate")
    plt.title(f"Churn Rate by {segment_col}" + (f" (Top {top_n})" if top_n else ""))
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def add_tenure_bins(df: pd.DataFrame, tenure_col: str) -> pd.DataFrame:
    """
    Add tenure bins for segment-level comparison.
    Default bins match common telco patterns.
    """
    df = df.copy()
    # Ensure numeric tenure
    df[tenure_col] = pd.to_numeric(df[tenure_col], errors="coerce")

    # Drop missing tenure for binning (keep original rows but tenure_bin will be NaN)
    bins = [-np.inf, 3, 6, 12, 24, 36, 48, 60, np.inf]
    labels = ["0-3", "4-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61+"]

    df["tenure_bin"] = pd.cut(df[tenure_col], bins=bins, labels=labels)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Telco churn CSV file")
    parser.add_argument("--outdir", default="./outputs", help="Output directory for tables and charts")
    parser.add_argument("--topn", type=int, default=20, help="Top-N categories to plot (for high-cardinality segments)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.csv)
    df = _standardize_columns(df)

    # Detect key columns
    churn_col = _find_col(df, ["Churn"])
    contract_col = _find_col(df, ["Contract"])
    tenure_col = _find_col(df, ["tenure"])
    pay_col = _find_col(df, ["PaymentMethod", "Payment Method", "Payment_Method"])

    # Binary churn
    df["churn_bin"] = _to_churn_binary(df[churn_col])

    # ========== Overall churn rate ==========
    overall_rate = df["churn_bin"].mean()
    n = len(df)
    churners = int(df["churn_bin"].sum())
    print("\n=== Overall Churn Rate ===")
    print(f"Customers: {n:,}")
    print(f"Churners : {churners:,}")
    print(f"Churn rate: {overall_rate:.3f} ({overall_rate*100:.1f}%)")

    # Save a small summary txt
    (outdir / "overall_churn_summary.txt").write_text(
        f"Customers: {n}\nChurners: {churners}\nChurn rate: {overall_rate:.6f}\n",
        encoding="utf-8"
    )

    # ========== Segment-level comparisons ==========
    print("\n=== Segment-level Churn Rate Tables ===")

    # 1) Contract
    contract_tbl = churn_rate_table(df, contract_col)
    print("\n[Contract]")
    print(contract_tbl.to_string(index=False))
    contract_tbl.to_csv(outdir / "churn_rate_by_contract.csv", index=False)

    plot_churn_rate_bar(
        contract_tbl, contract_col,
        outdir / "churn_rate_by_contract.png",
        top_n=None
    )

    # 2) Tenure bins
    df2 = add_tenure_bins(df, tenure_col)
    tenure_tbl = churn_rate_table(df2.dropna(subset=["tenure_bin"]), "tenure_bin")
    print("\n[Tenure Bin]")
    print(tenure_tbl.to_string(index=False))
    tenure_tbl.to_csv(outdir / "churn_rate_by_tenure_bin.csv", index=False)

    plot_churn_rate_bar(
        tenure_tbl, "tenure_bin",
        outdir / "churn_rate_by_tenure_bin.png",
        top_n=None
    )

    # 3) Payment Method (can be more categories)
    pay_tbl = churn_rate_table(df, pay_col)
    print("\n[Payment Method]")
    print(pay_tbl.to_string(index=False))
    pay_tbl.to_csv(outdir / "churn_rate_by_payment_method.csv", index=False)

    plot_churn_rate_bar(
        pay_tbl, pay_col,
        outdir / "churn_rate_by_payment_method.png",
        top_n=min(args.topn, len(pay_tbl)) if len(pay_tbl) > 10 else None
    )

    # ========== Quick: identify top high-risk segments ==========
    # "High-risk" here means highest churn_rate; you can later combine with CLV to prioritize.
    def top3(tbl: pd.DataFrame, seg: str) -> pd.DataFrame:
        return tbl[[seg, "customers", "churn_rate"]].head(3)

    print("\n=== Top High-Risk Segments (by churn rate) ===")
    print("\nTop Contract types:")
    print(top3(contract_tbl, contract_col).to_string(index=False))
    print("\nTop Tenure bins:")
    print(top3(tenure_tbl, "tenure_bin").to_string(index=False))
    print("\nTop Payment methods:")
    print(top3(pay_tbl, pay_col).to_string(index=False))

    print(f"\nSaved outputs to: {outdir.resolve()}\n")


if __name__ == "__main__":
    main()