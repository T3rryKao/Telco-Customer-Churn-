#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
retention_roi_curve.py

Build a Retention ROI Curve (Profit vs Target Rate) for churn retention targeting.

Assumptions:
- You have a per-customer predicted churn probability column (default: churn_probability).
- You either have per-customer CLV column OR a MonthlyCharges column to estimate CLV.
- Expected saved revenue per customer = p_churn * CLV
- Expected profit per customer = expected_saved_revenue - retention_cost

Outputs:
- A CSV with ROI results (target_rate, n_targeted, total_profit, total_cost, total_saved_revenue, avg_profit_per_targeted)
- A PNG plot of profit vs target percentage
- Prints best target rate (max total expected profit)

Example:
  python retention_roi_curve.py --input customer_risk_scores.csv \
    --prob_col churn_probability --monthly_col MonthlyCharges \
    --clv_months 24 --retention_cost 50 --max_target 0.50 --step 0.01
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retention ROI Curve (Profit vs Target Rate)")
    p.add_argument("--input", required=True, help="Path to input CSV (must contain churn_probability).")
    p.add_argument("--output_dir", default="outputs", help="Directory to save outputs (csv + png).")

    p.add_argument("--prob_col", default="churn_probability", help="Column name for predicted churn probability.")
    p.add_argument("--clv_col", default="CLV", help="Optional column name for CLV. If provided, MonthlyCharges is optional.")
    p.add_argument("--monthly_col", default="MonthlyCharges", help="Monthly charges column used to estimate CLV (if clv_col not provided).")
    p.add_argument("--clv_months", type=float, default=24.0, help="Months to estimate CLV = MonthlyCharges * clv_months (if clv_col not provided).")

    p.add_argument("--retention_cost", type=float, default=50.0, help="Retention intervention cost per targeted customer.")
    p.add_argument("--max_target", type=float, default=1.0, help="Maximum targeting rate (0-1). Example 0.50 = 50%.")
    p.add_argument("--step", type=float, default=0.01, help="Step size for targeting rate. Example 0.01 = 1% increments.")

    p.add_argument("--plot_title", default="Retention ROI Curve (Profit vs Target Rate)", help="Title for the plot.")
    p.add_argument("--save_prefix", default="retention_roi", help="Filename prefix for outputs.")

    return p.parse_args()


def validate_inputs(df: pd.DataFrame, prob_col: str, clv_col: Optional[str], monthly_col: str) -> None:
    if prob_col not in df.columns:
        raise ValueError(f"Missing required probability column: '{prob_col}'")

    if clv_col is not None:
        if clv_col not in df.columns:
            raise ValueError(f"clv_col was provided but column not found: '{clv_col}'")
    else:
        if monthly_col not in df.columns:
            raise ValueError(
                f"CLV column not provided, so MonthlyCharges column is required but missing: '{monthly_col}'"
            )


def compute_clv(df: pd.DataFrame, clv_col: Optional[str], monthly_col: str, clv_months: float) -> pd.Series:
    if clv_col is not None:
        clv = pd.to_numeric(df[clv_col], errors="coerce")
    else:
        monthly = pd.to_numeric(df[monthly_col], errors="coerce")
        clv = monthly * float(clv_months)

    if clv.isna().any():
        n_bad = int(clv.isna().sum())
        raise ValueError(f"CLV has {n_bad} missing/invalid values after conversion. Please clean your data.")
    return clv


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    validate_inputs(df, args.prob_col, args.clv_col, args.monthly_col)

    # Convert probability to numeric, ensure valid range
    p_churn = pd.to_numeric(df[args.prob_col], errors="coerce")
    if p_churn.isna().any():
        n_bad = int(p_churn.isna().sum())
        raise ValueError(f"Probability column '{args.prob_col}' has {n_bad} missing/invalid values.")

    if (p_churn < 0).any() or (p_churn > 1).any():
        raise ValueError(f"Probability column '{args.prob_col}' must be within [0, 1].")

    df = df.copy()
    df["_p_churn"] = p_churn

    # Build/Load CLV
    df["_CLV"] = compute_clv(df, args.clv_col, args.monthly_col, args.clv_months)

    # Sort by predicted churn descending (highest risk first)
    df = df.sort_values("_p_churn", ascending=False).reset_index(drop=True)

    # Per-customer expected values
    retention_success_rate = 0.25

    df["_expected_saved_revenue"] = (
    df["_p_churn"] * retention_success_rate * df["_CLV"])
    df["_expected_profit"] = df["_expected_saved_revenue"] - float(args.retention_cost)

    # Simulate targeting rates
    max_target = float(args.max_target)
    step = float(args.step)
    if not (0 < max_target <= 1):
        raise ValueError("--max_target must be in (0, 1].")
    if not (0 < step <= 1):
        raise ValueError("--step must be in (0, 1].")

    rates = np.arange(step, max_target + 1e-12, step)
    results = []

    N = len(df)
    for rate in rates:
        n_target = int(np.floor(N * rate))
        if n_target <= 0:
            continue

        targeted = df.iloc[:n_target]

        total_saved = float(targeted["_expected_saved_revenue"].sum())
        total_cost = float(args.retention_cost) * n_target
        total_profit = float(targeted["_expected_profit"].sum())
        avg_profit = total_profit / n_target

        results.append({
            "target_rate": rate,
            "target_pct": rate * 100.0,
            "n_targeted": n_target,
            "total_expected_saved_revenue": total_saved,
            "total_retention_cost": total_cost,
            "total_expected_profit": total_profit,
            "avg_profit_per_targeted": avg_profit,
        })

    if not results:
        raise RuntimeError("No results were computed. Check max_target/step and dataset size.")

    res = pd.DataFrame(results)

    # Find best target rate by total_expected_profit
    best_idx = int(res["total_expected_profit"].idxmax())
    best_row = res.loc[best_idx]

    # Save results CSV
    out_csv = os.path.join(args.output_dir, f"{args.save_prefix}_roi_results.csv")
    res.to_csv(out_csv, index=False)

    # Plot: Profit vs Target %
    plt.figure(figsize=(9, 5))
    plt.plot(res["target_pct"], res["total_expected_profit"])
    plt.xlabel("Target Customers (%)")
    plt.ylabel("Total Expected Profit ($)")
    plt.title(args.plot_title)

    # Mark best point
    plt.scatter([best_row["target_pct"]], [best_row["total_expected_profit"]])
    plt.annotate(
        f"Best: {best_row['target_pct']:.0f}%\nProfit: ${best_row['total_expected_profit']:,.0f}",
        xy=(best_row["target_pct"], best_row["total_expected_profit"]),
        xytext=(best_row["target_pct"] + 2, best_row["total_expected_profit"]),
        arrowprops=dict(arrowstyle="->", lw=1),
    )

    out_png = os.path.join(args.output_dir, f"{args.save_prefix}_roi_curve.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)

    print("=== Retention ROI Curve Completed ===")
    print(f"Input: {args.input}")
    print(f"Saved results CSV: {out_csv}")
    print(f"Saved plot PNG:    {out_png}")
    print()
    print("=== Best Targeting Strategy (by total expected profit) ===")
    print(f"Best target rate:  {best_row['target_pct']:.0f}% (n={int(best_row['n_targeted'])})")
    print(f"Total profit:      ${best_row['total_expected_profit']:,.2f}")
    print(f"Total saved rev:   ${best_row['total_expected_saved_revenue']:,.2f}")
    print(f"Total cost:        ${best_row['total_retention_cost']:,.2f}")
    print(f"Avg profit/target: ${best_row['avg_profit_per_targeted']:,.2f}")


if __name__ == "__main__":
    main()