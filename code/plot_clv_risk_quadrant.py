#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLV × Churn Risk Quadrant Plot

Quadrant definition:
- X axis: churn risk (predicted probability)
- Y axis: proxy CLV = MonthlyCharges × tenure

Two modes:
A) If a probability column exists (e.g., churn_prob), use it.
B) Otherwise train a quick baseline logistic regression (sklearn) to generate probabilities.

Outputs:
- clv_risk_quadrant.png
- quadrant_summary.csv (counts per quadrant)
- high_value_high_risk_topN.csv (top customers by prob*CLV)

Usage:
  python plot_clv_risk_quadrant.py --csv Telco-Customer-Churn.csv --outdir outputs

Optional:
  --prob_col churn_prob           # if you already have it
  --risk_thr 0.5                  # default 0.5
  --clv_thr median                # median | p75 | number (e.g., 2000)
  --topn 200
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- helpers ----------
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    raise KeyError(f"Missing column. Tried {candidates}. Available: {list(df.columns)}")


def to_churn_bin(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    mp = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
    out = x.map(mp)
    if out.isna().any():
        bad = s[out.isna()].unique()[:10]
        raise ValueError(f"Unrecognized Churn labels (sample): {bad}")
    return out.astype(int)


def auto_prob_col(df: pd.DataFrame) -> str | None:
    cands = [
        "churn_prob", "churn_probability", "pred_prob", "pred_probability",
        "y_prob", "p_churn", "prob_churn", "probability"
    ]
    lower = {c.lower(): c for c in df.columns}
    for cand in cands:
        if cand in lower:
            return lower[cand]
    return None


def make_proxy_clv(df: pd.DataFrame, monthly_col: str, tenure_col: str) -> pd.Series:
    m = pd.to_numeric(df[monthly_col], errors="coerce")
    t = pd.to_numeric(df[tenure_col], errors="coerce")
    return m * t


def resolve_clv_threshold(clv: pd.Series, clv_thr_arg: str) -> float:
    clv_valid = pd.to_numeric(clv, errors="coerce").dropna()
    if clv_thr_arg.lower() == "median":
        return float(clv_valid.median())
    if clv_thr_arg.lower() in ["p75", "q75", "75%"]:
        return float(clv_valid.quantile(0.75))
    # numeric
    return float(clv_thr_arg)


# ---------- model fallback (only if no prob_col) ----------
def train_baseline_prob(df: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """
    Quick baseline using sklearn:
    - drop customerID-like col
    - numeric passthrough + categorical one-hot
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression

    X = df.copy()

    # drop obvious ID col if present
    for cand in ["customerID", "customerId", "customer_id", "CustomerID"]:
        if cand in X.columns:
            X = X.drop(columns=[cand])
            break

    # Separate feature types
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])

    # robust baseline
    clf = LogisticRegression(max_iter=2000, n_jobs=None)

    model = Pipeline([
        ("pre", pre),
        ("clf", clf)
    ])

    # train-test split just for training stability; we still output prob for all rows
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X)[:, 1]
    return prob


# ---------- plotting ----------
def plot_quadrant(prob: pd.Series, clv: pd.Series, risk_thr: float, clv_thr: float, outpath: Path) -> None:
    d = pd.DataFrame({"prob": prob, "clv": clv}).copy()
    d["prob"] = pd.to_numeric(d["prob"], errors="coerce")
    d["clv"] = pd.to_numeric(d["clv"], errors="coerce")
    d = d.dropna()

    plt.figure(figsize=(8, 6))
    plt.scatter(d["prob"], d["clv"], s=10)

    plt.axvline(risk_thr)
    plt.axhline(clv_thr)

    plt.xlabel("Churn Risk (Predicted Probability)")
    plt.ylabel("Proxy CLV (MonthlyCharges × tenure)")
    plt.title("CLV × Churn Risk Quadrant")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def quadrant_labels(prob: pd.Series, clv: pd.Series, risk_thr: float, clv_thr: float) -> pd.Series:
    """
    Quadrants:
      Q1: Low Risk, Low CLV
      Q2: Low Risk, High CLV
      Q3: High Risk, Low CLV
      Q4: High Risk, High CLV  (target segment)
    """
    p = pd.to_numeric(prob, errors="coerce")
    v = pd.to_numeric(clv, errors="coerce")
    q = pd.Series(index=prob.index, dtype="object")

    q[(p < risk_thr) & (v < clv_thr)] = "Q1 LowRisk-LowCLV"
    q[(p < risk_thr) & (v >= clv_thr)] = "Q2 LowRisk-HighCLV"
    q[(p >= risk_thr) & (v < clv_thr)] = "Q3 HighRisk-LowCLV"
    q[(p >= risk_thr) & (v >= clv_thr)] = "Q4 HighRisk-HighCLV"
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="./outputs")
    ap.add_argument("--prob_col", default=None, help="Name of churn probability column (optional).")
    ap.add_argument("--risk_thr", type=float, default=0.5)
    ap.add_argument("--clv_thr", default="median", help="median | p75 | numeric value (e.g., 2000)")
    ap.add_argument("--topn", type=int, default=200)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = standardize(df)

    churn_col = find_col(df, ["Churn"])
    monthly_col = find_col(df, ["MonthlyCharges", "Monthly Charges", "Monthly_Charges"])
    tenure_col = find_col(df, ["tenure"])

    y = to_churn_bin(df[churn_col])
    clv = make_proxy_clv(df, monthly_col, tenure_col)

    # get/compute probability
    prob_col = args.prob_col or auto_prob_col(df)
    if prob_col is not None and prob_col in df.columns:
        prob = pd.to_numeric(df[prob_col], errors="coerce")
        used_mode = f"Used existing probability column: {prob_col}"
    else:
        # baseline training
        prob = pd.Series(train_baseline_prob(df.drop(columns=[churn_col]), y), index=df.index, name="churn_prob")
        prob_col = "churn_prob"
        used_mode = "No probability column found → trained baseline logistic regression to generate churn_prob"

    # thresholds
    clv_thr = resolve_clv_threshold(clv, str(args.clv_thr))
    risk_thr = float(args.risk_thr)

    # plot
    plot_path = outdir / "clv_risk_quadrant.png"
    plot_quadrant(prob, clv, risk_thr, clv_thr, plot_path)

    # quadrant summary
    q = quadrant_labels(prob, clv, risk_thr, clv_thr)
    summary = q.value_counts(dropna=False).rename_axis("quadrant").reset_index(name="customers")
    summary.to_csv(outdir / "quadrant_summary.csv", index=False)

    # export top high value high risk (score = prob*CLV)
    tmp = df.copy()
    tmp["proxy_clv"] = clv
    tmp[prob_col] = prob
    tmp["risk_value_score"] = tmp[prob_col] * tmp["proxy_clv"]
    tmp["quadrant"] = q

    # keep ID if present
    id_col = None
    for cand in ["customerID", "customerId", "customer_id", "CustomerID"]:
        if cand in tmp.columns:
            id_col = cand
            break

    keep = []
    if id_col:
        keep.append(id_col)
    keep += [churn_col, monthly_col, tenure_col, "proxy_clv", prob_col, "risk_value_score", "quadrant"]

    top = tmp[tmp["quadrant"] == "Q4 HighRisk-HighCLV"].dropna(subset=["risk_value_score"]).sort_values(
        "risk_value_score", ascending=False
    ).head(args.topn)

    top[keep].to_csv(outdir / "high_value_high_risk_topN.csv", index=False)

    print("=== Quadrant Plot Done ===")
    print(used_mode)
    print(f"Risk threshold (x): {risk_thr}")
    print(f"CLV threshold (y): {clv_thr:.2f}  (from --clv_thr {args.clv_thr})")
    print(f"Saved plot: {plot_path.resolve()}")
    print(f"Saved summary: {(outdir / 'quadrant_summary.csv').resolve()}")
    print(f"Saved top list: {(outdir / 'high_value_high_risk_topN.csv').resolve()}")


if __name__ == "__main__":
    main()