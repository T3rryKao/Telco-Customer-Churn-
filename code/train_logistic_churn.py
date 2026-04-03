# train_logistic_churn.py
# High-risk customer identification via Logistic Regression (Telco Customer Churn)
#
# Usage:
#   python train_logistic_churn.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --top_pct 0.2
#
# Outputs:
#   outputs/model_metrics.json
#   outputs/customer_risk_scores.csv
#   outputs/high_risk_customers.csv

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix


@dataclass
class Config:
    data_path: str
    output_dir: str = "outputs"
    test_size: float = 0.25
    random_state: int = 42
    top_pct: float = 0.2  # top 20% as high-risk
    C: float = 1.0  # inverse regularization strength


def load_and_clean(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    # Standard Telco columns expected:
    # customerID, Churn, TotalCharges may be strings with blanks
    if "customerID" not in df.columns:
        raise ValueError("Expected column 'customerID' not found.")
    if "Churn" not in df.columns:
        raise ValueError("Expected column 'Churn' not found.")

    # Convert TotalCharges to numeric if present (common gotcha in Telco dataset)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int64")

    return df


def build_pipeline(X: pd.DataFrame, C: float) -> Tuple[Pipeline, List[str], List[str]]:
    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Logistic Regression classifier (good baseline for churn + interpretable)
    clf = LogisticRegression(
        max_iter=2000,
        C=C,
        solver="lbfgs",
        n_jobs=None,
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
    return model, numeric_cols, categorical_cols


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, pred).tolist()
    report = classification_report(y_test, pred, output_dict=True)

    return {
        "roc_auc": float(auc),
        "avg_precision": float(ap),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def main(cfg: Config) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    df = load_and_clean(cfg.data_path)

    # Keep ID for outputs, but exclude from features
    customer_ids = df["customerID"].astype(str).copy()

    y = df["Churn"].to_numpy()
    X = df.drop(columns=["Churn", "customerID"])

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, customer_ids, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    model, numeric_cols, categorical_cols = build_pipeline(X_train, C=cfg.C)

    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)

    # Predict churn probabilities for ALL customers (for ranking / high-risk targeting)
    all_proba = model.predict_proba(X)[:, 1]

    scores = pd.DataFrame(
        {
            "customerID": customer_ids.values,
            "churn_probability": all_proba,
        }
    ).sort_values("churn_probability", ascending=False)

    # High-risk = top pct
    if not (0.0 < cfg.top_pct < 1.0):
        raise ValueError("--top_pct must be between 0 and 1 (e.g., 0.2 for top 20%).")

    n_high = int(np.ceil(len(scores) * cfg.top_pct))
    high_risk = scores.head(n_high).copy()
    high_risk["risk_bucket"] = f"top_{int(cfg.top_pct*100)}pct"

    # Save outputs
    with open(os.path.join(cfg.output_dir, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    scores.to_csv(os.path.join(cfg.output_dir, "customer_risk_scores.csv"), index=False)
    high_risk.to_csv(os.path.join(cfg.output_dir, "high_risk_customers.csv"), index=False)

    print("✅ Done.")
    print(f"- ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"- Avg Precision (PR-AUC proxy): {metrics['avg_precision']:.4f}")
    print(f"- Saved: {cfg.output_dir}/customer_risk_scores.csv")
    print(f"- Saved: {cfg.output_dir}/high_risk_customers.csv (n={n_high})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data_path", required=True, help="Path to Telco Customer Churn CSV")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--top_pct", type=float, default=0.2, help="Top fraction treated as high-risk (0-1)")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression")
    args = parser.parse_args()

    cfg = Config(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        top_pct=args.top_pct,
        C=args.C,
    )
    main(cfg)