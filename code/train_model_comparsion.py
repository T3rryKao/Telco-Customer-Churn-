#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model_comparison_churn.py

Compare Logistic Regression and Random Forest on the
Telco Customer Churn dataset.

Outputs:
- model metrics (ROC-AUC, Average Precision, classification report)
- ROC curve comparison plot
- Random Forest feature importance plot
- customer-level prediction csv
- high-risk customer csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. Config
# =========================================================
DATA_PATH = "Telco-Customer-Churn.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_RISK_PCT = 0.2   # top 20% highest-risk customers

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. Load and clean data
# =========================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert target to binary
    df["Churn_Flag"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


# =========================================================
# 3. Preprocessing
# =========================================================
def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features


# =========================================================
# 4. Build models
# =========================================================
def build_models(preprocessor):
    logit_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])

    rf_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    return logit_model, rf_model


# =========================================================
# 5. Evaluate model
# =========================================================
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    clf_report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 70)
    print(f"{name}")
    print("=" * 70)
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"Average Precision:{avg_precision:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(clf_report)

    return {
        "name": name,
        "model": model,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "y_proba": y_proba,
        "y_pred": y_pred
    }


# =========================================================
# 6. Plot ROC comparison
# =========================================================
def plot_roc_curves(y_test, results_list, output_path):
    plt.figure(figsize=(8, 6))

    for result in results_list:
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        plt.plot(
            fpr, tpr,
            label=f'{result["name"]} (AUC = {result["roc_auc"]:.3f})'
        )

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# =========================================================
# 7. Random Forest feature importance
# =========================================================
def get_feature_names_from_pipeline(preprocessor, numeric_features, categorical_features):
    """
    Extract transformed feature names after preprocessing.
    """
    cat_ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_ohe.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names
    return all_feature_names


def plot_rf_feature_importance(rf_pipeline, numeric_features, categorical_features, output_path, top_n=20):
    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_model = rf_pipeline.named_steps["model"]

    feature_names = get_feature_names_from_pipeline(
        preprocessor,
        numeric_features,
        categorical_features
    )

    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Random Forest Feature Importance (Top {top_n})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

    csv_path = os.path.join(os.path.dirname(output_path), "random_forest_feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    print("\nTop 15 Random Forest Features:")
    print(importance_df.head(15).to_string(index=False))

    return importance_df


# =========================================================
# 8. Logistic regression coefficients
# =========================================================
def export_logistic_coefficients(logit_pipeline, numeric_features, categorical_features, output_dir):
    preprocessor = logit_pipeline.named_steps["preprocessor"]
    logit_model = logit_pipeline.named_steps["model"]

    feature_names = get_feature_names_from_pipeline(
        preprocessor,
        numeric_features,
        categorical_features
    )

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": logit_model.coef_[0]
    })

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    path = os.path.join(output_dir, "logistic_regression_coefficients.csv")
    coef_df.to_csv(path, index=False)
    print(f"Saved: {path}")

    print("\nTop 15 Logistic Regression Coefficients:")
    print(coef_df.head(15).to_string(index=False))

    return coef_df


# =========================================================
# 9. Export customer prediction files
# =========================================================
def export_customer_predictions(
    df_full,
    X_full,
    y_full,
    trained_pipeline,
    output_dir,
    model_name="random_forest",
    top_risk_pct=0.2
):
    all_proba = trained_pipeline.predict_proba(X_full)[:, 1]

    out_df = df_full.copy()
    out_df["predicted_churn_probability"] = all_proba
    out_df["risk_rank"] = out_df["predicted_churn_probability"].rank(ascending=False, method="first")

    threshold = out_df["predicted_churn_probability"].quantile(1 - top_risk_pct)
    out_df["risk_segment"] = np.where(
        out_df["predicted_churn_probability"] >= threshold,
        "High Risk",
        "Lower Risk"
    )

    pred_path = os.path.join(output_dir, f"{model_name}_customer_predictions.csv")
    out_df.to_csv(pred_path, index=False)
    print(f"Saved: {pred_path}")

    high_risk_df = out_df[out_df["risk_segment"] == "High Risk"].copy()
    high_risk_df = high_risk_df.sort_values("predicted_churn_probability", ascending=False)

    high_risk_path = os.path.join(output_dir, f"{model_name}_high_risk_customers.csv")
    high_risk_df.to_csv(high_risk_path, index=False)
    print(f"Saved: {high_risk_path} (n={len(high_risk_df)})")

    return out_df, high_risk_df


# =========================================================
# 10. Save summary metrics
# =========================================================
def save_metrics_summary(results_list, output_dir):
    metrics_df = pd.DataFrame([
        {
            "model": r["name"],
            "roc_auc": r["roc_auc"],
            "average_precision": r["avg_precision"]
        }
        for r in results_list
    ])

    path = os.path.join(output_dir, "model_comparison_metrics.csv")
    metrics_df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print("\nModel Comparison Summary:")
    print(metrics_df.to_string(index=False))

    return metrics_df


# =========================================================
# 11. Main
# =========================================================
def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    # Drop rows with missing target or key data if needed
    df = df.dropna(subset=["Churn_Flag"]).copy()

    # Features / target
    drop_cols = ["customerID", "Churn", "Churn_Flag"]
    X = df.drop(columns=drop_cols)
    y = df["Churn_Flag"]

    # Build preprocessor
    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Churn rate (train): {y_train.mean():.4f}")
    print(f"Churn rate (test):  {y_test.mean():.4f}")

    # Build models
    logit_model, rf_model = build_models(preprocessor)

    # Evaluate
    logit_result = evaluate_model(
        "Logistic Regression",
        logit_model,
        X_train, X_test, y_train, y_test
    )

    rf_result = evaluate_model(
        "Random Forest",
        rf_model,
        X_train, X_test, y_train, y_test
    )

    results_list = [logit_result, rf_result]

    # Save metrics summary
    save_metrics_summary(results_list, OUTPUT_DIR)

    # ROC curve comparison
    plot_roc_curves(
        y_test=y_test,
        results_list=results_list,
        output_path=os.path.join(OUTPUT_DIR, "roc_curve_model_comparison.png")
    )

    # Export coefficients / importance
    export_logistic_coefficients(
        logit_pipeline=logit_result["model"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        output_dir=OUTPUT_DIR
    )

    plot_rf_feature_importance(
        rf_pipeline=rf_result["model"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        output_path=os.path.join(OUTPUT_DIR, "random_forest_feature_importance.png"),
        top_n=20
    )

    # Refit best / chosen model on full dataset for full-customer scoring
    # Here we use Random Forest for richer ranking, but you can switch to logistic regression.
    final_model = rf_result["model"]
    final_model.fit(X, y)

    export_customer_predictions(
        df_full=df,
        X_full=X,
        y_full=y,
        trained_pipeline=final_model,
        output_dir=OUTPUT_DIR,
        model_name="random_forest",
        top_risk_pct=TOP_RISK_PCT
    )

    print("\nDone.")
    print("Generated files in outputs/:")
    print("- model_comparison_metrics.csv")
    print("- roc_curve_model_comparison.png")
    print("- logistic_regression_coefficients.csv")
    print("- random_forest_feature_importance.csv")
    print("- random_forest_feature_importance.png")
    print("- random_forest_customer_predictions.csv")
    print("- random_forest_high_risk_customers.csv")


if __name__ == "__main__":
    main()