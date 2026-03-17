#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_logistic_results.py

Create report-ready plots for deeper logistic regression results:
1. Marginal Effects bar chart
2. What-if Simulation bar chart

Input files expected in input_dir:
- marginal_effects.csv
- what_if_simulation.csv

Usage:
    python plot_logistic_results.py --input_dir regression_outputs --output_dir regression_outputs/figures
"""

import os
import argparse
import textwrap

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def wrap_text(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def shorten_feature_name(name: str) -> str:
    """Make feature names more readable in plots."""
    replacements = {
        "InternetService_Fiber optic": "Fiber optic internet",
        "PaperlessBilling_Yes": "Paperless billing",
        "PaymentMethod_Electronic check": "Electronic check",
        "PaymentMethod_Credit card (automatic)": "Credit card (auto)",
        "PaymentMethod_Mailed check": "Mailed check",
        "OnlineBackup_Yes": "Online backup",
        "OnlineSecurity_Yes": "Online security",
        "TechSupport_Yes": "Tech support",
        "SeniorCitizen": "Senior citizen",
        "MonthlyCharges": "Monthly charges",
        "TotalCharges": "Total charges",
        "tenure": "Tenure",
    }
    return replacements.get(name, name.replace("_", " "))


def plot_marginal_effects(input_dir: str, output_dir: str, top_n: int = 10):
    path = os.path.join(input_dir, "marginal_effects.csv")
    df = pd.read_csv(path)

    # select top features by absolute effect size
    df = df.copy()
    df["abs_effect"] = df["marginal_effect"].abs()
    df = df.sort_values("abs_effect", ascending=False).head(top_n)

    # sort for horizontal bar plot
    df = df.sort_values("marginal_effect")
    df["feature_label"] = df["feature"].apply(shorten_feature_name)

 

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["feature_label"], df["marginal_effect"], color="#6aa84f")
    ax.set_xlabel("Marginal Effect on Churn Probability")
    ax.set_ylabel("Feature")
    ax.set_title("Top Marginal Effects on Customer Churn")

    # add value labels
    for i, v in enumerate(df["marginal_effect"]):
        if v >= 0:
            ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
        else:
            ax.text(v - 0.002, i, f"{v:.3f}", va="center", ha="right", fontsize=9)

   

    out_png = os.path.join(output_dir, "marginal_effects_plot.png")
    out_pdf = os.path.join(output_dir, "marginal_effects_plot.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def plot_what_if_simulation(input_dir: str, output_dir: str):
    path = os.path.join(input_dir, "what_if_simulation.csv")
    df = pd.read_csv(path)

    df = df.copy()

    # nicer labels
    rename_map = {
        "Retention_Bundle_Contract_Support_Payment": "Retention bundle",
        "Switch_to_Two_Year_Contract": "Switch to 2-year contract",
        "Switch_to_One_Year_Contract": "Switch to 1-year contract",
        "Add_Online_Security": "Add online security",
        "Add_Tech_Support": "Add tech support",
        "Add_Online_Backup": "Add online backup",
        "Switch_from_Electronic_Check_to_Credit_Card": "Switch payment method",
        "Baseline": "Baseline",
    }
    df["scenario_label"] = df["scenario"].map(rename_map).fillna(df["scenario"])

    # sort by churn probability, smallest at top
    df = df.sort_values("predicted_churn_probability", ascending=True)



    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["scenario_label"], df["predicted_churn_probability"],color="#6aa84f")
    ax.set_xlabel("Predicted Churn Probability")
    ax.set_ylabel("Scenario")
    ax.set_title("What-if Simulation: Retention Strategy Impact")

    # add value labels
    for i, v in enumerate(df["predicted_churn_probability"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)



    out_png = os.path.join(output_dir, "what_if_simulation_plot.png")
    out_pdf = os.path.join(output_dir, "what_if_simulation_plot.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main(args):
    ensure_dir(args.output_dir)

    print("=" * 70)
    print("Creating plots")
    print("=" * 70)

    plot_marginal_effects(args.input_dir, args.output_dir, top_n=args.top_n)
    plot_what_if_simulation(args.input_dir, args.output_dir)

    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="regression_outputs",
        help="Directory containing marginal_effects.csv and what_if_simulation.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="regression_outputs/figures",
        help="Directory to save generated plots"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of top marginal effect features to plot"
    )
    args = parser.parse_args()
    main(args)