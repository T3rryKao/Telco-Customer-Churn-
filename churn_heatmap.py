# churn_heatmap.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. Config
# =========================================================
DATA_PATH = "Telco-Customer-Churn.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. Load data
# =========================================================
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Convert Churn to binary
df["Churn_Flag"] = df["Churn"].map({"Yes": 1, "No": 0})

# =========================================================
# 3. Create bins for numeric heatmap
# =========================================================
df["tenure_bin"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"],
    include_lowest=True
)

df["MonthlyCharges_bin"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 70, 100, np.inf],
    labels=["0-35", "35-70", "70-100", "100+"],
    include_lowest=True
)

# =========================================================
# 4. Helper function to draw heatmap
# =========================================================
def plot_churn_heatmap(data, row_col, col_col, value_col="Churn_Flag",
                       title="", output_file="heatmap.png", cmap="YlOrRd"):
    """
    Plot churn rate heatmap without seaborn.
    """
    pivot = data.pivot_table(
        index=row_col,
        columns=col_col,
        values=value_col,
        aggfunc="mean"
    ) * 100  # convert to percentage

    # Keep order
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)

    # ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticklabels(pivot.index)

    # annotate values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=9)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Churn Rate (%)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")
    print("\nPivot table (%):")
    print(pivot.round(2))
    print("-" * 60)

# =========================================================
# 5. Heatmap 1: Contract × PaymentMethod
# =========================================================
plot_churn_heatmap(
    data=df,
    row_col="Contract",
    col_col="PaymentMethod",
    title="Churn Rate Heatmap: Contract vs Payment Method",
    output_file=os.path.join(OUTPUT_DIR, "heatmap_contract_payment.png")
)

# =========================================================
# 6. Heatmap 2: InternetService × Contract
# =========================================================
plot_churn_heatmap(
    data=df,
    row_col="InternetService",
    col_col="Contract",
    title="Churn Rate Heatmap: Internet Service vs Contract",
    output_file=os.path.join(OUTPUT_DIR, "heatmap_internet_contract.png")
)

# =========================================================
# 7. Heatmap 3: Tenure Bin × MonthlyCharges Bin
# =========================================================
plot_churn_heatmap(
    data=df,
    row_col="tenure_bin",
    col_col="MonthlyCharges_bin",
    title="Churn Rate Heatmap: Tenure Bin vs Monthly Charges Bin",
    output_file=os.path.join(OUTPUT_DIR, "heatmap_tenure_monthlycharges.png")
)

print("\nDone. All churn heatmaps are saved in the outputs folder.")