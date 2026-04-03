import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# =========================================================
# 1. Load data
# =========================================================
df = pd.read_csv("Telco-Customer-Churn.csv")

# =========================================================
# 2. Basic cleaning
# =========================================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna().copy()

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# =========================================================
# 3. Build model matrix
# =========================================================
X_raw = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

X = pd.get_dummies(X_raw, drop_first=True)

# =========================================================
# 4. Train logistic regression
# =========================================================
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X, y)

# =========================================================
# 5. Predict churn probability
# =========================================================
df["churn_probability"] = model.predict_proba(X)[:, 1]

# =========================================================
# 6. Define high-risk customers
# =========================================================
threshold = df["churn_probability"].quantile(0.80)

df["risk_segment"] = np.where(
    df["churn_probability"] >= threshold,
    "High Risk",
    "Normal Risk"
)

# =========================================================
# 7. CLV and expected loss
# =========================================================
df["CLV"] = df["MonthlyCharges"] * 12
df["expected_loss"] = df["churn_probability"] * df["CLV"]

# =========================================================
# 8. Quadrant segments
# =========================================================
avg_clv = df["CLV"].mean()

conditions = [
    (df["churn_probability"] >= threshold) & (df["CLV"] >= avg_clv),
    (df["churn_probability"] >= threshold) & (df["CLV"] < avg_clv),
    (df["churn_probability"] < threshold) & (df["CLV"] >= avg_clv),
    (df["churn_probability"] < threshold) & (df["CLV"] < avg_clv),
]

choices = [
    "High Risk - High Value",
    "High Risk - Low Value",
    "Low Risk - High Value",
    "Low Risk - Low Value"
]

df["quadrant_segment"] = np.select(conditions, choices, default="Unknown")

# =========================================================
# 9. Multi-treatment retention strategy
#    Forced intervention version:
#    choose best of A/B/C even if profit is negative
# =========================================================
retention_options = {
    "A_Low_Cost": {"cost": 5, "success_rate": 0.10},
    "B_Medium_Cost": {"cost": 30, "success_rate": 0.25},
    "C_High_Cost": {"cost": 50, "success_rate": 0.30},
}

for option_name, params in retention_options.items():
    cost = params["cost"]
    success_rate = params["success_rate"]

    df[f"{option_name}_cost"] = cost
    df[f"{option_name}_success_rate"] = success_rate
    df[f"{option_name}_expected_saved_revenue"] = df["expected_loss"] * success_rate
    df[f"{option_name}_expected_profit"] = df[f"{option_name}_expected_saved_revenue"] - cost

# =========================================================
# 10. Best treatment if intervention is forced
#     IMPORTANT: no "No Intervention" here
# =========================================================
profit_cols = [f"{name}_expected_profit" for name in retention_options.keys()]

df["forced_best_profit"] = df[profit_cols].max(axis=1)
df["forced_best_option"] = df[profit_cols].idxmax(axis=1).str.replace(
    "_expected_profit", "", regex=False
)

def get_option_cost(option):
    return retention_options[option]["cost"]

def get_option_success_rate(option):
    return retention_options[option]["success_rate"]

df["forced_best_cost"] = df["forced_best_option"].apply(get_option_cost)
df["forced_best_success_rate"] = df["forced_best_option"].apply(get_option_success_rate)
df["forced_best_expected_saved_revenue"] = df["expected_loss"] * df["forced_best_success_rate"]

# =========================================================
# 11. Optional: keep original "No Intervention" version too
#     for comparison
# =========================================================
df["best_retention_profit"] = df["forced_best_profit"]
df["best_retention_option"] = df["forced_best_option"]

df["best_retention_option"] = np.where(
    df["best_retention_profit"] > 0,
    df["best_retention_option"],
    "No Intervention"
)

df["best_retention_profit"] = np.where(
    df["best_retention_profit"] > 0,
    df["best_retention_profit"],
    0
)

df["best_retention_cost"] = np.where(
    df["best_retention_option"] == "No Intervention",
    0,
    df["forced_best_cost"]
)

df["best_retention_success_rate"] = np.where(
    df["best_retention_option"] == "No Intervention",
    0,
    df["forced_best_success_rate"]
)

df["best_expected_saved_revenue"] = df["expected_loss"] * df["best_retention_success_rate"]

# =========================================================
# 12. Simulation for declining profit curve
#     + add A-only / B-only / C-only curves
# =========================================================
df_ranked = df.sort_values("expected_loss", ascending=False).reset_index(drop=True)

target_rates = np.arange(0.01, 1.01, 0.01)

simulation_results = []

for rate in target_rates:
    n_target = max(1, int(len(df_ranked) * rate))
    target_df = df_ranked.head(n_target).copy()

    total_clv = target_df["CLV"].sum()
    total_expected_loss = target_df["expected_loss"].sum()

    # -----------------------------------------------------
    # Best mixed strategy (forced intervention)
    # -----------------------------------------------------
    total_expected_saved_revenue = target_df["forced_best_expected_saved_revenue"].sum()
    total_retention_cost = target_df["forced_best_cost"].sum()
    cumulative_profit = target_df["forced_best_profit"].sum()

    roi = np.nan
    if total_retention_cost > 0:
        roi = total_expected_saved_revenue / total_retention_cost

    option_counts = target_df["forced_best_option"].value_counts().to_dict()

    # -----------------------------------------------------
    # Single-treatment curves
    # -----------------------------------------------------
    profit_A_only = target_df["A_Low_Cost_expected_profit"].sum()
    profit_B_only = target_df["B_Medium_Cost_expected_profit"].sum()
    profit_C_only = target_df["C_High_Cost_expected_profit"].sum()

    saved_rev_A_only = target_df["A_Low_Cost_expected_saved_revenue"].sum()
    saved_rev_B_only = target_df["B_Medium_Cost_expected_saved_revenue"].sum()
    saved_rev_C_only = target_df["C_High_Cost_expected_saved_revenue"].sum()

    cost_A_only = len(target_df) * retention_options["A_Low_Cost"]["cost"]
    cost_B_only = len(target_df) * retention_options["B_Medium_Cost"]["cost"]
    cost_C_only = len(target_df) * retention_options["C_High_Cost"]["cost"]

    roi_A_only = saved_rev_A_only / cost_A_only if cost_A_only > 0 else np.nan
    roi_B_only = saved_rev_B_only / cost_B_only if cost_B_only > 0 else np.nan
    roi_C_only = saved_rev_C_only / cost_C_only if cost_C_only > 0 else np.nan

    simulation_results.append({
        "target_rate_pct": rate * 100,
        "customers_targeted": n_target,
        "total_clv": total_clv,
        "total_expected_loss": total_expected_loss,

        # mixed strategy
        "expected_saved_revenue": total_expected_saved_revenue,
        "total_retention_cost": total_retention_cost,
        "incremental_profit": cumulative_profit,
        "retention_roi": roi,

        # strategy counts
        "A_Low_Cost_count": option_counts.get("A_Low_Cost", 0),
        "B_Medium_Cost_count": option_counts.get("B_Medium_Cost", 0),
        "C_High_Cost_count": option_counts.get("C_High_Cost", 0),

        # A-only curve
        "incremental_profit_A_only": profit_A_only,
        "retention_roi_A_only": roi_A_only,

        # B-only curve
        "incremental_profit_B_only": profit_B_only,
        "retention_roi_B_only": roi_B_only,

        # C-only curve
        "incremental_profit_C_only": profit_C_only,
        "retention_roi_C_only": roi_C_only,
    })

target_rate_simulation_declining = pd.DataFrame(simulation_results)

# marginal profit of newly added 1% segment
target_rate_simulation_declining["marginal_profit"] = (
    target_rate_simulation_declining["incremental_profit"].diff()
)
target_rate_simulation_declining.loc[0, "marginal_profit"] = (
    target_rate_simulation_declining.loc[0, "incremental_profit"]
)

target_rate_simulation_declining.to_csv(
    "target_rate_profit_declining_curve_with_3_options.csv",
    index=False
)

print("Saved: target_rate_profit_declining_curve_with_3_options.csv")

# =========================================================
# 13. Long-format profit curve for Tableau
# =========================================================
profit_curve_long = target_rate_simulation_declining.melt(
    id_vars=["target_rate_pct", "customers_targeted"],
    value_vars=[
        "incremental_profit",
        "incremental_profit_A_only",
        "incremental_profit_B_only",
        "incremental_profit_C_only"
    ],
    var_name="strategy_curve",
    value_name="incremental_profit_value"
)

profit_curve_long["strategy_curve"] = profit_curve_long["strategy_curve"].replace({
    "incremental_profit": "Best Mixed Strategy",
    "incremental_profit_A_only": "A Only",
    "incremental_profit_B_only": "B Only",
    "incremental_profit_C_only": "C Only"
})

profit_curve_long.to_csv("profit_curve_4_lines_long.csv", index=False)
print("Saved: profit_curve_4_lines_long.csv")
# =========================================================
# 14. Preview
# =========================================================
print("\n=== Forced Intervention Simulation Preview ===")
print(target_rate_simulation_declining.head(15))

print("\n=== Best Target Rate (by cumulative profit) ===")

print("\n=== Last 10 rows ===")
print(target_rate_simulation_declining.tail(10))