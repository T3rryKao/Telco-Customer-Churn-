import pandas as pd
import numpy as np

RETENTION_COST = 60

df = pd.read_csv("outputs/customer_risk_scores_with_charge.csv")

PROB_COL = "churn_probability"

df = df.sort_values(PROB_COL, ascending=False).reset_index(drop=True)

N = len(df)

results = []

save_probs = [0.1, 0.2, 0.3]

target_rates = np.arange(0.01, 1.01, 0.01)

for save_prob in save_probs:
    
    for rate in target_rates:

        n_target = int(rate * N)

        target_df = df.iloc[:n_target]

        # dynamic CLV using MonthlyCharges
        saved_clv = (
            target_df["churn_probability"] *
            save_prob *
            target_df["MonthlyCharges"] * 12
        ).sum()

        total_cost = n_target * RETENTION_COST

        profit = saved_clv - total_cost

        roi = profit / total_cost if total_cost > 0 else 0

        results.append({
            "scenario": f"{save_prob}",
            "save_prob": save_prob,
            "target_rate": round(rate, 5),
            "customers_targeted": n_target,
            "profit": profit,
            "roi": roi
        })

roi_df = pd.DataFrame(results)

roi_df.to_csv("retention_roi_tableau.csv", index=False)