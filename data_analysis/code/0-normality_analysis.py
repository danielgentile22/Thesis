# 0-normality_analysis.py

import os
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro, skew, kurtosis
import numpy as np

DB_PATH = "../../results_processed/experiment_results.db"
OUTPUT_DIR = "../results/0-normality"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM data", conn)
    conn.close()
    return df

def map_feedback_responses(df):
    feedback_mapping = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
        "Can't answer": np.nan
    }
    feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]
    for col in feedback_cols:
        if col in df.columns:
            df[col] = df[col].map(feedback_mapping)
    return df

def aggregate_accuracy(df):
    grouped = df.groupby(["subject_number", "model_name"]).apply(
        lambda x: np.mean(x["intended_digit"] == x["predicted_digit"]) * 100
    ).reset_index(name="mean_accuracy")
    return grouped

def aggregate_confidence(df):
    grouped = df.groupby(["subject_number", "model_name"])["confidence"].mean().reset_index(name="mean_confidence")
    return grouped

def normality_analysis(df, group_col=None, numeric_cols=None, out_dir="analysis_results", save_plots=True, results_filename="normality_test_results.csv"):
    if numeric_cols is None:
        numeric_cols = []

    plot_dir = os.path.join(out_dir, "normality_plots")
    os.makedirs(plot_dir, exist_ok=True)

    results_records = []

    groups = df[group_col].dropna().unique() if group_col else ["All_Data"]

    for col in numeric_cols:
        for g in groups:
            subset = df[df[group_col] == g][col].dropna() if group_col else df[col].dropna()
            n = len(subset)
            if n == 0:
                continue

            try:
                mean_val = subset.mean()
                std_val = subset.std()
                skew_val = skew(subset) if n > 2 else float("nan")
                kurt_val = kurtosis(subset) if n > 2 else float("nan")
            except TypeError:
                continue

            if n >= 3:
                w_stat, p_val = shapiro(subset)
            else:
                w_stat, p_val = float("nan"), float("nan")

            results_records.append({
                "Variable": col,
                "Group": g,
                "N": n,
                "Mean": mean_val,
                "StdDev": std_val,
                "Skew": skew_val,
                "Kurtosis": kurt_val,
                "Shapiro_W": w_stat,
                "Shapiro_p": p_val
            })

            if save_plots:
                plt.figure(figsize=(6, 4))
                sns.histplot(subset, kde=True, color='blue', bins=10)
                title = f"Histogram of {col} (Group={g})" if group_col else f"Histogram of {col}"
                plt.title(title)
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.tight_layout()
                hist_file = f"hist_{col}_{g}.png" if group_col else f"hist_{col}.png"
                plt.savefig(os.path.join(plot_dir, hist_file))
                plt.close()

                fig = sm.qqplot(subset, line='s')
                plt.title(f"QQ-plot of {col} (Group={g})" if group_col else f"QQ-plot of {col}")
                plt.tight_layout()
                qq_file = f"qq_{col}_{g}.png" if group_col else f"qq_{col}.png"
                plt.savefig(os.path.join(plot_dir, qq_file))
                plt.close()

    df_out = pd.DataFrame(results_records)
    csv_path = os.path.join(out_dir, results_filename)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df_out.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_out.to_csv(csv_path, index=False)

    print(f"Appended {len(df_out)} normality records to: {csv_path}")

def main():
    df = load_data(DB_PATH)
    df = map_feedback_responses(df)

    # Check for confidence and Likert answers
    normality_analysis(
        df=df,
        group_col="model_name",
        numeric_cols=["confidence", "q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"],
        out_dir=OUTPUT_DIR,
        save_plots=True,
        results_filename="normality_test_results.csv"
    )

    # Check for accuracy by itself
    acc_df = aggregate_accuracy(df)
    normality_analysis(
        df=acc_df,
        group_col="model_name",
        numeric_cols=["mean_accuracy"],
        out_dir=OUTPUT_DIR,
        save_plots=True,
        results_filename="normality_test_results.csv"
    )

    # Check for acc - conf difference
    conf_df = aggregate_confidence(df)
    merged = pd.merge(acc_df, conf_df, on=["subject_number", "model_name"])
    merged["acc_conf_diff"] = merged["mean_accuracy"] - merged["mean_confidence"]

    normality_analysis(
        df=merged,
        group_col="model_name",
        numeric_cols=["acc_conf_diff"],
        out_dir=OUTPUT_DIR,
        save_plots=True,
        results_filename="normality_test_results.csv"
    )

if __name__ == "__main__":
    main()