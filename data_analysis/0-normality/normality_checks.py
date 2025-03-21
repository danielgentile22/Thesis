#!/usr/bin/env python3
# normality_checks.py

import os
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro, skew, kurtosis
import numpy as np

DB_PATH = "../results_processed/experiment_results.db"
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    """Load the main table from your experiment_results SQLite database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def map_feedback_responses(df):
    """
    Convert string-based Likert responses to numeric values.
    """
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

def normality_analysis(
    df,
    group_col=None,
    numeric_cols=None,
    out_dir="analysis_results",
    save_plots=True
):
    """
    Check and plot distributions for specified numeric columns.
    Optionally group by 'group_col' (e.g., 'model_name').
    
    :param df: pandas DataFrame containing your data
    :param group_col: column name to group by (e.g., 'model_name') or None for no grouping
    :param numeric_cols: list of numeric columns to test (e.g., ['confidence','q1_answer',...])
    :param out_dir: folder to save output (plots, csvs)
    :param save_plots: if True, save hist/qq plots; otherwise just show them
    """
    
    if numeric_cols is None:
        numeric_cols = []
    
    # Prepare subfolder for normality plots
    plot_dir = os.path.join(out_dir, "normality_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Prepare a list or dict to store Shapiro results
    results_records = []

    if group_col:
        groups = df[group_col].unique()
    else:
        # If there's no group_col, treat the entire dataset as one group
        groups = ["All_Data"]

    for col in numeric_cols:
        for g in groups:
            if group_col:
                subset = df.loc[df[group_col] == g, col].dropna()
            else:
                subset = df[col].dropna()

            n = len(subset)
            if n == 0:
                # Skip empty group
                continue

            # Compute descriptive stats safely only if subset is numeric
            try:
                mean_val = subset.mean()
                std_val = subset.std()
                skew_val = skew(subset) if n > 2 else float("nan")
                kurt_val = kurtosis(subset) if n > 2 else float("nan")
            except TypeError as e:
                print(f"Cannot compute numeric stats for {col}, group={g}: {e}")
                continue

            # Shapiro–Wilk test (requires at least 3 data points)
            if n >= 3:
                w_stat, p_val = shapiro(subset)
            else:
                w_stat, p_val = float("nan"), float("nan")

            # Record results
            results_records.append({
                "Variable": col,
                "Group": g,
                "N": n,
                "Mean": mean_val,
                "StdDev": std_val,
                "Skewness": skew_val,
                "Kurtosis": kurt_val,
                "Shapiro_W": w_stat,
                "Shapiro_p": p_val
            })

            # Plot distributions if desired
            if save_plots:
                # 1) Histogram + KDE
                sns.set_style("whitegrid")
                plt.figure(figsize=(6, 4))
                sns.histplot(subset, kde=True, color='blue', bins=10)
                title_str = f"Histogram of {col}"
                if group_col: 
                    title_str += f" (Group={g})"
                plt.title(title_str)
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.tight_layout()
                filename_hist = f"hist_{col}_{g}.png" if group_col else f"hist_{col}.png"
                plt.savefig(os.path.join(plot_dir, filename_hist))
                plt.close()

                # 2) QQ-plot
                fig = sm.qqplot(subset, line='s')
                title_str = f"QQ-plot of {col}"
                if group_col:
                    title_str += f" (Group={g})"
                plt.title(title_str)
                plt.tight_layout()
                filename_qq = f"qq_{col}_{g}.png" if group_col else f"qq_{col}.png"
                plt.savefig(os.path.join(plot_dir, filename_qq))
                plt.close()

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results_records)
    results_df.to_csv(os.path.join(out_dir, "normality_test_results.csv"), index=False)
    print(f"Normality results saved to: {os.path.join(out_dir, 'normality_test_results.csv')}")

def main():
    # 1) Load the data
    df = load_data(DB_PATH)

    # 2) Convert string-based Likert responses to numeric codes
    df = map_feedback_responses(df)

    # 3) Define columns to check for normality (after mapping to numeric)
    #    Adjust if your column names are different.
    numeric_cols = [
        "confidence",
        "q1_answer", 
        "q2_answer", 
        "q3_answer", 
        "q4_answer", 
        "q5_answer"
    ]

    # 4) Run normality analysis, grouped by model_name (if you want per-model checks)
    normality_analysis(
        df=df,
        group_col="model_name",
        numeric_cols=numeric_cols,
        out_dir=OUTPUT_DIR,
        save_plots=True
    )

if __name__ == "__main__":
    main()