#!/usr/bin/env python3
# confidence_analysis.py

import os
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

DB_PATH = "../results_processed/experiment_results.db"
OUTPUT_DIR = "confidence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    """Load the main table from your experiment_results SQLite database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"  # Adjust table name/columns if needed
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def aggregate_confidence(df, subject_col="subject_number",
                         model_col="model_name", conf_col="confidence"):
    """
    Group by (participant, model) and compute the mean confidence.
    Returns a DataFrame with columns [subject_col, model_col, 'mean_confidence'].
    """
    grouped = df.groupby([subject_col, model_col])[conf_col].mean().reset_index()
    grouped.rename(columns={conf_col: "mean_confidence"}, inplace=True)
    return grouped

def run_friedman_test(agg_df, subject_col="subject_number",
                      model_col="model_name", value_col="mean_confidence"):
    """
    Perform a Friedman test for repeated measures across the 3 models.
    Assumes each subject has one mean_confidence value per model.
    """
    # Pivot so each row = one participant, each column = one model
    pivoted = agg_df.pivot(index=subject_col, columns=model_col, values=value_col)
    pivoted.dropna(axis=0, how="any", inplace=True)  # drop participants missing a model
    # Extract model columns in a consistent order
    # e.g. ["Base Model", "MC-Dropout", "Ensemble Model"]
    model_columns = list(pivoted.columns)
    data_arrays = [pivoted[mc].values for mc in model_columns]

    # Friedman test
    stat, p_value = friedmanchisquare(*data_arrays)
    return stat, p_value, pivoted

def posthoc_wilcoxon_friedman(pivoted_df):
    """
    If Friedman is significant, do post-hoc Wilcoxon signed-rank tests
    with Bonferroni correction. 
    pivoted_df: DataFrame with one row per participant, one column per model.
    """
    model_columns = list(pivoted_df.columns)
    pairs = []
    pvals = []

    # Compare each pair of models
    for i in range(len(model_columns)):
        for j in range(i + 1, len(model_columns)):
            col1 = model_columns[i]
            col2 = model_columns[j]
            # Wilcoxon signed-rank test on paired data
            w_stat, p_val = wilcoxon(pivoted_df[col1], pivoted_df[col2])
            pairs.append((col1, col2))
            pvals.append(p_val)

    # Bonferroni correction
    _, corrected_pvals, _, _ = multipletests(pvals, method="bonferroni")

    # Prepare a results summary
    results = []
    for (modelA, modelB), raw_p, corr_p in zip(pairs, pvals, corrected_pvals):
        results.append({
            "Comparison": f"{modelA} vs {modelB}",
            "Wilcoxon_p": raw_p,
            "Bonferroni_p": corr_p
        })
    results_df = pd.DataFrame(results)
    return results_df

def plot_boxplot(agg_df, model_col="model_name", value_col="mean_confidence"):
    """
    Creates a boxplot of confidence by model.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=model_col, y=value_col, data=agg_df)
    sns.stripplot(x=model_col, y=value_col, data=agg_df,
                  color="black", alpha=0.5, jitter=True)
    plt.title("Mean Confidence by Model (per Participant)")
    plt.ylabel("Confidence (%)")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_boxplot.png"))
    plt.close()

def plot_bar_chart(agg_df, model_col="model_name", value_col="mean_confidence"):
    """
    Creates a bar chart with mean +/- std or sem across participants.
    """
    # Compute mean & std for each model
    stats_df = agg_df.groupby(model_col)[value_col].agg(["mean", "sem"]).reset_index()

    plt.figure(figsize=(8, 6))
    bars = plt.bar(stats_df[model_col],
                   stats_df["mean"],
                   yerr=stats_df["sem"],
                   capsize=5,
                   color="skyblue",
                   edgecolor="black")

    # Annotate each bar
    for bar, mean_val in zip(bars, stats_df["mean"]):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 mean_val, 
                 f"{mean_val:.2f}", 
                 ha='center', va='bottom')

    plt.title("Average Participant Confidence by Model (Mean ± SEM)")
    plt.ylabel("Confidence (%)")
    plt.ylim(0, 100)  # Assuming 0-100% range
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_barchart.png"))
    plt.close()

def main():
    # 1) Load the data
    df = load_data(DB_PATH)

    # 2) Aggregate confidence by (participant, model)
    agg_conf = aggregate_confidence(df,
                                    subject_col="subject_number",
                                    model_col="model_name",
                                    conf_col="confidence")

    # 3) Plot the distribution
    plot_boxplot(agg_conf, "model_name", "mean_confidence")
    plot_bar_chart(agg_conf, "model_name", "mean_confidence")

    # 4) Run Friedman test
    friedman_stat, friedman_p, pivoted_df = run_friedman_test(
        agg_conf,
        subject_col="subject_number",
        model_col="model_name",
        value_col="mean_confidence"
    )

    print(f"Friedman Test statistic = {friedman_stat:.3f}, p-value = {friedman_p:.5f}")

    with open(os.path.join(OUTPUT_DIR, "confidence_stats.txt"), "w") as f:
        f.write(f"Friedman Test for Confidence:\n")
        f.write(f"Statistic = {friedman_stat:.3f}, p-value = {friedman_p:.5f}\n\n")

        # 5) If significant, post-hoc Wilcoxon
        if friedman_p < 0.05:
            posthoc_df = posthoc_wilcoxon_friedman(pivoted_df)
            f.write("Post-hoc Wilcoxon Signed-Rank (Bonferroni corrected):\n")
            f.write(posthoc_df.to_string(index=False))
            f.write("\n")
            print("Post-hoc Wilcoxon results:\n", posthoc_df)
        else:
            f.write("No significant difference -> No post-hoc tests performed.\n")

    print("Analysis complete. See 'analysis_results/' folder for outputs.")

if __name__ == "__main__":
    main()