#!/usr/bin/env python3
# accuracy_analysis.py

import os
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

DB_PATH = "../results_processed/experiment_results.db"
OUTPUT_DIR = "2-accuracy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    """Load the main table from your experiment_results SQLite database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"  # Adjust table name/columns if needed
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def aggregate_accuracy(df, subject_col="subject_number",
                       model_col="model_name",
                       true_col="intended_digit",
                       pred_col="predicted_digit"):
    """
    Group by (participant, model) and compute the fraction of correct predictions.
    Returns a DataFrame with columns [subject_col, model_col, 'mean_accuracy'].
    """
    def accuracy_func(x):
        return np.mean(x[true_col] == x[pred_col])

    grouped = df.groupby([subject_col, model_col]).apply(accuracy_func).reset_index(name="mean_accuracy")
    
    return grouped

def run_friedman_test(agg_df, subject_col="subject_number",
                      model_col="model_name", value_col="mean_accuracy"):
    """
    Perform a Friedman test for repeated measures across the 3 models.
    Assumes each subject has one mean_accuracy value per model.
    """
    # Pivot so each row = one participant, each column = one model
    pivoted = agg_df.pivot(index=subject_col, columns=model_col, values=value_col)
    pivoted.dropna(axis=0, how="any", inplace=True)  # drop participants missing a model
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

    for i in range(len(model_columns)):
        for j in range(i + 1, len(model_columns)):
            col1 = model_columns[i]
            col2 = model_columns[j]
            w_stat, p_val = wilcoxon(pivoted_df[col1], pivoted_df[col2])
            pairs.append((col1, col2))
            pvals.append(p_val)

    # Bonferroni correction
    _, corrected_pvals, _, _ = multipletests(pvals, method="bonferroni")

    results = []
    for (modelA, modelB), raw_p, corr_p in zip(pairs, pvals, corrected_pvals):
        results.append({
            "Comparison": f"{modelA} vs {modelB}",
            "Wilcoxon_p": raw_p,
            "Bonferroni_p": corr_p
        })
    results_df = pd.DataFrame(results)
    return results_df

def plot_boxplot(agg_df, model_col="model_name", value_col="mean_accuracy"):
    """
    Creates a boxplot of participant-level accuracy by model.
    'mean_accuracy' is typically between 0 and 1 (i.e., fraction correct).
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=model_col, y=value_col, data=agg_df)
    # Optional: add a stripplot to show individual points
    sns.stripplot(x=model_col, y=value_col, data=agg_df, color='black', alpha=0.5, jitter=True)
    plt.title("Mean Accuracy by Model (per Participant)")
    plt.ylabel("Accuracy (fraction correct)")
    plt.xlabel("Model")
    plt.ylim(0, 1)  # since accuracy is 0–1
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_boxplot.png"))
    plt.close()

def plot_bar_chart(agg_df, model_col="model_name", value_col="mean_accuracy"):
    """
    Creates a bar chart with mean ± SEM for participant-level accuracy.
    """
    stats_df = agg_df.groupby(model_col)[value_col].agg(["mean", "sem"]).reset_index()

    plt.figure(figsize=(8, 6))
    bars = plt.bar(stats_df[model_col],
                   stats_df["mean"],
                   yerr=stats_df["sem"],
                   capsize=5,
                   color="lightsalmon",
                   edgecolor="black")

    # Annotate each bar
    for bar, mean_val in zip(bars, stats_df["mean"]):
        plt.text(bar.get_x() + bar.get_width()/2,
                 mean_val,
                 f"{mean_val*100:.1f}%",   # convert fraction to %
                 ha='center', va='bottom')

    plt.title("Average Participant Accuracy by Model (Mean ± SEM)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_barchart.png"))
    plt.close()

def main():
    # 1) Load the data
    df = load_data(DB_PATH)

    # 2) Aggregate accuracy by (participant, model)
    agg_acc = aggregate_accuracy(df,
                                 subject_col="subject_number",
                                 model_col="model_name",
                                 true_col="intended_digit",
                                 pred_col="predicted_digit")

    # 3) Visualizations
    plot_boxplot(agg_acc, "model_name", "mean_accuracy")
    plot_bar_chart(agg_acc, "model_name", "mean_accuracy")

    # 4) Friedman test
    stat, p_value, pivoted = run_friedman_test(
        agg_acc,
        subject_col="subject_number",
        model_col="model_name",
        value_col="mean_accuracy"
    )

    print(f"Friedman Test statistic = {stat:.3f}, p-value = {p_value:.5f}")

    with open(os.path.join(OUTPUT_DIR, "accuracy_stats.txt"), "w") as f:
        f.write(f"Friedman Test for Accuracy:\n")
        f.write(f"Statistic = {stat:.3f}, p-value = {p_value:.5f}\n\n")

        # 5) If significant, post-hoc Wilcoxon
        if p_value < 0.05:
            posthoc_df = posthoc_wilcoxon_friedman(pivoted)
            f.write("Post-hoc Wilcoxon Signed-Rank (Bonferroni corrected):\n")
            f.write(posthoc_df.to_string(index=False))
            f.write("\n")
            print("Post-hoc Wilcoxon results:\n", posthoc_df)
        else:
            f.write("No significant difference -> No post-hoc tests.\n")

    print("Accuracy analysis complete. Check 'analysis_results/' for outputs.")

if __name__ == "__main__":
    main()
