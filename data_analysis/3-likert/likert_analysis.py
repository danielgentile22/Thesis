#!/usr/bin/env python3
# likert_analysis.py

import os
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

DB_PATH = "../results_processed/experiment_results.db"
OUTPUT_DIR = "3-likert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# These are your Likert columns (mapped 1-5, with np.nan for "Can't answer")
LIKERT_COLS = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]

def load_data(db_path):
    """Load the main table from your experiment_results SQLite database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"  # Adjust table name/columns if needed
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def aggregate_likert_responses(df, subject_col, model_col, question_col):
    """
    Group by (participant, model) and compute the mean rating for the given question_col.
    Returns a DataFrame with columns: [subject_col, model_col, 'mean_rating'].
    """
    # Some participants may have multiple rows per question & model if they drew multiple digits.
    grouped = (df.groupby([subject_col, model_col])[question_col]
                 .mean()  # average rating across multiple trials for the same participant-model
                 .reset_index(name="mean_rating"))
    return grouped

def run_friedman_test(agg_df, subject_col, model_col, value_col="mean_rating"):
    """
    Perform Friedman test for repeated measures across the 3 models.
    Expects one row per participant-model with a single numeric rating.
    """
    # Pivot => each row is a participant, each column is a model's rating
    pivoted = agg_df.pivot(index=subject_col, columns=model_col, values=value_col)
    pivoted.dropna(axis=0, how="any", inplace=True)  # remove participants missing a model's rating
    model_columns = list(pivoted.columns)
    data_arrays = [pivoted[m] for m in model_columns]

    # Friedman
    stat, p_value = friedmanchisquare(*data_arrays)
    return stat, p_value, pivoted

def posthoc_wilcoxon_friedman(pivoted_df):
    """
    If Friedman is significant, do Wilcoxon signed-rank pairwise tests with Bonferroni correction.
    """
    model_cols = pivoted_df.columns.tolist()
    pairs = []
    pvals = []
    
    for i in range(len(model_cols)):
        for j in range(i+1, len(model_cols)):
            col1 = model_cols[i]
            col2 = model_cols[j]
            w_stat, p_val = wilcoxon(pivoted_df[col1], pivoted_df[col2])
            pairs.append((col1, col2))
            pvals.append(p_val)
    
    # Bonferroni correction
    _, corrected_pvals, _, _ = multipletests(pvals, method="bonferroni")
    
    results = []
    for (mA, mB), raw_p, corr_p in zip(pairs, pvals, corrected_pvals):
        results.append({
            "Comparison": f"{mA} vs {mB}",
            "Wilcoxon_p": raw_p,
            "Bonferroni_p": corr_p
        })
    return pd.DataFrame(results)

def plot_likert_boxplot(agg_df, model_col, value_col, question_name):
    """
    Creates a boxplot for the aggregated Likert ratings, one box per model.
    Saves the figure as boxplot_<question_name>.png
    """
    plt.figure(figsize=(8,6))
    sns.boxplot(x=model_col, y=value_col, data=agg_df)
    # Add individual points
    sns.stripplot(x=model_col, y=value_col, data=agg_df, color='black', alpha=0.5, jitter=True)
    plt.title(f"Likert Ratings for {question_name} by Model")
    plt.xlabel("Model")
    plt.ylabel("Rating (1=Strongly Disagree, 5=Strongly Agree)")
    plt.ylim(0.5, 5.5)  # just to give some padding around 1..5
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{question_name}.png"))
    plt.close()

def main():
    df = load_data(DB_PATH)
    
    subject_col = "subject_number"  # or "participant_id", adjust as needed
    model_col = "model_name"

    import numpy as np

    feedback_mapping = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
        "Can't answer": np.nan
    }

    for col in LIKERT_COLS:
        df[col] = df[col].map(feedback_mapping)

    results_filename = os.path.join(OUTPUT_DIR, "likert_stats.txt")
    with open(results_filename, "w") as stats_file:
        stats_file.write("Likert Scale Analysis Results\n")
        stats_file.write("="*40 + "\n\n")

        # Loop over each question (q1_answer, q2_answer, etc.)
        for question_col in LIKERT_COLS:
            # 1) Aggregate
            agg = aggregate_likert_responses(df, subject_col, model_col, question_col)

            # 2) Plot boxplot for this question
            plot_likert_boxplot(agg, model_col, "mean_rating", question_col)

            # 3) Friedman test
            stat, p_value, pivoted = run_friedman_test(agg, subject_col, model_col, "mean_rating")

            # 4) Write results
            stats_file.write(f"Question: {question_col}\n")
            stats_file.write(f"  Friedman statistic = {stat:.3f}, p-value = {p_value:.5f}\n")

            # 5) If significant, do post-hoc
            if p_value < 0.05:
                posthoc_df = posthoc_wilcoxon_friedman(pivoted)
                stats_file.write("  Post-hoc Wilcoxon (Bonferroni corrected):\n")
                stats_file.write(posthoc_df.to_string(index=False))
                stats_file.write("\n\n")
            else:
                stats_file.write("  No significant difference -> No post-hoc tests.\n\n")

    print("Likert analysis complete. See 'analysis_results/' for outputs.")

if __name__ == "__main__":
    main()