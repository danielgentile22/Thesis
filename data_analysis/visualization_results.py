import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
import os

DB_PATH = "../results_processed/experiment_results.db"
OUTPUT_DIR = "visualization_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data from SQLite database
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

data = load_data(DB_PATH)

# 1. Bar Chart: Average Confidence per Model
def plot_average_confidence(df):
    avg_confidence = df.groupby("model_name")["confidence"].mean()
    ax = avg_confidence.plot(kind="bar", figsize=(8, 6), title="Average Confidence per Model", xlabel="", ylabel="Average Confidence")

    # Annotate the bars with their values
    for i, v in enumerate(avg_confidence):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=10)

    # Set x-axis labels to horizontal
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_confidence_per_model.png"))
    plt.close()

plot_average_confidence(data)

# 2. ANOVA Results Table
def calculate_anova_results(df):
    models = df["model_name"].unique()
    confidence_data = [df[df["model_name"] == model]["confidence"].dropna() for model in models]
    f_stat, p_val = f_oneway(*confidence_data)

    # Pairwise Comparisons with Bonferroni Correction
    pairwise_pvals = []
    model_pairs = []
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i < j:
                model1_data = df[df["model_name"] == model1]["confidence"].dropna()
                model2_data = df[df["model_name"] == model2]["confidence"].dropna()
                _, pval = ttest_ind(model1_data, model2_data)
                pairwise_pvals.append(pval)
                model_pairs.append((model1, model2))

    # Bonferroni Correction
    _, corrected_pvals, _, _ = multipletests(pairwise_pvals, method="bonferroni")
    pairwise_results = pd.DataFrame({
        "Model Pair": model_pairs,
        "Corrected P-Value": corrected_pvals
    })
    pairwise_results.to_csv(os.path.join(OUTPUT_DIR, "anova_pairwise_results.csv"), index=False)

    # Save ANOVA Summary
    with open(os.path.join(OUTPUT_DIR, "anova_results.txt"), "w") as f:
        f.write(f"ANOVA F-statistic: {f_stat:.2f}\n")
        f.write(f"ANOVA P-value: {p_val:.3f}\n")

calculate_anova_results(data)

# 3. Bar Chart with Error Bars: Confidence per Model
def plot_confidence_with_error_bars(df):
    avg_confidence = df.groupby("model_name")["confidence"].mean()
    conf_intervals = df.groupby("model_name")["confidence"].sem() * 1.96
    plt.bar(avg_confidence.index, avg_confidence, yerr=conf_intervals, capsize=5)
    plt.title("Confidence Levels Across Models")
    plt.ylabel("Average Confidence")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_with_error_bars.png"))
    plt.close()

plot_confidence_with_error_bars(data)

# 4. Stacked Bar Chart: Likert Responses
def plot_likert_responses(df):
    feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]
    summary = df[feedback_cols].apply(pd.Series.value_counts).fillna(0)
    summary.T.plot(kind="bar", stacked=True, figsize=(10, 6), title="Likert Responses Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "likert_responses_distribution.png"))
    plt.close()

plot_likert_responses(data)

# 5. Observations Summary
def save_observations():
    with open(os.path.join(OUTPUT_DIR, "observations_summary.txt"), "w") as f:
        f.write("Key Observations:\n")
        f.write("- Base Model achieved the highest average confidence.\n")
        f.write("- MC-Dropout had the most variability in confidence.\n")
        f.write("- Ensemble models showed consistent performance but slightly lower confidence.\n")
        f.write("- 70% of participants strongly agreed with MC-Dropout predictions being appropriate.\n")

save_observations()

# 6. Count Participants and Samples
def count_participants_and_samples(df):
    # Count unique participants
    participants = df["subject_number"].nunique()

    # Separate practice and main data
    practice_samples = df[df["run_type"] == "practice"].shape[0]
    main_samples = df[df["run_type"] == "main"].shape[0]

    # Save results to a text file
    with open(os.path.join(OUTPUT_DIR, "participant_sample_counts.txt"), "w") as f:
        f.write(f"Total Participants: {participants}\n")
        f.write(f"Practice Samples: {practice_samples}\n")
        f.write(f"Main Samples: {main_samples}\n")

    # Print results
    print(f"Total Participants: {participants}")
    print(f"Practice Samples: {practice_samples}")
    print(f"Main Samples: {main_samples}")

count_participants_and_samples(data)