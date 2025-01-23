import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
import os
from config import DB_PATH, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data from SQLite database
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

data = load_data(DB_PATH)

# 1. Bar Chart: Average Confidence per Model with Error Bars
def plot_average_confidence_chart(df):
    avg_confidence = df.groupby("model_name")["confidence"].mean()
    conf_intervals = df.groupby("model_name")["confidence"].sem() * 1.96

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(avg_confidence.index, avg_confidence, yerr=conf_intervals, capsize=5, color='skyblue', edgecolor='black')

    # Annotate each bar with its value inside the bar
    for bar, value in zip(bars, avg_confidence):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{value:.2f}", ha='center', va='center', fontsize=10, color='black')

    ax.set_title("Average Confidence per Model with Error Bars")
    ax.set_ylabel("Average Confidence")
    ax.set_xlabel("Model")
    ax.set_xticks(range(len(avg_confidence.index)))
    ax.set_xticklabels(avg_confidence.index, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_per_model.png"))
    plt.close()

plot_average_confidence_chart(data)

# 2. ANOVA Test for Feedback Answers
def calculate_anova_feedback(df):
    feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]
    likert_mapping = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
        "Can't answer": None
    }

    with open(os.path.join(OUTPUT_DIR, "anova_results.txt"), "w") as f:
        for col in feedback_cols:
            # Map Likert responses to numerical values
            df[f"{col}_numeric"] = df[col].map(likert_mapping)

            # Prepare data for ANOVA
            grouped_data = [df[df["model_name"] == model][f"{col}_numeric"].dropna() for model in df["model_name"].unique()]

            # Perform ANOVA
            f_stat, p_val = f_oneway(*grouped_data)

            # Save ANOVA results
            f.write(f"ANOVA Results for Distribution of Q{feedback_cols.index(col)+1} Responses:\n")
            f.write(f"F-statistic: {f_stat:.2f}\n")
            f.write(f"P-value: {p_val:.3f}\n\n")

            # Pairwise comparisons with Bonferroni correction
            model_names = df["model_name"].unique()
            pairwise_pvals = []
            model_pairs = []
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:
                        data1 = df[df["model_name"] == model1][f"{col}_numeric"].dropna()
                        data2 = df[df["model_name"] == model2][f"{col}_numeric"].dropna()
                        _, pval = ttest_ind(data1, data2)
                        pairwise_pvals.append(pval)
                        model_pairs.append((model1, model2))

            _, corrected_pvals, _, _ = multipletests(pairwise_pvals, method="bonferroni")
            f.write(f"Pairwise Comparisons for Q{feedback_cols.index(col)+1}:\n")
            for (model1, model2), pval in zip(model_pairs, corrected_pvals):
                f.write(f"  {model1} vs {model2}: Corrected P-value = {pval:.3f}\n")
            f.write("\n")

        print("ANOVA results saved to anova_results.txt")

calculate_anova_feedback(data)

# 3. Stacked Bar Chart: Likert Responses by Model
def plot_likert_responses(df):
    feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]
    response_order = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree", "Can't answer"]
    for col in feedback_cols:
        likert_model_counts = (
            df.groupby(["model_name", col]).size().unstack(level=0, fill_value=0).reindex(response_order)
        )
        likert_model_counts.plot(kind="bar", figsize=(10, 8), title=f"Distribution of Q{feedback_cols.index(col)+1} Responses by Model")
        plt.ylabel("Count")
        plt.xlabel("Likert Scale Response")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Q{feedback_cols.index(col)+1}_responses_distribution.png"))
        plt.close()

plot_likert_responses(data)

# 4. Count Participants and Samples
def count_participants_and_samples(df):
    # Count unique participants
    participants = df["subject_number"].nunique()

    # Total samples
    total_samples = df.shape[0]

    # Print results
    print(f"Total Participants: {participants}")
    print(f"Total Samples: {total_samples}")

count_participants_and_samples(data)

# 5. Feedback Distributions Visualization
def plot_feedback_distributions(df):
    feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]
    likert_mapping = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
        "Can't answer": 6  # Treat "Can't answer" as its own category
    }

    for col in feedback_cols:
        df[f"{col}_numeric"] = df[col].map(likert_mapping)
        sns.boxplot(x="model_name", y=f"{col}_numeric", data=df)
        plt.title(f"Feedback Distribution for Q{feedback_cols.index(col)+1}")
        plt.ylabel("Likert Scale (1-6)")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Q{feedback_cols.index(col)+1}_feedback_distribution.png"))
        plt.close()

plot_feedback_distributions(data)

# 6. Check variability and can't answer mapping
def check_var():
    feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]

    for col in feedback_cols:
        print(f"Question {col}:")
        for model in data["model_name"].unique():
            group_data = data[data["model_name"] == model][f"{col}_numeric"]
            print(f"  {model}: Mean = {group_data.mean():.2f}, Std = {group_data.std():.2f}")

    for col in feedback_cols:
        print(f"{col} 'Can't answer' counts:")
        print(data[data[col] == "Can't answer"]["model_name"].value_counts())

# check_var()