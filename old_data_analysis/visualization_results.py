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

# Define feedback columns
feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]

# 1. Bar Chart: Average Confidence per Model with Standard Error Bars
def plot_average_confidence_chart(df):
    avg_confidence = df.groupby("model_name")["confidence"].mean()
    conf_intervals = df.groupby("model_name")["confidence"].sem()  # No multiplier, plain SEM

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(avg_confidence.index, avg_confidence, yerr=conf_intervals, capsize=5, color='skyblue', edgecolor='black')

    # Annotate each bar with its value inside the bar
    for bar, value in zip(bars, avg_confidence):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{value:.2f}", ha='center', va='center', fontsize=10, color='black')

    ax.set_title("Average Confidence per Model with SE Bars")
    ax.set_ylabel("Average Confidence (%)")
    # ax.set_xlabel("Model")
    ax.set_ylim(0, 100)  # Ensure y-axis goes up to 100
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_per_model.png"))
    plt.close()

plot_average_confidence_chart(data)

# 2. Bar Chart: Accuracy per Model (Ordered by Accuracy)
def plot_accuracy_per_model(df):
    # Calculate accuracy for each model
    accuracies = df.groupby("model_name").apply(lambda x: (x["intended_digit"] == x["predicted_digit"]).mean())
    
    # Sort models by accuracy (descending order)
    accuracies = accuracies.sort_values(ascending=False)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(accuracies.index, accuracies * 100, capsize=5, color='limegreen', edgecolor='black')

    # Annotate each bar with its value inside the bar
    for bar, value in zip(bars, accuracies * 100):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2, f"{value:.1f}%", ha='center', va='center', fontsize=10, color='black')

    ax.set_title("Accuracy per Model")
    ax.set_ylabel("Accuracy (%)")
    # ax.set_xlabel("Model")
    ax.set_ylim(0, 100)  # Ensure y-axis goes up to 100 for consistency
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_per_model_ordered.png"))
    plt.close()

plot_accuracy_per_model(data)

# 3. ANOVA Test for Feedback Answers
def calculate_anova_feedback(df):
    likert_mapping = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
        "Can't answer": 6  # Include Can't answer as its own category
    }

    with open(os.path.join(OUTPUT_DIR, "anova_results.txt"), "w") as f:
        for col in feedback_cols:
            df[f"{col}_numeric"] = df[col].map(likert_mapping)

            grouped_data = [df[df["model_name"] == model][f"{col}_numeric"].dropna() for model in df["model_name"].unique()]
            f_stat, p_val = f_oneway(*grouped_data)

            f.write(f"ANOVA Results for Distribution of Q{feedback_cols.index(col) + 1} Responses:\n")
            f.write(f"F-statistic: {f_stat:.2f}\n")
            f.write(f"P-value: {p_val:.3f}\n\n")

            model_names = df["model_name"].unique()
            pairwise_pvals, model_pairs = [], []
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:
                        data1 = df[df["model_name"] == model1][f"{col}_numeric"].dropna()
                        data2 = df[df["model_name"] == model2][f"{col}_numeric"].dropna()
                        _, pval = ttest_ind(data1, data2)
                        pairwise_pvals.append(pval)
                        model_pairs.append((model1, model2))

            _, corrected_pvals, _, _ = multipletests(pairwise_pvals, method="bonferroni")
            f.write(f"Pairwise Comparisons for Q{feedback_cols.index(col) + 1}:\n")
            for (model1, model2), pval in zip(model_pairs, corrected_pvals):
                f.write(f"  {model1} vs {model2}: Corrected P-value = {pval:.3f}\n")
            f.write("\n")

calculate_anova_feedback(data)

# 5. Feedback Distributions Visualization
def plot_feedback_distributions(df):
    likert_mapping = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
        "Can't answer": 6
    }

    for col in feedback_cols:
        df[f"{col}_numeric"] = df[col].map(likert_mapping)
        sns.boxplot(x="model_name", y=f"{col}_numeric", data=df)
        plt.title(f"Feedback Distribution for Q{feedback_cols.index(col) + 1}")
        plt.ylabel("Likert Scale (1-6)")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Q{feedback_cols.index(col) + 1}_feedback_distribution.png"))
        plt.close()

# plot_feedback_distributions(data)

"""
# Print the number of times each model was used
model_usage_counts = data["model_name"].value_counts()
print("Number of times each model was used:")
for model, count in model_usage_counts.items():
    print(f"  {model}: {count}")
"""

# 4. Stacked Bar Chart: Likert Responses by Model
def plot_likert_responses(df):
    response_order = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree", "Can't answer"]
    for col in feedback_cols:
        likert_model_counts = (
            df.groupby(["model_name", col]).size().unstack(level=0, fill_value=0).reindex(response_order)
        )
        ax = likert_model_counts.plot(kind="bar", figsize=(10, 8), title=f"Distribution of Q{feedback_cols.index(col) + 1} Responses by Model")
        ax.set_ylabel("Count")
        ax.set_xlabel("Likert Scale Response")
        ax.legend(title="Model Name")

        # Set horizontal x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Q{feedback_cols.index(col) + 1}_responses_distribution.png"))
        plt.close()

plot_likert_responses(data)

# 6. Stacked Bar Charts: Compare Base Model with MC-Dropout + Ensemble Average for Each Question (Normalized)
def plot_combined_likert_responses_per_question_normalized(df):
    feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]
    response_order = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree", "Can't answer"]

    for col in feedback_cols:
        # Group data by model and Likert responses
        likert_model_counts = (
            df.groupby(["model_name", col]).size().unstack(level=0, fill_value=0)
        )

        # Normalize the counts to proportions
        likert_model_counts = likert_model_counts.div(likert_model_counts.sum(axis=0), axis=1)

        # Extract Base Model proportions
        base_model_counts = likert_model_counts["Base Model"]

        # Calculate the average proportions for MC-Dropout and Ensemble
        combined_average_counts = (
            likert_model_counts[["MC-Dropout", "Ensemble Model"]].mean(axis=1)
        )

        # Combine Base Model and Average into a new DataFrame
        combined_counts = pd.DataFrame({
            "Base Model": base_model_counts,
            "MC-Dropout + Ensemble (Avg)": combined_average_counts
        })

        # Create the stacked bar plot
        ax = combined_counts.reindex(response_order).plot(
            kind="bar", figsize=(10, 8), color=["steelblue", "orange"], edgecolor="black"
        )

        # Update the title, labels, and legend
        ax.set_title(f"Normalized Distribution of Q{feedback_cols.index(col) + 1} Responses by Model")
        ax.set_ylabel("Proportion")
        ax.set_xlabel("Likert Scale Response")
        ax.legend(title="Model Name")

        # Set horizontal x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        # Save the plot for this question
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Q{feedback_cols.index(col) + 1}_combined_likert_responses_normalized.png"))
        plt.close()

plot_combined_likert_responses_per_question_normalized(data)