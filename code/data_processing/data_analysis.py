import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DB_PATH = "experiment_results.db"

# Load data from SQLite database
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

data = load_data(DB_PATH)

# Create an output directory
import os
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Accuracy and Model Performance
def calculate_accuracy(df):
    model_groups = df.groupby("model_name")[["intended_digit", "predicted_digit"]]
    accuracies = model_groups.apply(lambda x: (x["intended_digit"] == x["predicted_digit"]).mean())
    accuracies.to_csv(os.path.join(OUTPUT_DIR, "model_accuracies.csv"))
    return accuracies

def plot_confusion_matrix(df, model_name):
    subset = df[df["model_name"] == model_name]
    cm = confusion_matrix(subset["intended_digit"], subset["predicted_digit"], labels=range(10))
    disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.replace(' ', '_')}.png"))
    plt.close()

accuracies = calculate_accuracy(data)
for model in data["model_name"].unique():
    plot_confusion_matrix(data, model)

# 2. Feedback Analysis
def feedback_summary(df):
    feedback_cols = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]
    summary = df[feedback_cols].apply(pd.Series.value_counts).fillna(0)
    summary.to_csv(os.path.join(OUTPUT_DIR, "feedback_summary.csv"))
    summary.plot(kind="bar", figsize=(10, 6))
    plt.title("Feedback Distribution")
    plt.ylabel("Count")
    plt.xlabel("Feedback Options")
    plt.savefig(os.path.join(OUTPUT_DIR, "feedback_distribution.png"))
    plt.close()

feedback_summary(data)

# 3. Statistical Comparisons
def compare_confidence(df):
    models = df["model_name"].unique()
    confidence_data = [df[df["model_name"] == model]["confidence"].dropna() for model in models]
    f_stat, p_val = f_oneway(*confidence_data)
    with open(os.path.join(OUTPUT_DIR, "confidence_comparison.txt"), "w") as f:
        f.write(f"ANOVA F-statistic: {f_stat:.2f}, P-value: {p_val:.3f}\n")
    sns.boxplot(x="model_name", y="confidence", data=df)
    plt.title("Confidence Levels Across Models")
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_comparison.png"))
    plt.close()

compare_confidence(data)

# 4. Error Analysis
def error_analysis(df):
    incorrect = df[df["intended_digit"] != df["predicted_digit"]]
    error_counts = incorrect.groupby(["intended_digit", "predicted_digit"]).size().unstack(fill_value=0)
    sns.heatmap(error_counts, annot=True, fmt="d", cmap="Reds", cbar=False)
    plt.title("Error Patterns: Intended vs Predicted Digits")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Intended Digit")
    plt.savefig(os.path.join(OUTPUT_DIR, "error_analysis.png"))
    plt.close()

error_analysis(data)

# 5. Subject-wise Variability
def subject_variability(df):
    subject_accuracy = df.groupby("subject_number")[["intended_digit", "predicted_digit"]].apply(lambda x: (x["intended_digit"] == x["predicted_digit"]).mean())
    subject_accuracy.to_csv(os.path.join(OUTPUT_DIR, "subject_accuracy.csv"))
    subject_accuracy.plot(kind="bar", figsize=(12, 6))
    plt.title("Subject-wise Prediction Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Subject Number")
    plt.savefig(os.path.join(OUTPUT_DIR, "subject_accuracy.png"))
    plt.close()

subject_variability(data)

print(f"All results saved in {OUTPUT_DIR}")