import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from fpdf import FPDF
import os

DB_PATH = "../results_processed/experiment_results.db"
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data from SQLite database
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

data = load_data(DB_PATH)

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

# 3. Confidence Analysis with ANOVA and Post-hoc Bonferroni Correction
def confidence_analysis(df):
    models = df["model_name"].unique()
    confidence_data = [df[df["model_name"] == model]["confidence"].dropna() for model in models]
    f_stat, p_val = f_oneway(*confidence_data)
    
    # Perform pairwise comparisons
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

    # Apply Bonferroni correction
    _, corrected_pvals, _, _ = multipletests(pairwise_pvals, method="bonferroni")
    pairwise_results = pd.DataFrame({
        "Model Pair": model_pairs,
        "Corrected P-Value": corrected_pvals
    })
    pairwise_results.to_csv(os.path.join(OUTPUT_DIR, "pairwise_comparisons.csv"), index=False)

    # Save ANOVA results
    with open(os.path.join(OUTPUT_DIR, "confidence_comparison.txt"), "w") as f:
        f.write(f"ANOVA F-statistic: {f_stat:.2f}, P-value: {p_val:.3f}\n")
    sns.boxplot(x="model_name", y="confidence", data=df)
    plt.title("Confidence Levels Across Models")
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_comparison.png"))
    plt.close()

confidence_analysis(data)

# 4. Error Patterns
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
    subject_accuracy = df.groupby("subject_number")[["intended_digit", "predicted_digit"]]
    subject_accuracy = subject_accuracy.apply(lambda x: (x["intended_digit"] == x["predicted_digit"]).mean())
    subject_accuracy.to_csv(os.path.join(OUTPUT_DIR, "subject_accuracy.csv"))
    subject_accuracy.plot(kind="bar", figsize=(12, 6))
    plt.title("Subject-wise Prediction Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Subject Number")
    plt.savefig(os.path.join(OUTPUT_DIR, "subject_accuracy.png"))
    plt.close()

subject_variability(data)

# 6. Create PDF Report
def create_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(0, 10, "Experiment Analysis Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "1. Accuracy Comparison", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Accuracy comparison of models based on prediction correctness.")
    pdf.ln(5)

    # Read and add CSV content
    pdf.set_font("Courier", size=10)
    with open(os.path.join(OUTPUT_DIR, "model_accuracies.csv"), "r") as f:
        for line in f:
            pdf.cell(0, 10, line.strip(), ln=True)
    pdf.ln(10)

    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "2. Confusion Matrices", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Confusion matrices show prediction accuracy and misclassifications.")
    pdf.ln(5)

    for model in data["model_name"].unique():
        pdf.cell(0, 10, f"Confusion Matrix: {model}", ln=True)
        pdf.image(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model.replace(' ', '_')}.png"), w=160)
        pdf.ln(5)

    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "3. Feedback Distribution", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Feedback ratings across models.")
    pdf.image(os.path.join(OUTPUT_DIR, "feedback_distribution.png"), w=160)

    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "4. Confidence Analysis", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Analysis of confidence levels with ANOVA and pairwise comparisons.")
    pdf.image(os.path.join(OUTPUT_DIR, "confidence_comparison.png"), w=160)
    pdf.add_page()
    pdf.set_font("Courier", size=10)
    with open(os.path.join(OUTPUT_DIR, "confidence_comparison.txt"), "r") as f:
        for line in f:
            pdf.cell(0, 10, line.strip(), ln=True)
    pdf.add_page()
    pdf.set_font("Courier", size=10)
    pdf.cell(0, 10, "Pairwise Comparisons:", ln=True)
    with open(os.path.join(OUTPUT_DIR, "pairwise_comparisons.csv"), "r") as f:
        for line in f:
            pdf.cell(0, 10, line.strip(), ln=True)

    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "5. Error Patterns", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Heatmap of common errors in predictions.")
    pdf.image(os.path.join(OUTPUT_DIR, "error_analysis.png"), w=160)

    pdf.add_page()
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(0, 10, "6. Subject-wise Prediction Accuracy", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "Variability in accuracy across participants.")
    pdf.image(os.path.join(OUTPUT_DIR, "subject_accuracy.png"), w=160)

    output_path = os.path.join(OUTPUT_DIR, "Experiment_Analysis_Report.pdf")
    pdf.output(output_path)
    print(f"PDF Report saved at {output_path}")

create_pdf()

print(f"All results saved in {OUTPUT_DIR}")