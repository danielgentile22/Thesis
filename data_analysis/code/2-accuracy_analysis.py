# 2-accuracy_analysis.py

import os
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

DB_PATH = "../../results_processed/experiment_results.db"
OUTPUT_DIR = "../results/2-accuracy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM data"  # Adjust table name/columns if needed
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def agg_acc(df, subj="subject_number", model="model_name", true="intended_digit", pred="predicted_digit"):
    # Multiply by 100 so that accuracy is expressed as a percentage (0-100)
    def acc(x):
        return np.mean(x[true] == x[pred]) * 100
    grp = df.groupby([subj, model], group_keys=False).apply(acc).reset_index(name="mean_accuracy")
    return grp

def friedman_acc(df, subj="subject_number", model="model_name", col="mean_accuracy"):
    piv = df.pivot(index=subj, columns=model, values=col)
    piv.dropna(inplace=True)
    arrs = [piv[m].values for m in piv.columns]
    stat, p = friedmanchisquare(*arrs)
    return stat, p, piv

def posthoc_wilcox_acc(piv):
    cols = list(piv.columns)
    res = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            w, p_val = wilcoxon(piv[cols[i]], piv[cols[j]])
            res.append({"Comparison": f"{cols[i]} vs {cols[j]}", "Wilcoxon_p": p_val})
    ps = [x["Wilcoxon_p"] for x in res]
    _, corr, _, _ = multipletests(ps, method="bonferroni")
    for k, r in enumerate(res):
        r["Bonferroni_p"] = corr[k]
    return pd.DataFrame(res)

def boxplot_acc(df, model="model_name", col="mean_accuracy"):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=model, y=col, data=df)
    sns.stripplot(x=model, y=col, data=df, color='black', alpha=0.5, jitter=True)
    plt.title("Mean Accuracy by Model (per Participant)")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Model")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_boxplot.png"))
    plt.close()

def plot_bar_chart(df, model="model_name", col="mean_accuracy"):
    stats_df = df.groupby(model)[col].agg(["mean", "sem"]).reset_index()
    plt.figure(figsize=(8,6))
    bars = plt.bar(stats_df[model],
                   stats_df["mean"],
                   yerr=stats_df["sem"],
                   capsize=5,
                   color="lightsalmon",
                   edgecolor="black")
    for bar, m in zip(bars, stats_df["mean"]):
        # m is already a percentage
        plt.text(bar.get_x() + bar.get_width()/2, m, f"{m:.1f}%", ha="center", va="bottom")
    plt.title("Average Accuracy by Model (Mean ± SEM)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_barchart.png"))
    plt.close()

def main():
    df = load_data(DB_PATH)
    agg = agg_acc(df)
    boxplot_acc(agg, "model_name", "mean_accuracy")
    plot_bar_chart(agg, "model_name", "mean_accuracy")
    stat, p, piv = friedman_acc(agg)
    print(f"Friedman stat: {stat:.3f}, p = {p:.5f}")
    with open(os.path.join(OUTPUT_DIR, "accuracy_stats.txt"), "w") as f:
        f.write(f"Friedman stat: {stat:.3f}, p = {p:.5f}\n")
        if p < 0.05:
            post = posthoc_wilcox_acc(piv)
            f.write(post.to_string(index=False))
        else:
            f.write("No significant difference.\n")
        
    report_model_accuracies(agg, os.path.join(OUTPUT_DIR, "accuracy_stats.txt"))
            
    print("Accuracy analysis done. See", OUTPUT_DIR)
    
def report_model_accuracies(df, output_path=None):
    means = df.groupby("model_name")["mean_accuracy"].mean().round(2)
    output = "\nMean Accuracies by Model:\n"
    for model, acc in means.items():
        output += f"{model}: {acc:.2f}%\n"

    print(output.strip())

    if output_path:
        with open(output_path, "a") as f:
            f.write(output)


if __name__ == "__main__":
    main()
