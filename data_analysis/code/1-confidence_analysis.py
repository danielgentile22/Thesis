# 1-confidence_analysis.py

import os
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

DB_PATH = "../../results_processed/experiment_results.db"
OUTPUT_DIR = "../results/1-confidence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM data", conn)
    conn.close()
    return df

def agg_conf(df, subj="subject_number", model="model_name", conf="confidence"):
    grp = df.groupby([subj, model])[conf].mean().reset_index()
    grp.rename(columns={conf: "mean_confidence"}, inplace=True)
    return grp

def friedman_test(df, subj="subject_number", model="model_name", col="mean_confidence"):
    piv = df.pivot(index=subj, columns=model, values=col)
    piv.dropna(inplace=True)
    arrs = [piv[m].values for m in piv.columns]
    stat, p = friedmanchisquare(*arrs)
    return stat, p, piv

def posthoc_wilcox(piv):
    cols = list(piv.columns)
    out = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            w, p_val = wilcoxon(piv[cols[i]], piv[cols[j]])
            out.append({"Comparison": f"{cols[i]} vs {cols[j]}", "Wilcoxon_p": p_val})
    ps = [x["Wilcoxon_p"] for x in out]
    _, corr, _, _ = multipletests(ps, method="bonferroni")
    for k, r in enumerate(out):
        r["Bonferroni_p"] = corr[k]
    return pd.DataFrame(out)

def boxplot_conf(df, model="model_name", col="mean_confidence"):
    plt.figure(figsize=(8,6))
    sns.boxplot(x=model, y=col, data=df)
    sns.stripplot(x=model, y=col, data=df, color="black", alpha=0.5, jitter=True)
    plt.title("Mean Confidence by Model (per Participant)")
    plt.ylabel("Confidence (%)")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_boxplot.png"))
    plt.close()

def bar_chart_conf(df, model="model_name", col="mean_confidence"):
    stats = df.groupby(model)[col].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(8,6))
    bars = plt.bar(stats[model], stats["mean"], yerr=stats["std"],
                   capsize=5, color="skyblue", edgecolor="black")
    for bar, m in zip(bars, stats["mean"]):
        plt.text(bar.get_x() + bar.get_width() / 2, m, f"{m:.2f}", ha="center", va="bottom")
    plt.title("Average Confidence by Model (Mean ± SD)")
    plt.ylabel("Confidence (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_barchart.png"))
    plt.close()

def main():
    df = load_data(DB_PATH)
    conf_df = agg_conf(df)
    boxplot_conf(conf_df)
    bar_chart_conf(conf_df)
    stat, p, piv = friedman_test(conf_df)
    print(f"Friedman stat: {stat:.3f}, p = {p:.5f}")
    with open(os.path.join(OUTPUT_DIR, "confidence_stats.txt"), "w") as f:
        f.write(f"Friedman stat: {stat:.3f}, p = {p:.5f}\n")
        if p < 0.05:
            post = posthoc_wilcox(piv)
            f.write(post.to_string(index=False))
        else:
            f.write("No significant difference.\n")
    print("Confidence analysis done. See", OUTPUT_DIR)

if __name__ == "__main__":
    main()