# 3-likert_analysis.py

import os
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

DB_PATH = "../../results_processed/experiment_results.db"
OUTPUT_DIR = "../results/3-likert"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIKERT_COLS = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM data", conn)
    conn.close()
    return df

def agg_lik(df, subj="subject_number", model="model_name", ques="q1_answer"):
    grp = df.groupby([subj, model])[ques].mean().reset_index(name="mean_rating")
    return grp

def friedman_lik(df, subj="subject_number", model="model_name", col="mean_rating"):
    piv = df.pivot(index=subj, columns=model, values=col)
    piv.dropna(inplace=True)
    arrs = [piv[m].values for m in piv.columns]
    stat, p = friedmanchisquare(*arrs)
    return stat, p, piv

def posthoc_wilcox_lik(piv):
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

def boxplot_lik(df, model="model_name", col="mean_rating", ques="q1_answer"):
    plt.figure(figsize=(8,6))
    sns.boxplot(x=model, y=col, data=df)
    sns.stripplot(x=model, y=col, data=df, color="black", alpha=0.5, jitter=True)
    plt.title(f"{ques} Ratings by Model")
    plt.xlabel("Model")
    plt.ylabel("Rating (1=Strongly Disagree, 5=Strongly Agree)")
    plt.ylim(0.5, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{ques}.png"))
    plt.close()

def main():
    df = load_data(DB_PATH)
    subj = "subject_number"
    model = "model_name"
    mapping = {
        "Strongly disagree": 1,
        "Disagree": 2,
        "Neutral": 3,
        "Agree": 4,
        "Strongly agree": 5,
        "Can't answer": np.nan
    }
    for col in LIKERT_COLS:
        df[col] = df[col].map(mapping)
    
    out_file = os.path.join(OUTPUT_DIR, "likert_stats.txt")
    with open(out_file, "w") as f:
        f.write("Likert Scale Analysis Results\n")
        f.write("="*40 + "\n\n")
        for ques in LIKERT_COLS:
            agg = agg_lik(df, subj, model, ques)
            boxplot_lik(agg, model, "mean_rating", ques)
            stat, p, piv = friedman_lik(agg, subj, model, "mean_rating")
            f.write(f"Question: {ques}\n")
            f.write(f"  Friedman stat: {stat:.3f}, p = {p:.5f}\n")
            if p < 0.05:
                post = posthoc_wilcox_lik(piv)
                f.write(post.to_string(index=False) + "\n\n")
            else:
                f.write("  No significant difference.\n\n")
    print("Likert analysis complete. See", OUTPUT_DIR)

if __name__ == "__main__":
    main()