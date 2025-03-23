# 4-acc_vs_conf_analysis.py

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

DB_PATH = "../../results_processed/experiment_results.db"
OUTPUT_DIR = "../results/4-acc_vs_conf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM data", conn)
    conn.close()
    return df

def agg_acc(df):
    grp = df.groupby(["subject_number", "model_name"], group_keys=False).apply(
        lambda x: np.mean(x["intended_digit"] == x["predicted_digit"]) * 100
    )
    return grp.reset_index(name="mean_accuracy")

def agg_conf(df):
    grp = df.groupby(["subject_number", "model_name"])["confidence"].mean()
    return grp.reset_index(name="mean_confidence")

def wilcox_acc_conf(merged):
    res = []
    for m in merged["model_name"].unique():
        sub = merged[merged["model_name"] == m].dropna()
        acc = sub["mean_accuracy"].values
        conf = sub["mean_confidence"].values
        diff = acc - conf
        if len(diff) < 3:
            print(f"Skipping {m} - not enough data.")
            continue
        stat, p = wilcoxon(diff)
        print(f"{m}: Median acc: {np.median(acc):.2f}, Median conf: {np.median(conf):.2f}, p = {p:.5f}")
        res.append({"model": m, "n": len(diff), "statistic": stat, "p_value": p,
                    "median_accuracy": np.median(acc), "median_confidence": np.median(conf),
                    "mean_diff": np.mean(diff)})
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=acc, y=conf)
        mx = max(np.max(acc), np.max(conf)) * 1.05
        plt.plot([0, mx], [0, mx], "--", color="gray", label="x = y")
        plt.title(f"Confidence vs Accuracy: {m}")
        plt.xlabel("Mean Accuracy (%)")
        plt.ylabel("Mean Confidence (%)")
        plt.legend()
        plt.xlim(0,100)
        plt.ylim(0,100)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{m.replace(' ', '_')}.png"))
        plt.close()
    return pd.DataFrame(res)

def main():
    df = load_data(DB_PATH)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    acc_df = agg_acc(df)
    conf_df = agg_conf(df)
    merged = pd.merge(acc_df, conf_df, on=["subject_number", "model_name"])
    merged["diff"] = merged["mean_accuracy"] - merged["mean_confidence"]
    res = wilcox_acc_conf(merged)
    res.to_csv(os.path.join(OUTPUT_DIR, "wilcox_results.csv"), index=False)
    print("Acc vs Conf analysis complete. Results saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()