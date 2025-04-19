# 5-bayesian_acc_vs_conf_analysis.py

import os
import sqlite3
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde

DB_PATH = "../../results_processed/experiment_results.db"
OUTPUT_DIR = "../results/5-bayesian_acc_vs_conf"
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

def run_bayes(diff):
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=20)
        sigma = pm.HalfNormal("sigma", sigma=10)
        d_obs = pm.Normal("d_obs", mu=mu, sigma=sigma, observed=diff)
        trace = pm.sample(2000, tune=1000, return_inferencedata=True,
                          target_accept=0.95, progressbar=True)
    prior0 = norm.pdf(0, loc=0, scale=20)
    mu_samples = trace.posterior["mu"].values.flatten()
    kde = gaussian_kde(mu_samples)
    post0 = kde.evaluate(0)[0]
    bf10 = prior0 / post0
    return trace, bf10, mu_samples

# find missing subj
def debug_missing_rows(merged_df, model_name="Ensemble Model"):
    subset = merged_df[merged_df["model_name"] == model_name]
    missing = subset[subset[["mean_accuracy", "mean_confidence"]].isnull().any(axis=1)]
    print("\nMissing accuracy or confidence values for model:", model_name)
    print(missing[["subject_number", "mean_accuracy", "mean_confidence"]])
    
def check_participant_counts(df):
    print("\nParticipants per model BEFORE aggregation:")
    model_counts = df.groupby("model_name")["subject_number"].nunique()
    print(model_counts)

def bayes_test(merged):
    results = []
    for m in merged["model_name"].unique():
        sub = merged[merged["model_name"] == m].dropna(subset=["mean_accuracy", "mean_confidence"])
        if len(sub) < 3:
            print(f"Skipping {m} - not enough data.")
            continue
        diff = sub["mean_accuracy"].values - sub["mean_confidence"].values
        print(f"Running Bayesian test for {m} (n={len(diff)})")
        trace, bf10, mu_samples = run_bayes(diff)
        plt.figure(figsize=(6,4))
        az.plot_posterior(trace, var_names=["mu"], hdi_prob=0.95)
        plt.title(f"Posterior of μ for {m}")
        plt.savefig(os.path.join(OUTPUT_DIR, f"posterior_mu_{m.replace(' ', '_')}.png"))
        plt.close()
        results.append({
            "model": m,
            "n": len(sub),
            "BF10": bf10,
            "posterior_mu_mean": np.mean(mu_samples),
            "HDI": az.hdi(trace, var_names=["mu"])["mu"].to_dict()
        })
        
    print("\nParticipants per model AFTER merge:")
    print(merged.groupby("model_name")["subject_number"].nunique())

    debug_missing_rows(merged)
    
    return results

def main():
    df = load_data(DB_PATH)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    acc = agg_acc(df)
    conf = agg_conf(df)
    merged = pd.merge(acc, conf, on=["subject_number", "model_name"])
    merged["diff"] = merged["mean_accuracy"] - merged["mean_confidence"]
    res = bayes_test(merged)
    pd.DataFrame(res).to_csv(os.path.join(OUTPUT_DIR, "bayesian_results.csv"), index=False)
    print("Bayesian analysis complete. Results saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()