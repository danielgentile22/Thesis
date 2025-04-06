import os
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "../../results_processed/experiment_results.db"
OUTPUT_DIR = "../results/6-ca-counter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIKERT_COLS = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM data", conn)
    conn.close()
    return df

def count_cant_answers(df, cols, subj_col="subject_number"):
    # Count "Can't answer" per participant across all Likert questions
    ca_counts = df[cols].apply(lambda row: row.isin(["Can't answer"]), axis=1)
    df["cant_answer_total"] = ca_counts.sum(axis=1)
    summary = df.groupby(subj_col)["cant_answer_total"].sum().sort_index()

    return summary

def main():
    df = load_data(DB_PATH)
    output_path = os.path.join(OUTPUT_DIR, "cant_answer_summary.txt")

    # Count and report
    ca_summary = count_cant_answers(df, LIKERT_COLS)
    total_ca = int(ca_summary.sum())

    with open(output_path, "w") as f:
        f.write("Can't Answer Summary:\n")
        f.write("="*30 + "\n")
        for pid, count in ca_summary.items():
            f.write(f"Participant {pid}: {int(count)} CAs\n")
        f.write("\n")
        f.write(f"Total CAs: {total_ca}\n")

    print("Can't answer report complete. See", output_path)

if __name__ == "__main__":
    main()