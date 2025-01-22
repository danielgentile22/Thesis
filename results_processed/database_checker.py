import sqlite3
from collections import Counter

DB_NAME = "experiment_results.db"

# Connect to the database
def connect_to_db(db_name):
    return sqlite3.connect(db_name)

def print_general_info():
    conn = connect_to_db(DB_NAME)
    cursor = conn.cursor()

    print("Database General Information:")

    # Total entries in the database
    cursor.execute("SELECT COUNT(*) FROM data")
    total_entries = cursor.fetchone()[0]
    print(f"Total Entries: {total_entries}")

    # Entries per subject number
    cursor.execute("SELECT subject_number, COUNT(*) FROM data GROUP BY subject_number")
    entries_per_subject = cursor.fetchall()
    print("Entries Per Subject:")
    for subject, count in entries_per_subject:
        print(f"  Subject {subject}: {count} entries")

    # Total number of times each question is answered
    question_columns = ["q1_answer", "q2_answer", "q3_answer", "q4_answer", "q5_answer"]
    for question in question_columns:
        cursor.execute(f"SELECT {question} FROM data")
        answers = cursor.fetchall()
        answer_counts = Counter(answer[0] for answer in answers if answer[0] is not None)
        total_answers = sum(answer_counts.values())
        print(f"{question} Answer Counts:")
        for answer, count in answer_counts.items():
            print(f"  {answer}: {count}")
        print(f"  Total Answers for {question}: {total_answers}")

    conn.close()

if __name__ == "__main__":
    print_general_info()