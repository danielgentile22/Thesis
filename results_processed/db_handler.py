import os
import sqlite3
import argparse

BASE_DIR = "../results_raw"
DB_NAME = "experiment_results.db"

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Create a single-table SQLite database from experiment results.")
parser.add_argument("--no-practice", action="store_true", help="Exclude practice data from the dataset.")
parser.add_argument("--include-only-digits", nargs='*', type=int, default=None,
                    help="Include only the specified digits (e.g. --include-only-digits 0 1 2). If not provided, include all digits.")
args = parser.parse_args()

INCLUDE_PRACTICE = not args.no_practice
INCLUDE_ONLY_DIGITS = args.include_only_digits

# Create/connect to the SQLite database
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Create single table (if it doesn't exist)
cursor.execute("""
CREATE TABLE IF NOT EXISTS data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_number TEXT,
    run_type TEXT,
    intended_digit INTEGER,
    q1_answer TEXT,
    q2_answer TEXT,
    q3_answer TEXT,
    q4_answer TEXT,
    q5_answer TEXT,
    model_name TEXT,
    predicted_digit INTEGER,
    confidence REAL,
    original_drawing BLOB,
    processed_drawing BLOB,
    probabilities_plot BLOB
);
""")
conn.commit()

def get_subject_number(subject_path):
    """Read subject_info.txt to get subject number."""
    info_path = os.path.join(subject_path, 'subject_info.txt')
    subject_number = None
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                if line.startswith("Subject Number:"):
                    subject_number = line.split(':', 1)[1].strip()
                    # Format single-digit subject numbers as two digits
                    if len(subject_number) == 1:
                        subject_number = f"0{subject_number}"
    return subject_number

def parse_prediction_file(prediction_file_path):
    """
    Parse prediction.txt for intended digit, predictions, and feedback.
    Extracts Q1-Q5 answers and normalizes Likert-scale responses.
    """
    intended_digit = None
    predictions = []
    feedback_answers = {"q1": None, "q2": None, "q3": None, "q4": None, "q5": None}

    if not os.path.exists(prediction_file_path):
        return intended_digit, predictions, feedback_answers

    with open(prediction_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # Define valid Likert-scale answers
    likert_scale = {
        "strongly disagree": "Strongly disagree",
        "disagree": "Disagree",
        "neutral": "Neutral",
        "agree": "Agree",
        "strongly agree": "Strongly agree",
        "can't answer": "Can't answer",
        "other": "Can't answer"
    }

    for line in lines:
        # Intended Digit
        if line.startswith("Intended Digit:"):
            intended_digit = int(line.split(":", 1)[1].strip())

        # Predictions
        elif line.startswith("Model Used:"):
            model_name = line.split(":", 1)[1].strip()
        elif line.startswith("Predicted Digit:"):
            predicted_digit = int(line.split(":", 1)[1].strip())
        elif line.startswith("Confidence:"):
            confidence = float(line.split(":", 1)[1].strip().replace("%", ""))
            predictions.append((model_name, predicted_digit, confidence))

        # Feedback (Q1-Q5 answers)
        elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
            # Extract question number and answer
            parts = line.split(".", 1)
            if len(parts) == 2:
                question_number = f"q{parts[0].strip()}"
                raw_feedback = parts[1].strip()

                # Separate the answer from the question (if present)
                if "?" in raw_feedback:
                    answer = raw_feedback.split("?", 1)[1].strip()
                else:
                    answer = raw_feedback

                # Normalize the answer to match the Likert scale
                normalized_answer = likert_scale.get(answer.lower(), "Can't answer")
                feedback_answers[question_number] = normalized_answer

    return intended_digit, predictions, feedback_answers

def find_probabilities_plots(draw_folder):
    """Find probability plots for a given drawing folder."""
    plots = []
    if os.path.isdir(draw_folder):
        for f in os.listdir(draw_folder):
            if f.startswith("probabilities_plot_") and f.endswith(".png"):
                plots.append(os.path.join(draw_folder, f))
    return plots

def load_image_as_blob(filepath):
    """Load an image file as binary data."""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return f.read()
    return None

def check_duplicates(subject_number, run_type, intended_digit):
    """Check if the data already exists in the database to avoid duplicates."""
    cursor.execute("""
        SELECT COUNT(*) FROM data
        WHERE subject_number = ? AND run_type = ? AND intended_digit = ?
    """, (subject_number, run_type, intended_digit))
    count = cursor.fetchone()[0]
    return count > 0

def insert_data(subject_number, run_type, intended_digit, feedback_answers, predictions_list, original_path, processed_path, plots):
    """Insert data into the database."""
    if INCLUDE_ONLY_DIGITS is not None and intended_digit not in INCLUDE_ONLY_DIGITS:
        return

    if check_duplicates(subject_number, run_type, intended_digit):
        print(f"Skipping duplicate entry: Subject {subject_number}, Run Type {run_type}, Digit {intended_digit}")
        return

    for model_name, predicted_digit, confidence in predictions_list:
        original_blob = load_image_as_blob(original_path)
        processed_blob = load_image_as_blob(processed_path)
        plot_blob = None
        for plot in plots:
            if model_name.replace(" ", "_") in plot:
                plot_blob = load_image_as_blob(plot)
                break
        cursor.execute("""
            INSERT INTO data (
                subject_number, run_type, intended_digit,
                q1_answer, q2_answer, q3_answer, q4_answer, q5_answer,
                model_name, predicted_digit, confidence,
                original_drawing, processed_drawing, probabilities_plot
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subject_number, run_type, intended_digit,
            feedback_answers["q1"], feedback_answers["q2"], feedback_answers["q3"], feedback_answers["q4"], feedback_answers["q5"],
            model_name, predicted_digit, confidence,
            original_blob, processed_blob, plot_blob
        ))

def process_drawings(subject_number, run_type, path_to_digits):
    """Process all digit directories in the given path."""
    if not os.path.isdir(path_to_digits):
        return
    for digit_dir in sorted(os.listdir(path_to_digits)):
        if digit_dir.startswith("digit_"):
            digit_num = int(digit_dir.split("_")[1])
            digit_path = os.path.join(path_to_digits, digit_dir)
            for draw_dir in sorted(os.listdir(digit_path)):
                if draw_dir.startswith("draw_"):
                    draw_path = os.path.join(digit_path, draw_dir)
                    prediction_file_path = os.path.join(draw_path, "prediction.txt")
                    intended_digit, predictions_list, feedback_answers = parse_prediction_file(prediction_file_path)
                    original_path = os.path.join(draw_path, "original_drawing.png")
                    processed_path = os.path.join(draw_path, "processed_drawing.png")
                    plots = find_probabilities_plots(draw_path)
                    insert_data(subject_number, run_type, intended_digit, feedback_answers, predictions_list, original_path, processed_path, plots)

def process_subject(subject_path):
    """Process data for a single subject."""
    subject_number = get_subject_number(subject_path)
    if subject_number is None:
        return

    if "pilot" in subject_path.lower():
        print(f"Skipping Pilot folder: {subject_path}")
        return

    if INCLUDE_PRACTICE:
        practice_path = os.path.join(subject_path, "Practice")
        process_drawings(subject_number, "practice", practice_path)

    process_drawings(subject_number, "main", subject_path)
    conn.commit()

def main():
    """Main function to process all subjects."""
    for d in os.listdir(BASE_DIR):
        if d.startswith("Subject_"):
            subject_path = os.path.join(BASE_DIR, d)
            if os.path.isdir(subject_path):
                process_subject(subject_path)

    print("Database creation and data insertion completed successfully.")

if __name__ == "__main__":
    main()
