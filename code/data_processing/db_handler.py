import os
import sqlite3
import argparse

BASE_DIR = "../../exp_results"  # Adjust if needed
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
    subject_name TEXT,
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

def get_subject_info(subject_path):
    """Read subject_info.txt to get subject number and name."""
    info_path = os.path.join(subject_path, 'subject_info.txt')
    subject_number = None
    subject_name = None
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                if line.startswith("Subject Number:"):
                    subject_number = line.split(':', 1)[1].strip()
                elif line.startswith("Name:"):
                    subject_name = line.split(':', 1)[1].strip()
    return subject_number, subject_name

def parse_prediction_file(prediction_file_path):
    """
    Parse prediction.txt for intended digit, predictions, and feedback.
    predictions_list will be [(model_name, predicted_digit, confidence), ...]
    """
    intended_digit = None
    model_used = None
    predictions = []  # For multiple models scenario if "All Models" is used
    feedback_answers = {"q1": None, "q2": None, "q3": None, "q4": None, "q5": None}

    if not os.path.exists(prediction_file_path):
        return intended_digit, predictions, feedback_answers

    with open(prediction_file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    mode = "predictions"
    current_model = None
    temp_predicted_digit = None
    temp_confidence = None

    for line in lines:
        if line.startswith("Intended Digit:"):
            intended_digit = int(line.split(":", 1)[1].strip())
        elif line.startswith("Model Used:"):
            model_used = line.split(":", 1)[1].strip()
            current_model = model_used
        elif line.startswith("Predicted Digit:"):
            temp_predicted_digit = int(line.split(":", 1)[1].strip())
        elif line.startswith("Confidence:"):
            conf_str = line.split(":", 1)[1].strip().replace("%", "")
            temp_confidence = float(conf_str)
            # If single model, finalize prediction now
            if model_used != "All Models":
                predictions.append((current_model, temp_predicted_digit, temp_confidence))
                temp_predicted_digit = None
                temp_confidence = None
        elif line.startswith("Feedback:"):
            mode = "feedback"
        elif mode == "predictions" and line.endswith(":") and model_used == "All Models":
            # This indicates a new model block for "All Models"
            current_model = line.replace(":", "").strip()
        elif mode == "predictions" and model_used == "All Models":
            # If "All Models" mode
            if line.startswith("Predicted Digit:"):
                temp_predicted_digit = int(line.split(":", 1)[1].strip())
            elif line.startswith("Confidence:"):
                conf_str = line.split(":", 1)[1].strip().replace("%", "")
                temp_confidence = float(conf_str)
                predictions.append((current_model, temp_predicted_digit, temp_confidence))
                temp_predicted_digit = None
                temp_confidence = None
        elif mode == "feedback":
            # Parse feedback lines
            # Example: "1. Is the top prediction appropriate? Strongly agree"
            if line.startswith("1."):
                feedback_answers["q1"] = line.split("?")[1].strip()
            elif line.startswith("2."):
                feedback_answers["q2"] = line.split("?")[1].strip()
            elif line.startswith("3."):
                feedback_answers["q3"] = line.split("?")[1].strip()
            elif line.startswith("4."):
                feedback_answers["q4"] = line.split("?")[1].strip()
            elif line.startswith("5."):
                feedback_answers["q5"] = line.split("?")[1].strip()

    return intended_digit, predictions, feedback_answers

def find_probabilities_plots(draw_folder):
    plots = []
    if os.path.isdir(draw_folder):
        for f in os.listdir(draw_folder):
            if f.startswith("probabilities_plot_") and f.endswith(".png"):
                plots.append(os.path.join(draw_folder, f))
    return plots

def load_image_as_blob(filepath):
    if filepath and os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return f.read()
    return None

def insert_data(subject_number, subject_name, run_type, intended_digit, feedback_answers, predictions_list, original_path, processed_path, plots):
    # Apply digit filter if specified
    if INCLUDE_ONLY_DIGITS is not None and intended_digit not in INCLUDE_ONLY_DIGITS:
        return  # Skip insertion if digit not in the allowed list

    # Insert one row per prediction
    if predictions_list:
        for model_name, pred_digit, conf in predictions_list:
            # Try to find a matching plot for the model (by model name in filename)
            model_plot_path = None
            for p in plots:
                if model_name.replace(" ", "_") in os.path.basename(p):
                    model_plot_path = p
                    break
            # If not found, just pick first plot
            if model_plot_path is None and plots:
                model_plot_path = plots[0]

            # Load images as binary
            original_blob = load_image_as_blob(original_path)
            processed_blob = load_image_as_blob(processed_path)
            plot_blob = load_image_as_blob(model_plot_path)

            cursor.execute("""
                INSERT INTO data (
                    subject_number, subject_name, run_type, intended_digit, 
                    q1_answer, q2_answer, q3_answer, q4_answer, q5_answer,
                    model_name, predicted_digit, confidence,
                    original_drawing, processed_drawing, probabilities_plot
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                subject_number, subject_name, run_type, intended_digit,
                feedback_answers["q1"], feedback_answers["q2"], feedback_answers["q3"], feedback_answers["q4"], feedback_answers["q5"],
                model_name, pred_digit, conf,
                original_blob, processed_blob, plot_blob
            ))
    else:
        # If there were no predictions (which would be odd), insert a row with null model info
        original_blob = load_image_as_blob(original_path)
        processed_blob = load_image_as_blob(processed_path)
        # No predictions means no plot either
        cursor.execute("""
            INSERT INTO data (
                subject_number, subject_name, run_type, intended_digit, 
                q1_answer, q2_answer, q3_answer, q4_answer, q5_answer,
                model_name, predicted_digit, confidence,
                original_drawing, processed_drawing, probabilities_plot
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            subject_number, subject_name, run_type, intended_digit,
            feedback_answers["q1"], feedback_answers["q2"], feedback_answers["q3"], feedback_answers["q4"], feedback_answers["q5"],
            None, None, None,
            original_blob, processed_blob, None
        ))

def process_drawings(subject_number, subject_name, run_type, path_to_digits):
    """Process all digit directories (and their draws) in the given path."""
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
                    if intended_digit is None:
                        intended_digit = digit_num
                    original_path = os.path.join(draw_path, "original_drawing.png") if os.path.exists(os.path.join(draw_path, "original_drawing.png")) else None
                    processed_path = os.path.join(draw_path, "processed_drawing.png") if os.path.exists(os.path.join(draw_path, "processed_drawing.png")) else None
                    plots = find_probabilities_plots(draw_path)
                    insert_data(subject_number, subject_name, run_type, intended_digit, feedback_answers, predictions_list, original_path, processed_path, plots)

def process_subject(subject_path):
    subject_number, subject_name = get_subject_info(subject_path)
    if subject_number is None:
        return

    # If practice data is included, process it
    if INCLUDE_PRACTICE:
        practice_path = os.path.join(subject_path, "Practice")
        process_drawings(subject_number, subject_name, "practice", practice_path)

    # Main experiment digits
    process_drawings(subject_number, subject_name, "main", subject_path)
    conn.commit()

def main():
    # Find subject directories named "Subject_X"
    for d in os.listdir(BASE_DIR):
        if d.startswith("Subject_"):
            subject_path = os.path.join(BASE_DIR, d)
            if os.path.isdir(subject_path):
                process_subject(subject_path)

    print("Database creation and data insertion completed successfully.")

if __name__ == "__main__":
    main()