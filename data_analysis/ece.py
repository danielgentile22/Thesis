import os
import numpy as np
import pandas as pd
import sqlite3
from tensorflow.keras.datasets import mnist # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Function to compute Expected Calibration Error (ECE)
def compute_ece(probabilities, labels, num_bins=10):
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Get predictions in this bin
        bin_indices = (probabilities >= bin_lower) & (probabilities < bin_upper)
        bin_confidences = probabilities[bin_indices]
        bin_labels = labels[bin_indices]

        if len(bin_confidences) > 0:
            # Average confidence and accuracy in the bin
            avg_confidence = np.mean(bin_confidences)
            accuracy = np.mean(bin_labels)

            # Weighted difference
            ece += np.abs(avg_confidence - accuracy) * len(bin_confidences) / len(probabilities)

    return ece

# Path to database and output directory
DB_PATH = "../results_processed/experiment_results.db"
OUTPUT_DIR = "ece_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data from SQLite database
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT confidence, intended_digit, predicted_digit, run_type FROM data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Load MNIST test data and pass it through the model
def process_mnist_test_data(model_path):
    # Load MNIST test dataset
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0  # Normalize
    x_test = np.expand_dims(x_test, axis=-1)  # Add channel dimension for CNNs

    # Load the trained model
    model = load_model(model_path)

    # Get predictions
    predictions = model.predict(x_test)
    predicted_digits = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    return confidences, (predicted_digits == y_test).astype(int)

# Load human data from the database
data = load_data(DB_PATH)

# Check for required columns
required_columns = {"confidence", "intended_digit", "predicted_digit", "run_type"}
if not required_columns.issubset(data.columns):
    raise ValueError(f"Dataset must contain these columns: {required_columns}")

# Prepare data for human-generated digits
human_data = data[data["run_type"] == "practice"]
human_probs = human_data["confidence"].values
human_labels = (human_data["intended_digit"] == human_data["predicted_digit"]).astype(int).values

# Prepare data for MNIST test data
# MODEL_PATH = "../trained_models/base_model.keras"  # Base model
MODEL_PATH = "../trained_models/dropout_model.keras"  # MC Dropout model
mnist_probs, mnist_labels = process_mnist_test_data(MODEL_PATH)

# Compute ECE
human_ece = compute_ece(human_probs, human_labels)
mnist_ece = compute_ece(mnist_probs, mnist_labels)

# Save results
with open(os.path.join(OUTPUT_DIR, "ece_results.txt"), "w") as f:
    f.write(f"ECE for Human-Generated Digits (Practice): {human_ece:.4f}\n")
    f.write(f"ECE for MNIST Test Data: {mnist_ece:.4f}\n")

print(f"ECE for Human-Generated Digits (Practice): {human_ece:.4f}")
print(f"ECE for MNIST Test Data: {mnist_ece:.4f}")
