# config.py

# Paths
BASE_RESULTS_DIR = "../../exp_results/"
TRAINED_MODELS_DIR = "../../trained_models/"

# Model Paths
BASE_MODEL_PATH = f"{TRAINED_MODELS_DIR}base_model.keras"
MC_DROPOUT_MODEL_PATH = f"{TRAINED_MODELS_DIR}dropout_model.keras"
ENSEMBLE_MODEL_PREFIX = f"{TRAINED_MODELS_DIR}ensemble_model"
ENSEMBLE_SIZE = 5

# Experiment Settings
MAX_DRAW_PER_DIGIT = 2  # Number of times each digit is drawn
TOTAL_DIGITS = 10        # Digits from 0 to 9