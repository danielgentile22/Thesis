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
MAX_DRAW_PER_DIGIT = 1   # Number of times each digit is drawn
TOTAL_DIGITS = 10         # Digits from 0 to 9
NUM_PRACTICE_DIGITS = 3   # Number of practice digits

# Brush and Canvas Settings
BRUSH_DEFAULT_SIZE = 25
BRUSH_COLORS = ["black"]
BRUSH_DEFAULT_COLOR = "black"
BRUSH_COLOR_MODE = "fixed"

CANVAS_HEIGHT = 700
CANVAS_WIDTH = 800
CANVAS_SIZE = [1600, 1200]

# Image Processing Settings
IMAGE_SIZE = (28, 28)     # Image size for preprocessing
IMAGE_THRESHOLD = 20      # Threshold for binarizing the image