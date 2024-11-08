# models.py

import os
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Configuration variables for model paths
BASE_MODEL_PATH = "../../trained_models/base_model.keras"
MC_DROPOUT_MODEL_PATH = "../../trained_models/dropout_model.keras"
ENSEMBLE_MODEL_PREFIX = "../../trained_models/ensemble_model"
ENSEMBLE_SIZE = 5

def load_base_model(model_path=BASE_MODEL_PATH):
    """Load the base CNN model."""
    print("Loading Base model...")
    model = load_model(model_path)
    print("Base model loaded.")
    return model

def load_mc_dropout_model(model_path=MC_DROPOUT_MODEL_PATH):
    """Load the MC-Dropout model."""
    print("Loading MC-Dropout model...")
    model = load_model(model_path)
    print("MC-Dropout model loaded.")
    return model

def load_ensemble_models(model_path_prefix=ENSEMBLE_MODEL_PREFIX, ensemble_size=ENSEMBLE_SIZE):
    """Load all models in the ensemble."""
    print("Loading Ensemble models...")
    ensemble = []
    for i in range(ensemble_size):
        model_path = f"{model_path_prefix}_{i+1}.keras"
        model = load_model(model_path)
        ensemble.append(model)
    print("Ensemble models loaded.")
    return ensemble

def predict_with_base_model(model, x):
    """Predict using the base model."""
    probabilities = model.predict(x)
    predicted_labels = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1) * 100  # Convert to percentage
    return predicted_labels, confidence, probabilities

def predict_with_mc_dropout(model, x, num_samples=100):
    """Predict using the MC-Dropout model with uncertainty estimation."""
    predictions = np.stack([model(x, training=True) for _ in range(num_samples)])
    mean_predictions = np.mean(predictions, axis=0)
    confidence = np.max(mean_predictions, axis=1) * 100
    predicted_labels = np.argmax(mean_predictions, axis=1)
    return predicted_labels, confidence, mean_predictions

def predict_with_ensemble(ensemble, x):
    """Predict using the ensemble of models."""
    ensemble_predictions = [model.predict(x) for model in ensemble]
    mean_predictions = np.mean(ensemble_predictions, axis=0)
    predicted_labels = np.argmax(mean_predictions, axis=1)
    confidence = np.max(mean_predictions, axis=1) * 100
    return predicted_labels, confidence, mean_predictions