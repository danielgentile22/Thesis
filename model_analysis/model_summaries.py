#!/usr/bin/env python3

"""
check_model_summaries.py

Script to load each trained model (.keras file) and display a summary
of its architecture (layer shapes, parameter counts, etc.).
"""

import tensorflow as tf

def main():
    # 1. Base CNN
    try:
        print("\n=== Loading Base CNN Model ===")
        base_model = tf.keras.models.load_model("../trained_models/base_model.keras")
        base_model.summary()
    except Exception as e:
        print("Could not load Base CNN Model:", e)
    
    # 2. MC-Dropout Model
    try:
        print("\n=== Loading MC-Dropout Model ===")
        mc_dropout_model = tf.keras.models.load_model("../trained_models/dropout_model.keras")
        mc_dropout_model.summary()
    except Exception as e:
        print("Could not load MC-Dropout Model:", e)
    
    # 3. Ensemble Models
    #    If you have five ensemble members named ensemble_model_1.keras, ensemble_model_2.keras, etc.
    #    we can loop over them.
    print("\n=== Loading Ensemble Models ===")
    for i in range(1, 6):
        model_path = f"../trained_models/ensemble_model_{i}.keras"
        try:
            print(f"\n--- Ensemble Model {i} ---")
            ensemble_model = tf.keras.models.load_model(model_path)
            ensemble_model.summary()
        except Exception as e:
            print(f"Could not load Ensemble Model {i}:", e)

if __name__ == "__main__":
    main()
