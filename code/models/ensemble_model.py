import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
import numpy as np

# Configuration variables for save paths
MODEL_SAVE_PATH = "../../trained_models/deep_ensemble_base_model"  # Folder to save ensemble models
RESULTS_FILE_PATH = "../../trained_models/results/deep_ensemble_results.txt"  # Path to save training and evaluation results

# Number of models in the ensemble
ENSEMBLE_SIZE = 5

# Build a basic CNN model
def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes for MNIST digits
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Train the model with early stopping
def train_model(model, x_train, y_train, epochs=20, batch_size=64, patience=3):
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping])
    return history

# Evaluate a single model
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

# Ensemble prediction with confidence
def predict_with_ensemble(ensemble, x):
    ensemble_predictions = [model.predict(x) for model in ensemble]
    mean_predictions = np.mean(ensemble_predictions, axis=0)
    predicted_labels = np.argmax(mean_predictions, axis=1)
    confidence = np.max(mean_predictions, axis=1) * 100  # Convert to percentage
    return predicted_labels, confidence

# Save training, evaluation, and sample predictions with confidence to a text file
def save_results_to_file(histories, test_losses, test_accuracies, predictions, confidence, filename=RESULTS_FILE_PATH):
    with open(filename, "w") as file:
        file.write("Training and Validation Results for Each Ensemble Model:\n")
        for i, history in enumerate(histories):
            file.write(f"\nModel {i+1}:\n")
            for epoch in range(len(history.history['loss'])):
                file.write(f"Epoch {epoch+1} - "
                           f"Loss: {history.history['loss'][epoch]:.4f}, "
                           f"Accuracy: {history.history['accuracy'][epoch]:.4f}, "
                           f"Val Loss: {history.history['val_loss'][epoch]:.4f}, "
                           f"Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}\n")
            file.write(f"Test Loss: {test_losses[i]:.4f}, Test Accuracy: {test_accuracies[i]:.4f}\n")
        
        file.write("\nEnsemble Predictions with Confidence:\n")
        for i, (pred, conf) in enumerate(zip(predictions, confidence)):
            file.write(f"Sample {i} - Predicted Digit: {pred}, Confidence: {conf:.2f}%\n")

# Main execution
if __name__ == "__main__":
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0,1]
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    # Train and evaluate each model in the ensemble
    ensemble = []
    histories = []
    test_losses = []
    test_accuracies = []
    
    for i in range(ENSEMBLE_SIZE):
        print(f"Training model {i+1}/{ENSEMBLE_SIZE}...")
        
        # Build and train the model
        model = build_model()
        history = train_model(model, x_train, y_train)
        
        # Save training history and add model to ensemble
        histories.append(history)
        ensemble.append(model)
        
        # Evaluate the model on test data
        test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Save each model in the ensemble
        model.save(f"{MODEL_SAVE_PATH}_model_{i+1}.keras")
    
    # Predict with ensemble on a sample of test data
    predictions, confidence = predict_with_ensemble(ensemble, x_test[:10])
    for i, (pred, conf) in enumerate(zip(predictions, confidence)):
        print(f"Sample {i} - Predicted Digit: {pred}, Confidence: {conf:.2f}%")
    
    # Save training, evaluation results, and sample predictions to a text file
    save_results_to_file(histories, test_losses, test_accuracies, predictions, confidence)