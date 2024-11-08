import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
import numpy as np

# Configuration variables for save paths
MODEL_SAVE_PATH = "../../trained_models/base_model.h5"         # Path to save the trained model
RESULTS_FILE_PATH = "../../trained_models/results/base_model_results.txt"         # Path to save training and evaluation results

# Build a basic CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
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

# Train the model
def train_model(model, x_train, y_train, epochs=5, batch_size=64):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

# Predict with confidence percentages
def predict_with_confidence(model, x):
    probabilities = model.predict(x)
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1) * 100  # Convert to percentage
    return predictions, confidence

# Save training, evaluation, and sample predictions with confidence to a text file
def save_results_to_file(history, test_loss, test_accuracy, predictions, confidence, filename=RESULTS_FILE_PATH):
    with open(filename, "w") as file:
        file.write("Training and Validation Results:\n")
        for epoch in range(len(history.history['loss'])):
            file.write(f"Epoch {epoch+1} - "
                       f"Loss: {history.history['loss'][epoch]:.4f}, "
                       f"Accuracy: {history.history['accuracy'][epoch]:.4f}, "
                       f"Val Loss: {history.history['val_loss'][epoch]:.4f}, "
                       f"Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}\n")
        file.write(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")
        
        file.write("\nSample Predictions with Confidence:\n")
        for i, (pred, conf) in enumerate(zip(predictions, confidence)):
            file.write(f"Sample {i} - Predicted Digit: {pred}, Confidence: {conf:.2f}%\n")

# Main execution
if __name__ == "__main__":
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0,1]
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    # Build, train, and evaluate the model
    model = build_model()
    history = train_model(model, x_train, y_train)
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save the model
    model.save(MODEL_SAVE_PATH)
    
    # Predict with confidence on a sample of test data
    predictions, confidence = predict_with_confidence(model, x_test[:10])
    for i, (pred, conf) in enumerate(zip(predictions, confidence)):
        print(f"Sample {i} - Predicted Digit: {pred}, Confidence: {conf:.2f}%")
    
    # Save training, evaluation results, and sample predictions to a text file
    save_results_to_file(history, test_loss, test_accuracy, predictions, confidence)