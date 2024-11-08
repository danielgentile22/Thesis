# mc_dropout.py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

def create_mc_dropout_model():
    """
    Creates and compiles an MC-Dropout model using convolutional layers.
    """
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),  # Input layer for 28x28 grayscale images
        layers.Conv2D(32, (3, 3), activation='relu'),  # First convolutional layer
        layers.MaxPooling2D((2, 2)),  # Max pooling to down-sample
        layers.Dropout(0.2),  # MC-Dropout (active during training and inference)
        layers.Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
        layers.MaxPooling2D((2, 2)),  # Max pooling
        layers.Dropout(0.2),  # Another dropout layer for regularization
        layers.Flatten(),  # Flatten the 2D output to feed into Dense layers
        layers.Dense(128, activation='relu'),  # Fully connected layer
        layers.Dropout(0.5),  # Final dropout before output
        layers.Dense(10)  # Output layer with logits (no softmax yet, to allow logits)
    ])
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])  # Compiling the model with Adam optimizer and sparse categorical crossentropy
    return model

def train_and_save_model(model_path="mc_dropout.h5", epochs=10):
    """
    Trains the MC-Dropout model on the MNIST dataset and saves the trained model to an .h5 file.

    Args:
        model_path (str): The path where the model will be saved.
        epochs (int): Number of epochs to train the model.
    """
    # Load and preprocess MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the images to the range [0, 1]
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Reshape the images to include a single channel (28x28x1) for the model
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    # Create the MC-Dropout model
    model = create_mc_dropout_model()

    # Train the model
    model.fit(train_images, train_labels, epochs=epochs, validation_split=0.1)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"\nTest accuracy: {test_acc}")

    # Save the trained model to the specified path
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Run the model creation and training process
    train_and_save_model()