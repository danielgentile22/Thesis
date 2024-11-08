# utils.py

from PIL import Image
import numpy as np
import os
import io
import matplotlib.pyplot as plt

def preprocess_image(img):
    """
    Preprocess the image: convert to grayscale, resize, normalize.

    Args:
        img (PIL.Image): The PIL Image to preprocess.

    Returns:
        tuple: A tuple containing the preprocessed image array and resized PIL Image.
    """
    print("Preprocessing image...")
    # Convert to grayscale
    img = img.convert('L')
    # Resize to 28x28
    img_resized = img.resize((28, 28), Image.LANCZOS)
    # Convert to NumPy array and normalize
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    print("Image preprocessing completed.")
    return img_array, img_resized

def plot_probabilities(probabilities, model_name):
    """
    Plot the prediction probabilities as a bar chart.

    Args:
        probabilities (np.ndarray): The prediction probabilities.
        model_name (str): Name of the model used for prediction.

    Returns:
        PIL.Image: The plotted bar chart as an image.
    """
    digits = list(range(10))
    probabilities = probabilities.flatten()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(digits, probabilities, color='skyblue')
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.title(f"Prediction Probabilities ({model_name})")
    plt.xticks(digits)
    plt.ylim([0, 1])

    for bar, prob in zip(bars, probabilities):
        yval = bar.get_height()
        plt.text(bar.get_x() + 0.1, yval + 0.01, f"{prob:.2f}")

    plt.tight_layout()
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    # Read the image from buffer
    img = Image.open(buf)
    return img