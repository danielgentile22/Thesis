# utils.py

from PIL import Image, ImageOps
import numpy as np
import io
import matplotlib.pyplot as plt

def preprocess_image(img):
    """
    Preprocess the image: convert to grayscale and scale to 28x28
    without resizing, centering, or aspect ratio adjustments.

    Args:
        img (PIL.Image): The PIL Image to preprocess.

    Returns:
        tuple: A tuple containing the preprocessed image array and resized PIL Image.
    """
    print("Preprocessing image...")

    # If image has alpha channel, composite onto white background
    if img.mode in ('RGBA', 'LA'):
        print("Image has alpha channel, compositing onto white background.")
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img.convert('RGBA'))
        img = img.convert('RGB')

    # Convert to grayscale
    img = img.convert('L')  # Now img is grayscale

    # Invert image so that the digit is white on black background
    img = ImageOps.invert(img)

    # Resize to 28x28 without centering
    img = img.resize((28, 28), Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Apply a threshold to binarize the image
    threshold = 20
    img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)

    # Normalize image for model input
    img_array = img_array.astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    print("Image preprocessing completed.")
    return img_array, img

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
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{prob:.2f}", ha='center')

    plt.tight_layout()
    # Save plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    # Read the image from buffer
    img = Image.open(buf)
    return img