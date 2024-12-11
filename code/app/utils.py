from PIL import Image, ImageOps
import numpy as np
import io
import matplotlib.pyplot as plt
from config import IMAGE_SIZE, IMAGE_THRESHOLD

def preprocess_image(img):
    """
    Preprocess the image: convert to grayscale and resize.

    Args:
        img (PIL.Image): The PIL Image to preprocess.

    Returns:
        tuple: A tuple containing the preprocessed image array and resized PIL Image.
    """
    # If image has alpha channel, composite onto white background
    if img.mode in ('RGBA', 'LA'):
        background = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img.convert('RGBA'))
        img = img.convert('RGB')

    # Convert to grayscale
    img = img.convert('L')  # Now img is grayscale

    # Invert image so that the digit is white on black background
    img = ImageOps.invert(img)

    # Resize to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img)

    # Apply a threshold to binarize the image
    img_array = np.where(img_array > IMAGE_THRESHOLD, 255, 0).astype(np.uint8)

    # Normalize image for model input
    img_array = img_array.astype(np.float32) / 255.0
    img_array = img_array.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
    return img_array, img

def plot_probabilities(probabilities, model_name, content_options):
    """
    Plot the prediction probabilities as a bar chart.

    Args:
        probabilities (np.ndarray): The prediction probabilities.
        model_name (str): Name of the model used for prediction.
        content_options (list): List of content options selected.

    Returns:
        PIL.Image: The plotted bar chart as an image.
    """
    import matplotlib.pyplot as plt
    import io

    digits = list(range(10))
    probabilities = probabilities.flatten()

    # Find the index of the highest probability
    max_index = np.argmax(probabilities)

    # Set colors: highest probability in one color, others in another color
    colors = ['skyblue' if i != max_index else 'orange' for i in range(len(probabilities))]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(digits, probabilities, color=colors)
    plt.xlabel("Digit")
    plt.ylabel("Probability")

    # Adjust title based on whether to show model name
    if "Show Model Name" in content_options:
        plt.title(f"Prediction Probabilities ({model_name})")
    else:
        plt.title("Prediction Probabilities")

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