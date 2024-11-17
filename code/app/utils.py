# utils.py

from PIL import Image, ImageOps
import numpy as np
import io
import matplotlib.pyplot as plt

def preprocess_image(img):
    """
    Preprocess the image: convert to grayscale, center the drawing, scale to 28x28,
    ensuring the drawing is centered and scaled to occupy as much of the 28x28 canvas as possible
    without touching the borders and without changing the aspect ratio.

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

    # Convert to numpy array
    img_array = np.array(img)

    # Apply a threshold to binarize the image
    threshold = 20
    img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)

    # Find bounding box of the digit
    coords = np.column_stack(np.where(img_array > 0))
    if coords.size == 0:
        # Empty image
        print("Empty image after preprocessing.")
        return None, None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the image to the bounding box
    cropped_img_array = img_array[y_min:y_max+1, x_min:x_max+1]

    # Determine scaling factor to fit the image into 20x20 (to leave margins when centered in 28x28)
    max_dim = max(cropped_img_array.shape)
    scaling_factor = 20.0 / max_dim

    # Resize the image to new size while maintaining aspect ratio
    new_size = (
        int(cropped_img_array.shape[1] * scaling_factor),
        int(cropped_img_array.shape[0] * scaling_factor)
    )
    resized_img = Image.fromarray(cropped_img_array).resize(new_size, Image.LANCZOS)

    # Create a new 28x28 image and paste the resized image into the center
    new_img = Image.new('L', (28, 28), color=0)  # black background
    upper_left = (
        (28 - new_size[0]) // 2,
        (28 - new_size[1]) // 2
    )
    new_img.paste(resized_img, upper_left)

    # Convert to numpy array and normalize
    img_array = np.array(new_img).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    print("Image preprocessing completed.")
    return img_array, new_img

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