import gradio as gr
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore

# Function to load the MC-Dropout model
def load_mc_dropout_model(model_path="mc_dropout.h5"):
    print("Loading MC-Dropout model...")
    model = load_model(model_path)
    print("Model loaded.")
    return model

# Function for MC-Dropout predictions with uncertainty
def predict_with_uncertainty(f_model, x, n_iter=100):
    print("Predicting with uncertainty...")
    result = np.zeros((n_iter,) + f_model(x).shape)
    for i in range(n_iter):
        result[i] = f_model(x, training=True)
    prediction_mean = result.mean(axis=0)
    prediction_std = result.std(axis=0)
    prediction_mean_softmax = tf.nn.softmax(prediction_mean).numpy()
    print("Prediction completed.")
    return prediction_mean_softmax, prediction_std

# Function to plot probabilities as a bar chart
def plot_probabilities(mean_prediction):
    import matplotlib.pyplot as plt
    import io
    digits = list(range(10))
    probabilities = mean_prediction.flatten()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(digits, probabilities, color='skyblue')
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")
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

# Load the Keras model with MC-Dropout
mc_dropout_model = load_mc_dropout_model("mc_dropout.h5")

# Dictionary mapping model names to models
uncertainty_models = {
    "MC-Dropout": mc_dropout_model,
}

# Global variables to track the state
current_digit = 0
draw_count = 0
max_draw_per_digit = 2
digits_drawn = 0  # Total digits drawn

def preprocess_image(img):
    """
    Simplified preprocessing: Convert the image to 28x28 grayscale.

    Args:
        img: The PIL Image to preprocess.

    Returns:
        img_array: The preprocessed image as a NumPy array suitable for model input.
        img_resized: The processed PIL Image.
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

def process_drawing(
    drawing,
    subject_num,
    uncertainty_methods
):
    global current_digit, draw_count, max_draw_per_digit, digits_drawn

    print("Processing drawing...")
    folder_name = f"{subject_num}"
    os.makedirs(folder_name, exist_ok=True)

    # Handle the drawing input
    if isinstance(drawing, dict):
        # Extract the image from the 'composite' key
        if 'composite' in drawing and drawing['composite'] is not None:
            img_data = drawing['composite']
            print(f"Type of img_data: {type(img_data)}")
            if isinstance(img_data, np.ndarray):
                print("img_data is a NumPy array.")
                img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
                img = img.convert('RGB')
            elif isinstance(img_data, Image.Image):
                print("img_data is a PIL Image.")
                img = img_data.convert('RGB')
            else:
                print("Unsupported image data type.")
                return (
                    gr.update(),
                    None,
                    None,
                    "Unsupported image format.",
                    None,
                    gr.update(),
                    gr.update(),
                    gr.update()
                )
        else:
            # Handle missing image data
            print("No drawing data found in 'composite' key.")
            return (
                gr.update(),
                None,
                None,
                "No drawing data found.",
                None,
                gr.update(),
                gr.update(),
                gr.update()
            )
    else:
        # If drawing is not a dict, assume it's a PIL Image
        img = drawing
        print("Drawing is a PIL Image.")
        img = img.convert('RGB')

    # Save the original drawing
    original_file_name = f"original_drawing_digit_{current_digit}_draw_{draw_count + 1}.png"
    original_file_path = os.path.join(folder_name, original_file_name)
    img.save(original_file_path)
    print(f"Original drawing saved at {original_file_path}")

    # Preprocess the image (simplified)
    img_array, img_resized = preprocess_image(img)

    # Save the preprocessed image
    processed_file_name = f"processed_drawing_digit_{current_digit}_draw_{draw_count + 1}.png"
    processed_file_path = os.path.join(folder_name, processed_file_name)
    img_resized.save(processed_file_path)
    print(f"Processed drawing saved at {processed_file_path}")

    # Select the model
    model = uncertainty_models["MC-Dropout"]

    # Make prediction with uncertainty
    prediction_mean_softmax, prediction_std = predict_with_uncertainty(model, img_array)
    predicted_digit = np.argmax(prediction_mean_softmax)
    confidence = np.max(prediction_mean_softmax) * 100
    print(f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}%")

    # Save prediction and uncertainty
    prediction_file = os.path.join(folder_name, f"prediction_digit_{current_digit}_draw_{draw_count + 1}.txt")
    with open(prediction_file, 'w') as f:
        f.write(f"Intended Digit: {current_digit}\n")
        f.write(f"Predicted Digit: {predicted_digit}\n")
        f.write(f"Confidence: {confidence:.2f}%\n")
        f.write(f"Uncertainty: {prediction_std.max():.4f}\n")
        # Feedback will be saved later
    print(f"Prediction saved at {prediction_file}")

    # Prepare outputs
    prediction_text_output = (
        f"Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}%"
    ) if "Confidence %" in uncertainty_methods else f"Predicted Digit: {predicted_digit}"

    # Generate the bar plot if "Bar Plot" is selected
    if "Bar Plot" in uncertainty_methods:
        plot_image = plot_probabilities(prediction_mean_softmax)
    else:
        plot_image = None

    # Prepare images for display
    original_display = img.resize((200, 200))
    processed_display = img_resized.resize((200, 200))

    # Enable feedback_text and next_digit_button
    print("Processing completed.")
    return (
        gr.update(),               # Keep drawing as is
        original_display,          # Display original drawing
        processed_display,         # Display processed drawing
        prediction_text_output,    # Update prediction_text
        plot_image,                # Display probabilities_plot
        gr.update(),               # instruction_text remains the same
        gr.update(interactive=True),  # Enable feedback_text
        gr.update(interactive=True)   # Enable next_digit_button
    )

def submit_feedback(
    feedback,
    subject_num
):
    global current_digit, draw_count, max_draw_per_digit, digits_drawn

    print("Submitting feedback...")
    folder_name = f"{subject_num}"

    # Append feedback to prediction file
    prediction_file = os.path.join(
        folder_name, f"prediction_digit_{current_digit}_draw_{draw_count + 1}.txt"
    )
    with open(prediction_file, 'a') as f:
        f.write(f"Feedback: {feedback}\n")
    print(f"Feedback saved to {prediction_file}")

    # Update draw count and digit logic
    draw_count += 1
    digits_drawn += 1
    print(f"Draw count: {draw_count}, Digits drawn: {digits_drawn}")
    if draw_count == max_draw_per_digit:
        current_digit += 1
        draw_count = 0
        print(f"Moving to next digit: {current_digit}")

    # Check if all digits have been drawn twice
    if digits_drawn == 20:
        # Reset variables
        print("All digits have been drawn twice. Experiment completed.")
        current_digit = 0
        draw_count = 0
        digits_drawn = 0
        # Hide experiment page and show thank you page
        return (
            gr.update(value=None),   # Clear drawing
            None,                    # Clear original drawing display
            None,                    # Clear processed drawing display
            "",                      # Clear prediction_text
            None,                    # Clear probabilities_plot
            gr.update(value="Thank you for participating!"),  # Update instruction_text
            gr.update(value="", interactive=False),  # Clear and disable feedback_text
            gr.update(interactive=False),            # Disable next_digit_button
            gr.update(visible=False),                # Hide experiment_page_container
            gr.update(visible=True)                  # Show thank_you_page_container
        )

    # Prepare for next digit
    instruction = f"Please draw the digit {current_digit}"
    print(f"Instruction updated: {instruction}")
    return (
        gr.update(value=None),          # Clear drawing
        None,                           # Clear original drawing display
        None,                           # Clear processed drawing display
        "",                             # Clear prediction_text
        None,                           # Clear probabilities_plot
        gr.update(value=instruction),   # Update instruction_text
        gr.update(value="", interactive=False),  # Clear and disable feedback_text
        gr.update(interactive=False),   # Disable next_digit_button
        gr.update(),                    # Keep experiment_page_container as is
        gr.update()                     # Keep thank_you_page_container as is
    )

def home_page(subject_num, uncertainty_methods):
    print("Proceeding from home page...")
    if not subject_num or len(uncertainty_methods) == 0:
        print("Subject number or uncertainty methods not provided.")
        return gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    print("Moving to consent page.")
    return (
        gr.update(visible=False),  # Hide home page
        gr.update(visible=True),   # Show consent page
        gr.update(visible=False),  # Experiment page remains hidden
        gr.update(visible=False)   # Thank you page remains hidden
    )

def consent_page(agree):
    print("Consent page response received.")
    if agree:
        instruction = f"Please draw the digit {current_digit}"
        print("Consent given. Starting experiment.")
        return (
            gr.update(visible=False),  # Hide consent page
            gr.update(visible=True),   # Show experiment page
            gr.update(value=instruction)
        )
    else:
        print("Consent not given. Staying on consent page.")
        return gr.update(), gr.update(visible=False), gr.update()

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Handwritten Digit Recognition with Uncertainty Visualization")

    # Home Page
    with gr.Column(visible=True) as home_page_container:
        subject_num = gr.Textbox(label="Subject Number")
        uncertainty_methods = gr.CheckboxGroup(choices=["Confidence %", "Bar Plot"], label="Select Uncertainty Methods")
        proceed_button = gr.Button("Proceed to Consent")

    # Consent Page
    with gr.Column(visible=False) as consent_page_container:
        gr.Markdown("This is an experiment. Do you agree to participate?")
        agree_checkbox = gr.Checkbox(label="I agree to participate in this experiment.")
        start_experiment_button = gr.Button("Start Experiment")

    # Experiment Page
    with gr.Column(visible=False) as experiment_page_container:
        instruction_text = gr.Textbox(label="Instructions", interactive=False)
        drawing = gr.ImageEditor(label="Draw a Digit", height=400, width=400)
        submit_drawing_button = gr.Button("Submit Drawing")
        original_drawing_display = gr.Image(label="Your Drawing")  # Display original drawing
        processed_drawing_display = gr.Image(label="Processed Drawing (28x28)")  # Display processed image
        prediction_text = gr.Textbox(label="Prediction", interactive=False)
        probabilities_plot = gr.Image(label="Prediction Probabilities")  # Added back
        feedback_text = gr.Textbox(label="Feedback on the Prediction", placeholder="Enter your feedback here...", interactive=False)
        next_digit_button = gr.Button("Next Digit", interactive=False)

    # Thank You Page
    with gr.Column(visible=False) as thank_you_page_container:
        gr.Markdown("Thank you for participating in the experiment!")

    # Home Page Button Click
    proceed_button.click(
        home_page,
        inputs=[subject_num, uncertainty_methods],
        outputs=[home_page_container, consent_page_container, experiment_page_container, thank_you_page_container]
    )

    # Consent Page Button Click
    start_experiment_button.click(
        consent_page,
        inputs=[agree_checkbox],
        outputs=[consent_page_container, experiment_page_container, instruction_text]
    )

    # Submit Drawing Button Click
    submit_drawing_button.click(
        process_drawing,
        inputs=[drawing, subject_num, uncertainty_methods],
        outputs=[
            drawing,                  # Keep drawing as is
            original_drawing_display, # Display original drawing
            processed_drawing_display,# Display processed drawing
            prediction_text,          # Update prediction_text
            probabilities_plot,       # Display probabilities_plot
            instruction_text,         # Keep instruction_text
            feedback_text,            # Enable feedback_text
            next_digit_button         # Enable next_digit_button
        ]
    )

    # Next Digit Button Click
    next_digit_button.click(
        submit_feedback,
        inputs=[feedback_text, subject_num],
        outputs=[
            drawing,                   # Clear drawing
            original_drawing_display,  # Clear original drawing display
            processed_drawing_display, # Clear processed drawing display
            prediction_text,           # Clear prediction_text
            probabilities_plot,        # Clear probabilities_plot
            instruction_text,          # Update instruction_text
            feedback_text,             # Clear and disable feedback_text
            next_digit_button,         # Disable next_digit_button
            experiment_page_container, # Show/hide experiment page
            thank_you_page_container   # Show/hide thank you page
        ]
    )

demo.launch()