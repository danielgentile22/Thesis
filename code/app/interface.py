# interface.py

import gradio as gr
from PIL import Image
import numpy as np
import os
import random
from models import (
    load_base_model,
    load_mc_dropout_model,
    load_ensemble_models,
    predict_with_base_model,
    predict_with_mc_dropout,
    predict_with_ensemble,
)
from utils import preprocess_image, plot_probabilities
from config import BASE_RESULTS_DIR, MAX_DRAW_PER_DIGIT, TOTAL_DIGITS

# Load the models
base_model = load_base_model()
mc_dropout_model = load_mc_dropout_model()
ensemble_models = load_ensemble_models()

# Mapping model names to their corresponding objects
uncertainty_models = {
    "Base Model": base_model,
    "MC-Dropout": mc_dropout_model,
    "Ensemble Model": ensemble_models
}

# Global variables to track the experiment state
digits_to_draw = []
current_index = 0

# Global variable to store content options
content_options_global = []

def initialize_experiment_state():
    """Initialize and shuffle the list of digits to draw."""
    global digits_to_draw, current_index
    # Create a list with two instances of each digit (0-9)
    digits_to_draw = [digit for digit in range(TOTAL_DIGITS) for _ in range(MAX_DRAW_PER_DIGIT)]
    random.shuffle(digits_to_draw)
    current_index = 0
    print(f"Digits to draw (shuffled): {digits_to_draw}")

def handle_drawing_input(drawing):
    """Handle the drawing input from Gradio and return a PIL Image."""
    if isinstance(drawing, dict):
        # For Gradio versions >= 3.0, the image data is under the 'image' key
        if 'image' in drawing and drawing['image'] is not None:
            img_data = drawing['image']
            if isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
                return img
            elif isinstance(img_data, Image.Image):
                img = img_data.convert('RGBA')
                return img
        # For older versions or different configurations
        elif 'composite' in drawing and drawing['composite'] is not None:
            img_data = drawing['composite']
            if isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
                return img
            elif isinstance(img_data, Image.Image):
                img = img_data.convert('RGBA')
                return img
        else:
            print("No drawing data found in 'image' or 'composite' key.")
            return None
    elif isinstance(drawing, np.ndarray):
        img = Image.fromarray(drawing.astype('uint8'), 'RGBA')
        return img
    elif isinstance(drawing, Image.Image):
        img = drawing.convert('RGBA')
        return img
    else:
        print("Unsupported drawing input type.")
        return None

def generate_error_response(error_message):
    """Generate an error response for Gradio outputs."""
    print(error_message)
    # Adjust outputs based on content options
    outputs = [
        gr.update(),                # drawing
        None if "Your Drawing" in content_options_global else gr.update(visible=False),                       # original_drawing_display
        None if "Processed Drawing" in content_options_global else gr.update(visible=False),                  # processed_drawing_display
        error_message if "Prediction Text" in content_options_global else gr.update(visible=False),           # prediction_text
        None if "Probabilities Plot" in content_options_global else gr.update(visible=False),                 # probabilities_plot
        gr.update(),                # instruction_text
        gr.update(),                # q1
        gr.update(),                # q2
        gr.update(),                # q3
        gr.update(interactive=False)  # next_digit_button
    ]
    return tuple(outputs)

def process_drawing(
    drawing,
    subject_num,
    uncertainty_methods,
    model_selection_mode,
    content_options
):
    global digits_to_draw, current_index, content_options_global
    content_options_global = content_options  # Store content options globally

    if current_index >= len(digits_to_draw):
        # All digits have been drawn; reset the experiment state
        print("All digits have been drawn. Experiment completed.")
        reset_experiment_state()
        # Hide experiment page and show thank you page
        return end_experiment_response()

    current_digit = digits_to_draw[current_index]
    print(f"Processing drawing for digit {current_digit} (Index: {current_index})")

    # Create directories for saving results
    subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")
    digit_folder = os.path.join(subject_folder, f"digit_{current_digit}")
    os.makedirs(digit_folder, exist_ok=True)
    draw_number = len([name for name in os.listdir(digit_folder) if os.path.isdir(os.path.join(digit_folder, name))]) + 1
    draw_folder = os.path.join(digit_folder, f"draw_{draw_number}")
    os.makedirs(draw_folder, exist_ok=True)

    # Handle the drawing input
    img = handle_drawing_input(drawing)
    if img is None:
        return generate_error_response("Unsupported image format.")

    # Save the original drawing
    original_file_path = os.path.join(draw_folder, "original_drawing.png")
    img.save(original_file_path)
    print(f"Original drawing saved at {original_file_path}")

    # Preprocess the image
    img_array, img_resized = preprocess_image(img)
    if img_array is None or img_resized is None:
        return generate_error_response("Preprocessing failed. Please draw a clearer digit.")

    # Save the preprocessed image
    processed_file_path = os.path.join(draw_folder, "processed_drawing.png")
    img_resized.save(processed_file_path)
    print(f"Processed drawing saved at {processed_file_path}")

    # Prepare outputs
    original_display = img.resize((200, 200))
    processed_display = img_resized.resize((200, 200))

    # Initialize variables for predictions
    prediction_text_output = ""
    plot_images = []

    # Determine model(s) to use
    if model_selection_mode == "Randomly pick one model per digit":
        prediction_text_output, plot_images = process_single_model(
            img_array, uncertainty_methods, draw_folder, current_digit
        )
    elif model_selection_mode == "Use all models for each digit":
        prediction_text_output, plot_images = process_all_models(
            img_array, uncertainty_methods, draw_folder, current_digit
        )
    else:
        return generate_error_response("Unknown model selection mode.")

    # Enable Likert Scale Questions and next_digit_button
    print("Processing completed.")

    # Prepare outputs based on content options
    outputs = [
        gr.update(),                               # Keep drawing as is
        original_display if "Your Drawing" in content_options else gr.update(visible=False),           # original_drawing_display
        processed_display if "Processed Drawing" in content_options else gr.update(visible=False),     # processed_drawing_display
        prediction_text_output if "Prediction Text" in content_options else gr.update(visible=False),  # prediction_text
        plot_images if "Probabilities Plot" in content_options else gr.update(visible=False),          # probabilities_plot
        gr.update(),                               # instruction_text remains the same
    ]

    # Handle feedback questions
    if "Feedback Questions" in content_options:
        outputs.extend([
            gr.update(interactive=True, visible=True),  # Enable q1
            gr.update(interactive=True, visible=True),  # Enable q2
            gr.update(interactive=True, visible=True),  # Enable q3
        ])
    else:
        outputs.extend([
            gr.update(visible=False),  # Hide q1
            gr.update(visible=False),  # Hide q2
            gr.update(visible=False),  # Hide q3
        ])

    outputs.append(gr.update(interactive=True))  # Enable next_digit_button

    return tuple(outputs)

def process_single_model(img_array, uncertainty_methods, draw_folder, current_digit):
    """Process the drawing using a single randomly selected model."""
    model_options = ["Base Model", "MC-Dropout", "Ensemble Model"]
    selected_model_name = random.choice(model_options)
    print(f"Selected model: {selected_model_name}")

    model = uncertainty_models[selected_model_name]
    predicted_digit, confidence_value, probabilities_for_plot = predict_model(
        selected_model_name, model, img_array
    )

    prediction_file = os.path.join(draw_folder, "prediction.txt")
    save_prediction(
        prediction_file, current_digit, selected_model_name, predicted_digit, confidence_value
    )

    prediction_text_output = format_prediction_text(
        selected_model_name, predicted_digit, confidence_value, uncertainty_methods, content_options_global
    )

    plot_images = generate_plots(
        probabilities_for_plot, selected_model_name, uncertainty_methods, draw_folder, content_options_global
    )

    return prediction_text_output, plot_images

def process_all_models(img_array, uncertainty_methods, draw_folder, current_digit):
    """Process the drawing using all available models."""
    models_to_use = ["Base Model", "MC-Dropout", "Ensemble Model"]
    prediction_text_output_list = []
    plot_images = []

    prediction_file = os.path.join(draw_folder, "prediction.txt")
    with open(prediction_file, 'w') as f:
        f.write(f"Intended Digit: {current_digit}\n")
        f.write(f"Models Used: All Models\n")

        for model_name in models_to_use:
            model = uncertainty_models[model_name]
            predicted_digit, confidence_value, probabilities_for_plot = predict_model(
                model_name, model, img_array
            )

            prediction_text = format_prediction_text(
                model_name, predicted_digit, confidence_value, uncertainty_methods, content_options_global
            )
            prediction_text_output_list.append(prediction_text)

            # Save to prediction file
            f.write(f"\n{model_name}:\n")
            f.write(f"Predicted Digit: {predicted_digit}\n")
            f.write(f"Confidence: {confidence_value:.2f}%\n")

            # Generate bar plot if selected
            plots = generate_plots(
                probabilities_for_plot, model_name, uncertainty_methods, draw_folder, content_options_global
            )
            plot_images.extend(plots)

    prediction_text_output = "\n\n".join(prediction_text_output_list)
    return prediction_text_output, plot_images

def predict_model(model_name, model, img_array):
    """Predict using the specified model."""
    if model_name == "Base Model":
        predicted_labels, confidence, probabilities = predict_with_base_model(model, img_array)
    elif model_name == "MC-Dropout":
        predicted_labels, confidence, probabilities = predict_with_mc_dropout(model, img_array)
    elif model_name == "Ensemble Model":
        predicted_labels, confidence, probabilities = predict_with_ensemble(model, img_array)
    else:
        raise ValueError("Unknown model name.")
    predicted_digit = int(predicted_labels[0])
    confidence_value = confidence[0]
    return predicted_digit, confidence_value, probabilities

def save_prediction(prediction_file, intended_digit, model_name, predicted_digit, confidence_value):
    """Save prediction details to a file."""
    with open(prediction_file, 'w') as f:
        f.write(f"Intended Digit: {intended_digit}\n")
        f.write(f"Model Used: {model_name}\n")
        f.write(f"Predicted Digit: {predicted_digit}\n")
        f.write(f"Confidence: {confidence_value:.2f}%\n")
    print(f"Prediction saved at {prediction_file}")

def format_prediction_text(model_name, predicted_digit, confidence_value, uncertainty_methods, content_options):
    """Format the prediction text for display."""
    if "Confidence %" in uncertainty_methods:
        text = f"Predicted Digit: {predicted_digit}, Confidence: {confidence_value:.2f}%"
    else:
        text = f"Predicted Digit: {predicted_digit}"

    if "Show Model Name" in content_options:
        text = f"{model_name} - {text}"

    return text

def generate_plots(probabilities, model_name, uncertainty_methods, draw_folder, content_options):
    """Generate and save plots if required."""
    plot_images = []
    if "Bar Plot" in uncertainty_methods:
        plot_image = plot_probabilities(probabilities, model_name, content_options)
        # Save the plot
        plot_file_path = os.path.join(draw_folder, f"probabilities_plot_{model_name.replace(' ', '_')}.png")
        plot_image.save(plot_file_path)
        print(f"Probabilities plot saved at {plot_file_path}")
        plot_images.append(plot_image)
    return plot_images

def submit_feedback(q1_answer, q2_answer, q3_answer, subject_num):
    """Handle feedback submission and update the experiment state."""
    global digits_to_draw, current_index

    current_digit = digits_to_draw[current_index]
    print(f"Submitting feedback for digit {current_digit} (Index: {current_index})")

    # Locate the prediction file
    subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")
    digit_folder = os.path.join(subject_folder, f"digit_{current_digit}")
    draw_folders = sorted([d for d in os.listdir(digit_folder) if d.startswith("draw_")])
    if draw_folders:
        last_draw_folder = draw_folders[-1]
        prediction_file = os.path.join(digit_folder, last_draw_folder, "prediction.txt")
    else:
        print("No prediction file found.")
        return generate_error_response("No prediction file found.")

    # Append feedback to prediction file
    with open(prediction_file, 'a') as f:
        f.write(f"\nFeedback:\n")
        f.write(f"1. Is the top prediction appropriate? {q1_answer}\n")
        f.write(f"2. Are the alternative predictions appropriate? {q2_answer}\n")
        f.write(f"3. In relation to how clear the drawing is, is the prediction too confident? {q3_answer}\n")
    print(f"Feedback saved to {prediction_file}")

    # Move to the next digit
    current_index += 1

    if current_index >= len(digits_to_draw):
        # All digits have been drawn; reset the experiment state
        print("All digits have been drawn. Experiment completed.")
        reset_experiment_state()
        # Hide experiment page and show thank you page
        return end_experiment_response()

    # Prepare for next digit
    next_digit = digits_to_draw[current_index]
    instruction = f"Please draw the digit {next_digit}"
    print(f"Instruction updated: {instruction}")
    return continue_experiment_response(instruction)

def reset_experiment_state():
    """Reset the global experiment state."""
    initialize_experiment_state()

def end_experiment_response():
    """Generate the response to end the experiment."""
    # Adjust outputs based on content options
    outputs = [
        gr.update(value=None),   # Clear drawing
        None if "Your Drawing" in content_options_global else gr.update(visible=False),   # Clear original drawing display
        None if "Processed Drawing" in content_options_global else gr.update(visible=False),   # Clear processed drawing display
        "" if "Prediction Text" in content_options_global else gr.update(visible=False),  # Clear prediction_text
        None if "Probabilities Plot" in content_options_global else gr.update(visible=False),  # Clear probabilities_plot
        gr.update(value="Thank you for participating!"),  # Update instruction_text
        gr.update(value=None, interactive=False, visible=False),  # Clear and disable q1
        gr.update(value=None, interactive=False, visible=False),  # Clear and disable q2
        gr.update(value=None, interactive=False, visible=False),  # Clear and disable q3
        gr.update(interactive=False),              # Disable next_digit_button
        gr.update(visible=False),                  # Hide experiment_page_container
        gr.update(visible=True)                    # Show thank_you_page_container
    ]
    return tuple(outputs)

def continue_experiment_response(instruction):
    """Generate the response to continue the experiment."""
    # Adjust outputs based on content options
    outputs = [
        gr.update(value=None),          # Clear drawing
        None if "Your Drawing" in content_options_global else gr.update(visible=False),   # Clear original drawing display
        None if "Processed Drawing" in content_options_global else gr.update(visible=False),   # Clear processed drawing display
        "" if "Prediction Text" in content_options_global else gr.update(visible=False),  # Clear prediction_text
        None if "Probabilities Plot" in content_options_global else gr.update(visible=False),  # Clear probabilities_plot
        gr.update(value=instruction),   # Update instruction_text
    ]

    # Handle feedback questions
    if "Feedback Questions" in content_options_global:
        outputs.extend([
            gr.update(value=None, interactive=False, visible=True),  # Clear and disable q1
            gr.update(value=None, interactive=False, visible=True),  # Clear and disable q2
            gr.update(value=None, interactive=False, visible=True),  # Clear and disable q3
        ])
    else:
        outputs.extend([
            gr.update(visible=False),  # Hide q1
            gr.update(visible=False),  # Hide q2
            gr.update(visible=False),  # Hide q3
        ])

    outputs.append(gr.update(interactive=False))  # Disable next_digit_button
    outputs.extend([
        gr.update(),                    # Keep experiment_page_container as is
        gr.update()                     # Keep thank_you_page_container as is
    ])
    return tuple(outputs)

def home_page(subject_num, uncertainty_methods, model_selection_mode, content_options):
    """Handle the transition from the home page to the consent page."""
    global content_options_global
    content_options_global = content_options  # Store content options globally
    print("Proceeding from home page...")
    if not subject_num or len(uncertainty_methods) == 0 or not model_selection_mode:
        print("Required information not provided.")
        return (
            gr.update(),                # home_page_container remains as is
            gr.update(visible=False),   # consent_page_container remains hidden
            gr.update(visible=False),   # instructions_page_container remains hidden
            gr.update(visible=False),   # experiment_page_container remains hidden
            gr.update(visible=False),   # thank_you_page_container remains hidden
            gr.update(),                # original_drawing_display
            gr.update(),                # processed_drawing_display
            gr.update(),                # prediction_text
            gr.update(),                # probabilities_plot
            gr.update(),                # feedback_instruction
            gr.update(),                # q1
            gr.update(),                # q2
            gr.update(),                # q3
        )
    print("Moving to consent page.")
    return (
        gr.update(visible=False),  # Hide home page
        gr.update(visible=True),   # Show consent page
        gr.update(visible=False),  # instructions_page_container remains hidden
        gr.update(visible=False),  # experiment_page_container remains hidden
        gr.update(visible=False),  # thank_you_page_container remains hidden
        # Update visibility of content boxes
        gr.update(visible="Your Drawing" in content_options),
        gr.update(visible="Processed Drawing" in content_options),
        gr.update(visible="Prediction Text" in content_options),
        gr.update(visible="Probabilities Plot" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
    )

def consent_page(agree):
    """Handle the consent page logic."""
    if agree:
        print("Consent given. Proceeding to instructions.")
        return (
            gr.update(visible=False),  # Hide consent page
            gr.update(visible=True)    # Show instructions page
        )
    else:
        print("Consent not given. Staying on consent page.")
        return (
            gr.update(),               # consent_page_container remains as is
            gr.update(visible=False)   # instructions_page_container remains hidden
        )

def instructions_page():
    """Handle the transition from instructions page to experiment page."""
    global current_index
    initialize_experiment_state()
    current_digit = digits_to_draw[current_index]
    instruction = f"Please draw the digit {current_digit}"
    print("Starting experiment.")
    return (
        gr.update(visible=False),  # Hide instructions page
        gr.update(visible=True),   # Show experiment page
        gr.update(value=instruction)
    )