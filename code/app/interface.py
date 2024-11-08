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
from config import BASE_RESULTS_DIR, MAX_DRAW_PER_DIGIT

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

# Global variables to track the state
current_digit = 0
draw_count = 0
digits_drawn = 0  # Total digits drawn

def process_drawing(
    drawing,
    subject_num,
    uncertainty_methods,
    model_selection_mode
):
    global current_digit, draw_count, MAX_DRAW_PER_DIGIT, digits_drawn

    print("Processing drawing...")

    # Create directories for saving results
    subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")
    digit_folder = os.path.join(subject_folder, f"digit_{current_digit}")
    draw_folder = os.path.join(digit_folder, f"draw_{draw_count + 1}")
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
            img_array, uncertainty_methods, draw_folder
        )
    elif model_selection_mode == "Use all models for each digit":
        prediction_text_output, plot_images = process_all_models(
            img_array, uncertainty_methods, draw_folder
        )
    else:
        return generate_error_response("Unknown model selection mode.")

    # Enable feedback_text and next_digit_button
    print("Processing completed.")
    return (
        gr.update(),               # Keep drawing as is
        original_display,          # Display original drawing
        processed_display,         # Display processed drawing
        prediction_text_output,    # Update prediction_text
        plot_images,               # Display probabilities_plot(s)
        gr.update(),               # instruction_text remains the same
        gr.update(interactive=True),  # Enable feedback_text
        gr.update(interactive=True)   # Enable next_digit_button
    )

def handle_drawing_input(drawing):
    """Handle the drawing input from Gradio and return a PIL Image."""
    if isinstance(drawing, dict):
        if 'composite' in drawing and drawing['composite'] is not None:
            img_data = drawing['composite']
            if isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
                img = img.convert('RGB')
                return img
            elif isinstance(img_data, Image.Image):
                img = img_data.convert('RGB')
                return img
        else:
            print("No drawing data found in 'composite' key.")
            return None
    elif isinstance(drawing, Image.Image):
        img = drawing.convert('RGB')
        return img
    else:
        print("Unsupported drawing input type.")
        return None

def generate_error_response(error_message):
    """Generate an error response for Gradio outputs."""
    print(error_message)
    return (
        gr.update(),
        None,
        None,
        error_message,
        None,
        gr.update(),
        gr.update(),
        gr.update()
    )

def process_single_model(img_array, uncertainty_methods, draw_folder):
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
        selected_model_name, predicted_digit, confidence_value, uncertainty_methods
    )

    plot_images = generate_plots(
        probabilities_for_plot, selected_model_name, uncertainty_methods, draw_folder
    )

    return prediction_text_output, plot_images

def process_all_models(img_array, uncertainty_methods, draw_folder):
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
                model_name, predicted_digit, confidence_value, uncertainty_methods
            )
            prediction_text_output_list.append(prediction_text)

            # Save to prediction file
            f.write(f"\n{model_name}:\n")
            f.write(f"Predicted Digit: {predicted_digit}\n")
            f.write(f"Confidence: {confidence_value:.2f}%\n")

            # Generate bar plot if selected
            plots = generate_plots(
                probabilities_for_plot, model_name, uncertainty_methods, draw_folder
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
    predicted_digit = predicted_labels[0]
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

def format_prediction_text(model_name, predicted_digit, confidence_value, uncertainty_methods):
    """Format the prediction text for display."""
    if "Confidence %" in uncertainty_methods:
        return f"{model_name} - Predicted Digit: {predicted_digit}, Confidence: {confidence_value:.2f}%"
    else:
        return f"{model_name} - Predicted Digit: {predicted_digit}"

def generate_plots(probabilities, model_name, uncertainty_methods, draw_folder):
    """Generate and save plots if required."""
    plot_images = []
    if "Bar Plot" in uncertainty_methods:
        plot_image = plot_probabilities(probabilities, model_name)
        # Save the plot
        plot_file_path = os.path.join(draw_folder, f"probabilities_plot_{model_name.replace(' ', '_')}.png")
        plot_image.save(plot_file_path)
        print(f"Probabilities plot saved at {plot_file_path}")
        plot_images.append(plot_image)
    return plot_images

def submit_feedback(feedback, subject_num):
    """Handle feedback submission and update the experiment state."""
    global current_digit, draw_count, MAX_DRAW_PER_DIGIT, digits_drawn

    print("Submitting feedback...")

    # Locate the prediction file
    prediction_file = os.path.join(
        BASE_RESULTS_DIR,
        f"Subject_{subject_num}",
        f"digit_{current_digit}",
        f"draw_{draw_count + 1}",
        "prediction.txt"
    )

    # Append feedback to prediction file
    with open(prediction_file, 'a') as f:
        f.write(f"Feedback: {feedback}\n")
    print(f"Feedback saved to {prediction_file}")

    # Update draw count and digit logic
    draw_count += 1
    digits_drawn += 1
    print(f"Draw count: {draw_count}, Digits drawn: {digits_drawn}")
    if draw_count == MAX_DRAW_PER_DIGIT:
        current_digit += 1
        draw_count = 0
        print(f"Moving to next digit: {current_digit}")

    # Check if all digits have been drawn twice
    if digits_drawn == MAX_DRAW_PER_DIGIT * 10:
        # Reset variables
        print("All digits have been drawn twice. Experiment completed.")
        reset_experiment_state()
        # Hide experiment page and show thank you page
        return end_experiment_response()

    # Prepare for next digit
    instruction = f"Please draw the digit {current_digit}"
    print(f"Instruction updated: {instruction}")
    return continue_experiment_response(instruction)

def reset_experiment_state():
    """Reset the global experiment state."""
    global current_digit, draw_count, digits_drawn
    current_digit = 0
    draw_count = 0
    digits_drawn = 0

def end_experiment_response():
    """Generate the response to end the experiment."""
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

def continue_experiment_response(instruction):
    """Generate the response to continue the experiment."""
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

def home_page(subject_num, uncertainty_methods, model_selection_mode):
    """Handle the transition from the home page to the consent page."""
    print("Proceeding from home page...")
    if not subject_num or len(uncertainty_methods) == 0 or not model_selection_mode:
        print("Required information not provided.")
        return gr.update(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    print("Moving to consent page.")
    return (
        gr.update(visible=False),  # Hide home page
        gr.update(visible=True),   # Show consent page
        gr.update(visible=False),  # Experiment page remains hidden
        gr.update(visible=False)   # Thank you page remains hidden
    )

def consent_page(agree):
    """Handle the consent page logic."""
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