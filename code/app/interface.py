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
from config import (
    BASE_RESULTS_DIR,
    MAX_DRAW_PER_DIGIT,
    TOTAL_DIGITS,
    NUM_PRACTICE_DIGITS
)

# Load the models
base_model = load_base_model()
mc_dropout_model = load_mc_dropout_model()
ensemble_models = load_ensemble_models()

uncertainty_models = {
    "Base Model": base_model,
    "MC-Dropout": mc_dropout_model,
    "Ensemble Model": ensemble_models
}

# Global variables to track the experiment state
practice_digits_to_draw = []
practice_current_index = 0
digits_to_draw = []
current_index = 0
is_practice = True

# Global variable to store content options
content_options_global = []
subject_num = None

def initialize_experiment_state(skip_practice, selected_digits):
    """Initialize and shuffle the list of digits to draw based on user selections."""
    global practice_digits_to_draw, practice_current_index, digits_to_draw, current_index, is_practice

    # Convert selected digits to integers
    chosen_digits = [int(d) for d in selected_digits]

    if len(chosen_digits) == 0:
        # No digits selected
        practice_digits_to_draw = []
        digits_to_draw = []
        is_practice = False
        return

    if skip_practice:
        # Skip practice runs entirely
        practice_digits_to_draw = []
        is_practice = False
    else:
        # Select practice digits from chosen_digits
        random.shuffle(chosen_digits)
        practice_digits_to_draw = chosen_digits[:min(NUM_PRACTICE_DIGITS, len(chosen_digits))]
        is_practice = len(practice_digits_to_draw) > 0

    # Shuffle main digits for actual experiment
    # If we had practice digits, ensure main digits are also from chosen set
    # Actually, let's just keep it simple: main digits = chosen_digits
    # If some digits used for practice, they can appear again in main since the code does not currently exclude them.
    # If you want to exclude practiced digits from the main run, you'd remove them from chosen_digits.
    digits_to_draw = chosen_digits.copy()
    random.shuffle(digits_to_draw)

    practice_current_index = 0
    current_index = 0

def handle_drawing_input(drawing):
    if isinstance(drawing, dict):
        img_data = drawing.get('image') or drawing.get('composite')
        if img_data is None:
            return None
        if isinstance(img_data, np.ndarray):
            return Image.fromarray(img_data.astype('uint8'), 'RGBA')
        elif isinstance(img_data, Image.Image):
            return img_data.convert('RGBA')
    elif isinstance(drawing, np.ndarray):
        return Image.fromarray(drawing.astype('uint8'), 'RGBA')
    elif isinstance(drawing, Image.Image):
        return drawing.convert('RGBA')
    return None

def generate_error_response(error_message):
    print(error_message)
    # Returning 13 outputs plus the final button update (total 13) from process_drawing scenario
    # process_drawing expects 13 outputs plus next_digit_button (14 total)
    outputs = [
        gr.update(value=None),  # drawing
        None if "Your Drawing" in content_options_global else gr.update(visible=False),
        None if "Processed Drawing" in content_options_global else gr.update(visible=False),
        error_message if "Prediction Text" in content_options_global else gr.update(visible=False),
        None if "Probabilities Plot" in content_options_global else gr.update(visible=False),
        gr.update(),  # instruction_text
        gr.update(),  # progress_text
        gr.update(value=None, visible=False, interactive=False), # q1
        gr.update(value=None, visible=False, interactive=False), # q2
        gr.update(value=None, visible=False, interactive=False), # q3
        gr.update(value=None, visible=False, interactive=False), # q4
        gr.update(value=None, visible=False, interactive=False), # q5
        gr.update(interactive=False) # next_digit_button
    ]
    return tuple(outputs)

def home_page(subject_num_input, subject_name_input, uncertainty_methods, model_selection_mode, content_options, skip_practice, selected_digits):
    global content_options_global, subject_num
    subject_num = subject_num_input
    subject_name = subject_name_input
    content_options_global = content_options
    print("Proceeding from home page...")

    subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")
    os.makedirs(subject_folder, exist_ok=True)

    subject_info_file = os.path.join(subject_folder, 'subject_info.txt')
    with open(subject_info_file, 'w') as f:
        f.write(f"Subject Number: {subject_num}\n")
        f.write(f"Name: {subject_name}\n")
    print(f"Subject information saved at {subject_info_file}")

    # Check required fields
    if not subject_num or len(uncertainty_methods) == 0 or not model_selection_mode:
        print("Required information not provided.")
        return (
            gr.update(),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update()
        )

    # Initialize experiment state with skip_practice and selected_digits
    initialize_experiment_state(skip_practice, selected_digits)

    # If no digits chosen at all
    if len(digits_to_draw) == 0 and not is_practice:
        # Can't start experiment
        print("No digits selected. Cannot start experiment.")
        return (
            gr.update(visible=False),  # home page
            gr.update(visible=False),  # experiment page
            gr.update(value="No digits selected. Cannot start the experiment."), # instruction
            gr.update(value=""),  # progress
            gr.update(visible=False), # original
            gr.update(visible=False), # processed
            gr.update(visible=False), # pred_text
            gr.update(visible=False), # probs
            gr.update(visible=False), # feedback_instruction
            gr.update(visible=False), # q1
            gr.update(visible=False), # q2
            gr.update(visible=False), # q3
            gr.update(visible=False), # q4
            gr.update(visible=False), # q5
        )

    # Start instructions
    if is_practice and len(practice_digits_to_draw) > 0:
        current_digit = practice_digits_to_draw[practice_current_index]
        instruction = f"Practice Run: Please draw the digit {current_digit}"
        progress = f"Practice Run {practice_current_index + 1} / {len(practice_digits_to_draw)}"
    else:
        # No practice or skip practice
        if current_index < len(digits_to_draw):
            current_digit = digits_to_draw[current_index]
            instruction = f"Please draw the digit {current_digit}"
            progress = f"{current_index + 1} / {len(digits_to_draw)}"
        else:
            # No digits to draw
            instruction = "No digits to draw."
            progress = ""

    return (
        gr.update(visible=False),  # Hide home page
        gr.update(visible=True),   # Show experiment page
        gr.update(value=instruction),
        gr.update(value=progress),
        gr.update(visible="Your Drawing" in content_options_global),
        gr.update(visible="Processed Drawing" in content_options_global),
        gr.update(visible="Prediction Text" in content_options_global),
        gr.update(visible="Probabilities Plot" in content_options_global),
        gr.update(visible="Feedback Questions" in content_options_global),
        gr.update(visible="Feedback Questions" in content_options_global),
        gr.update(visible="Feedback Questions" in content_options_global),
        gr.update(visible="Feedback Questions" in content_options_global),
        gr.update(visible="Feedback Questions" in content_options_global),
        gr.update()
    )


def process_drawing(
    drawing,
    subject_num_input,
    uncertainty_methods,
    model_selection_mode,
    content_options
):
    global digits_to_draw, current_index, practice_digits_to_draw, practice_current_index, is_practice, content_options_global, subject_num
    content_options_global = content_options
    subject_num = subject_num_input

    if is_practice:
        if practice_current_index >= len(practice_digits_to_draw):
            # Move to main experiment
            is_practice = False
            if current_index >= len(digits_to_draw):
                # no main digits
                return end_experiment_response()
            current_digit = digits_to_draw[current_index]
            instruction = f"Please draw the digit {current_digit}"
            progress = f"{current_index + 1} / {len(digits_to_draw)}"
            print("Practice runs completed. Proceeding to main experiment.")
            return continue_experiment_response(instruction, progress)
        else:
            current_digit = practice_digits_to_draw[practice_current_index]
            progress = f"Practice Run {practice_current_index + 1} / {len(practice_digits_to_draw)}"
            print(f"Processing practice drawing for digit {current_digit} (Practice Index: {practice_current_index})")
    else:
        if current_index >= len(digits_to_draw):
            print("All digits have been drawn. Experiment completed.")
            reset_experiment_state()
            return end_experiment_response()
        current_digit = digits_to_draw[current_index]
        progress = f"{current_index + 1} / {len(digits_to_draw)}"
        print(f"Processing drawing for digit {current_digit} (Index: {current_index})")

    img = handle_drawing_input(drawing)
    if img is None:
        return generate_error_response("Unsupported image format.")

    img_array, img_resized = preprocess_image(img)
    if img_array is None or img_resized is None:
        return generate_error_response("Preprocessing failed. Please draw a clearer digit.")

    original_display = img.resize((200, 200))
    processed_display = img_resized.resize((200, 200))

    if is_practice:
        subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}", "Practice")
    else:
        subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")

    digit_folder = os.path.join(subject_folder, f"digit_{current_digit}")
    os.makedirs(digit_folder, exist_ok=True)
    draw_number = len([name for name in os.listdir(digit_folder) if os.path.isdir(os.path.join(digit_folder, name))]) + 1
    draw_folder = os.path.join(digit_folder, f"draw_{draw_number}")
    os.makedirs(draw_folder, exist_ok=True)

    original_file_path = os.path.join(draw_folder, "original_drawing.png")
    img.save(original_file_path)
    print(f"Original drawing saved at {original_file_path}")

    processed_file_path = os.path.join(draw_folder, "processed_drawing.png")
    img_resized.save(processed_file_path)
    print(f"Processed drawing saved at {processed_file_path}")

    if model_selection_mode == "Randomly pick one model per digit":
        prediction_text_output, plot_images = process_single_model(
            img_array, uncertainty_methods, current_digit, draw_folder
        )
    elif model_selection_mode == "Use all models for each digit":
        prediction_text_output, plot_images = process_all_models(
            img_array, uncertainty_methods, current_digit, draw_folder
        )
    else:
        return generate_error_response("Unknown model selection mode.")

    outputs = [
        gr.update(value=None),
        original_display if "Your Drawing" in content_options else gr.update(visible=False),
        processed_display if "Processed Drawing" in content_options else gr.update(visible=False),
        prediction_text_output if "Prediction Text" in content_options else gr.update(visible=False),
        plot_images if "Probabilities Plot" in content_options else gr.update(visible=False),
        gr.update(),
        gr.update(),
    ]

    if "Feedback Questions" in content_options:
        outputs.extend([
            gr.update(interactive=True, visible=True),
            gr.update(interactive=True, visible=True),
            gr.update(interactive=True, visible=True),
            gr.update(interactive=True, visible=True),
            gr.update(interactive=True, visible=True),
        ])
    else:
        outputs.extend([
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ])

    outputs.append(gr.update(interactive=True))
    return tuple(outputs)

def process_single_model(img_array, uncertainty_methods, current_digit, draw_folder):
    model_options = ["Base Model", "MC-Dropout", "Ensemble Model"]
    selected_model_name = random.choice(model_options)
    print(f"Selected model: {selected_model_name}")

    model = uncertainty_models[selected_model_name]
    predicted_digit, confidence_value, probabilities_for_plot = predict_model(
        selected_model_name, model, img_array
    )

    save_prediction(
        current_digit, selected_model_name, predicted_digit, confidence_value, draw_folder
    )

    prediction_text_output = format_prediction_text(
        selected_model_name, predicted_digit, confidence_value, uncertainty_methods, content_options_global
    )

    plot_images = generate_plots(
        probabilities_for_plot, selected_model_name, uncertainty_methods, content_options_global, draw_folder
    )

    return prediction_text_output, plot_images

def process_all_models(img_array, uncertainty_methods, current_digit, draw_folder):
    models_to_use = ["Base Model", "MC-Dropout", "Ensemble Model"]
    prediction_text_output_list = []
    plot_images = []

    save_prediction(
        current_digit, "All Models", None, None, draw_folder
    )

    for model_name in models_to_use:
        model = uncertainty_models[model_name]
        predicted_digit, confidence_value, probabilities_for_plot = predict_model(
            model_name, model, img_array
        )

        prediction_text = format_prediction_text(
            model_name, predicted_digit, confidence_value, uncertainty_methods, content_options_global
        )
        prediction_text_output_list.append(prediction_text)

        plots = generate_plots(
            probabilities_for_plot, model_name, uncertainty_methods, content_options_global, draw_folder
        )
        plot_images.extend(plots)

        append_prediction(
            model_name, predicted_digit, confidence_value, draw_folder
        )

    prediction_text_output = "\n\n".join(prediction_text_output_list)
    return prediction_text_output, plot_images

def predict_model(model_name, model, img_array):
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

def save_prediction(current_digit, model_name, predicted_digit, confidence_value, draw_folder):
    prediction_file = os.path.join(draw_folder, "prediction.txt")
    with open(prediction_file, 'w') as f:
        f.write(f"Intended Digit: {current_digit}\n")
        f.write(f"Model Used: {model_name}\n")
        if predicted_digit is not None:
            f.write(f"Predicted Digit: {predicted_digit}\n")
            f.write(f"Confidence: {confidence_value:.2f}%\n")
    print(f"Prediction saved at {prediction_file}")

def append_prediction(model_name, predicted_digit, confidence_value, draw_folder):
    prediction_file = os.path.join(draw_folder, "prediction.txt")
    if not os.path.exists(prediction_file):
        print("No prediction file found.")
        return
    with open(prediction_file, 'a') as f:
        f.write(f"\n{model_name}:\n")
        f.write(f"Predicted Digit: {predicted_digit}\n")
        f.write(f"Confidence: {confidence_value:.2f}%\n")
    print(f"Prediction appended to {prediction_file}")

def format_prediction_text(model_name, predicted_digit, confidence_value, uncertainty_methods, content_options):
    if "Confidence %" in uncertainty_methods:
        text = f"Predicted Digit: {predicted_digit}, Confidence: {confidence_value:.2f}%"
    else:
        text = f"Predicted Digit: {predicted_digit}"

    if "Show Model Name" in content_options:
        text = f"{model_name} - {text}"

    return text

def generate_plots(probabilities, model_name, uncertainty_methods, content_options, draw_folder):
    plot_images = []
    if "Bar Plot" in uncertainty_methods:
        plot_image = plot_probabilities(probabilities, model_name, content_options)
        plot_file_path = os.path.join(draw_folder, f"probabilities_plot_{model_name.replace(' ', '_')}.png")
        plot_image.save(plot_file_path)
        print(f"Probabilities plot saved at {plot_file_path}")
        plot_images.append(plot_image)
    return plot_images

def submit_feedback(q1_answer, q2_answer, q3_answer, q4_answer, q5_answer, subject_num_input):
    global digits_to_draw, current_index, practice_digits_to_draw, practice_current_index, is_practice, subject_num
    subject_num = subject_num_input

    if is_practice:
        if practice_current_index >= len(practice_digits_to_draw):
            # This should not happen, means we finished practice already
            is_practice = False

        current_digit = practice_digits_to_draw[practice_current_index]
        print(f"Submitting feedback for practice digit {current_digit} (Practice Index: {practice_current_index})")
        subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}", "Practice")
    else:
        if current_index >= len(digits_to_draw):
            # All done
            return end_experiment_response()
        current_digit = digits_to_draw[current_index]
        print(f"Submitting feedback for digit {current_digit} (Index: {current_index})")
        subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")

    digit_folder = os.path.join(subject_folder, f"digit_{current_digit}")
    draw_folders = sorted([d for d in os.listdir(digit_folder) if d.startswith("draw_")])
    if not draw_folders:
        print("No prediction file found.")
        return generate_error_response("No prediction file found.")

    last_draw_folder = draw_folders[-1]
    prediction_file = os.path.join(digit_folder, last_draw_folder, "prediction.txt")
    with open(prediction_file, 'a') as f:
        f.write(f"\nFeedback:\n")
        f.write(f"1. Is the top prediction appropriate? {q1_answer}\n")
        f.write(f"2. Is the top prediction's confidence appropriate? {q2_answer}\n")
        f.write(f"3. Are the alternative predictions appropriate? {q3_answer}\n")
        f.write(f"4. Are the alternative predictions' confidence appropriate? {q4_answer}\n")
        f.write(f"5. In relation to how clear the drawing is, is the prediction too confident? {q5_answer}\n")
    print(f"Feedback saved to {prediction_file}")

    if is_practice:
        practice_current_index += 1
        if practice_current_index >= len(practice_digits_to_draw):
            # Move to main experiment
            is_practice = False
            current_index = 0
            if current_index < len(digits_to_draw):
                next_digit = digits_to_draw[current_index]
                instruction = f"Please draw the digit {next_digit}"
                progress = f"{current_index + 1} / {len(digits_to_draw)}"
                print("Practice runs completed. Proceeding to main experiment.")
                return continue_experiment_response(instruction, progress)
            else:
                # No main digits
                return end_experiment_response()
        else:
            next_digit = practice_digits_to_draw[practice_current_index]
            instruction = f"Practice Run: Please draw the digit {next_digit}"
            progress = f"Practice Run {practice_current_index + 1} / {len(practice_digits_to_draw)}"
            print(f"Instruction updated: {instruction}")
            return continue_experiment_response(instruction, progress)
    else:
        current_index += 1
        if current_index >= len(digits_to_draw):
            print("All digits have been drawn. Experiment completed.")
            reset_experiment_state()
            return end_experiment_response()

        next_digit = digits_to_draw[current_index]
        instruction = f"Please draw the digit {next_digit}"
        progress = f"{current_index + 1} / {len(digits_to_draw)}"
        print(f"Instruction updated: {instruction}")
        return continue_experiment_response(instruction, progress)

def reset_experiment_state():
    global practice_digits_to_draw, practice_current_index, digits_to_draw, current_index, is_practice
    practice_digits_to_draw = []
    practice_current_index = 0
    digits_to_draw = []
    current_index = 0
    is_practice = True

def end_experiment_response():
    outputs = [
        gr.update(value=None),
        None if "Your Drawing" in content_options_global else gr.update(visible=False),
        None if "Processed Drawing" in content_options_global else gr.update(visible=False),
        "" if "Prediction Text" in content_options_global else gr.update(visible=False),
        None if "Probabilities Plot" in content_options_global else gr.update(visible=False),
        gr.update(value="Thank you for participating!"),
        gr.update(value=""),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(interactive=False),
        gr.update(visible=False),
        gr.update(visible=True)
    ]
    return tuple(outputs)

def continue_experiment_response(instruction, progress):
    outputs = [
        gr.update(value=None),
        None if "Your Drawing" in content_options_global else gr.update(visible=False),
        None if "Processed Drawing" in content_options_global else gr.update(visible=False),
        "" if "Prediction Text" in content_options_global else gr.update(visible=False),
        None if "Probabilities Plot" in content_options_global else gr.update(visible=False),
        gr.update(value=instruction),
        gr.update(value=progress),
    ]

    if "Feedback Questions" in content_options_global:
        outputs.extend([
            gr.update(value=None, interactive=False, visible=True),
            gr.update(value=None, interactive=False, visible=True),
            gr.update(value=None, interactive=False, visible=True),
            gr.update(value=None, interactive=False, visible=True),
            gr.update(value=None, interactive=False, visible=True),
        ])
    else:
        outputs.extend([
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ])

    outputs.append(gr.update(interactive=False))
    outputs.extend([
        gr.update(),
        gr.update()
    ])
    return tuple(outputs)