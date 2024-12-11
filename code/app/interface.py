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
    NUM_PRACTICE_DIGITS
)

# Load models once at startup
base_model = load_base_model()
mc_dropout_model = load_mc_dropout_model()
ensemble_models = load_ensemble_models()

uncertainty_models = {
    "Base Model": base_model,
    "MC-Dropout": mc_dropout_model,
    "Ensemble Model": ensemble_models
}

def log(state, message):
    subject_num = state.get("subject_num", "UNKNOWN")
    # Initialize a static variable for remembering the last subject
    if not hasattr(log, "last_subject"):
        log.last_subject = None

    # If subject changes, print blank lines for readability
    if log.last_subject is not None and log.last_subject != subject_num:
        print("\n\n", end='')  # Print two blank lines before changing subject

    print(f"[Subject {subject_num}] {message}")
    log.last_subject = subject_num

def initialize_experiment_state(state):
    chosen_digits = state["selected_digits"]
    chosen_digits_int = [int(d) for d in chosen_digits]

    if len(chosen_digits_int) == 0:
        state["practice_digits_to_draw"] = []
        state["digits_to_draw"] = []
        state["is_practice"] = False
        log(state, "No digits selected. Cannot start experiment.")
        return

    if state["skip_practice"]:
        state["practice_digits_to_draw"] = []
        state["is_practice"] = False
        log(state, "Skipping practice runs as requested.")
    else:
        random.shuffle(chosen_digits_int)
        practice = chosen_digits_int[:min(NUM_PRACTICE_DIGITS, len(chosen_digits_int))]
        state["practice_digits_to_draw"] = practice
        state["is_practice"] = len(practice) > 0
        if practice:
            log(state, f"Practice digits to draw: {practice}")
        else:
            log(state, "No practice digits chosen.")

    digits_main = chosen_digits_int.copy()
    random.shuffle(digits_main)
    state["digits_to_draw"] = digits_main
    state["practice_current_index"] = 0
    state["current_index"] = 0

    log(state, f"Main digits to draw (shuffled): {digits_main}")

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

def generate_error_response(error_message, state):
    content_options = state["content_options"]
    log(state, f"Error encountered: {error_message}")
    return (
        gr.update(value=None),
        None if "Your Drawing" in content_options else gr.update(visible=False),
        None if "Processed Drawing" in content_options else gr.update(visible=False),
        error_message if "Prediction Text" in content_options else gr.update(visible=False),
        None if "Probabilities Plot" in content_options else gr.update(visible=False),
        gr.update(),
        gr.update(),
        gr.update(value=None, visible=False, interactive=False),
        gr.update(value=None, visible=False, interactive=False),
        gr.update(value=None, visible=False, interactive=False),
        gr.update(value=None, visible=False, interactive=False),
        gr.update(value=None, visible=False, interactive=False),
        gr.update(interactive=False),
        state
    )

def home_page(subject_num_input, subject_name_input, uncertainty_methods, model_selection_mode, content_options, skip_practice, selected_digits, state):
    state["subject_num"] = subject_num_input
    state["content_options"] = content_options
    state["skip_practice"] = skip_practice
    state["selected_digits"] = selected_digits
    state["uncertainty_methods"] = uncertainty_methods
    state["model_selection_mode"] = model_selection_mode

    log(state, "Proceeding from home page...")

    subject_num = state["subject_num"]
    subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")
    os.makedirs(subject_folder, exist_ok=True)
    subject_info_file = os.path.join(subject_folder, 'subject_info.txt')
    with open(subject_info_file, 'w') as f:
        f.write(f"Subject Number: {subject_num}\n")
        f.write(f"Name: {subject_name_input}\n")

    log(state, f"Subject information saved at {subject_info_file}")

    if not subject_num or len(uncertainty_methods) == 0 or not model_selection_mode:
        log(state, "Required information not provided. Returning to home page.")
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
            gr.update(value=state)
        )

    initialize_experiment_state(state)

    if len(state["digits_to_draw"]) == 0 and not state["is_practice"]:
        # No digits to draw at all
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="No digits selected. Cannot start the experiment."),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(value=state)
        )

    if state["is_practice"] and len(state["practice_digits_to_draw"]) > 0:
        current_digit = state["practice_digits_to_draw"][state["practice_current_index"]]
        instruction = f"Practice Run: Please draw the digit {current_digit}"
        progress = f"Practice Run {state['practice_current_index'] + 1} / {len(state['practice_digits_to_draw'])}"
        log(state, f"Starting with practice run for digit {current_digit}")
    else:
        # No practice or skip practice
        if state["current_index"] < len(state["digits_to_draw"]):
            current_digit = state["digits_to_draw"][state["current_index"]]
            instruction = f"Please draw the digit {current_digit}"
            progress = f"{state['current_index'] + 1} / {len(state['digits_to_draw'])}"
            log(state, f"Starting main experiment with digit {current_digit}")
        else:
            instruction = "No digits to draw."
            progress = ""

    log(state, "Starting experiment immediately.")
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=instruction),
        gr.update(value=progress),
        gr.update(visible="Your Drawing" in content_options),
        gr.update(visible="Processed Drawing" in content_options),
        gr.update(visible="Prediction Text" in content_options),
        gr.update(visible="Probabilities Plot" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
        gr.update(visible="Feedback Questions" in content_options),
        gr.update(),
        gr.update(value=state)
    )

def process_drawing(drawing, subject_num_input, uncertainty_methods, model_selection_mode, content_options, state):
    is_practice = state["is_practice"]
    practice_digits_to_draw = state["practice_digits_to_draw"]
    practice_current_index = state["practice_current_index"]
    digits_to_draw = state["digits_to_draw"]
    current_index = state["current_index"]

    # Determine current digit
    if is_practice:
        if practice_current_index >= len(practice_digits_to_draw):
            state["is_practice"] = False
            if current_index >= len(digits_to_draw):
                log(state, "All practice done and no main digits. Ending experiment.")
                return end_experiment_response(state)
            current_digit = digits_to_draw[current_index]
            log(state, "Practice runs completed. Proceeding to main experiment.")
        else:
            current_digit = practice_digits_to_draw[practice_current_index]
            log(state, f"Processing practice drawing for digit {current_digit} (Practice Index: {practice_current_index})")
    else:
        if current_index >= len(digits_to_draw):
            log(state, "All digits have been drawn. Experiment completed.")
            return end_experiment_response(state)
        current_digit = digits_to_draw[current_index]
        log(state, f"Processing drawing for digit {current_digit} (Index: {current_index})")

    img = handle_drawing_input(drawing)
    if img is None:
        return generate_error_response("Unsupported image format.", state)

    img_array, img_resized = preprocess_image(img)
    if img_array is None or img_resized is None:
        return generate_error_response("Preprocessing failed. Please draw a clearer digit.", state)

    subject_num = state["subject_num"]
    if is_practice:
        subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}", "Practice")
    else:
        subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")

    digit_folder = os.path.join(subject_folder, f"digit_{current_digit}")
    os.makedirs(digit_folder, exist_ok=True)
    draw_folders = [d for d in os.listdir(digit_folder) if d.startswith("draw_")]
    new_draw_num = len(draw_folders) + 1
    draw_folder = os.path.join(digit_folder, f"draw_{new_draw_num}")
    os.makedirs(draw_folder, exist_ok=True)

    original_file_path = os.path.join(draw_folder, "original_drawing.png")
    img.save(original_file_path)
    log(state, f"Original drawing saved at {original_file_path}")

    processed_file_path = os.path.join(draw_folder, "processed_drawing.png")
    img_resized.save(processed_file_path)
    log(state, f"Processed drawing saved at {processed_file_path}")

    if model_selection_mode == "Randomly pick one model per digit":
        prediction_text_output, plot_images = process_single_model(img_array, uncertainty_methods, current_digit, draw_folder, state)
    else:
        prediction_text_output, plot_images = process_all_models(img_array, uncertainty_methods, current_digit, draw_folder, state)

    original_display = img.resize((200, 200))
    processed_display = img_resized.resize((200, 200))

    content_options_global = state["content_options"]
    outputs = [
        gr.update(value=None), # Clear drawing after submission
        original_display if "Your Drawing" in content_options_global else gr.update(visible=False),
        processed_display if "Processed Drawing" in content_options_global else gr.update(visible=False),
        prediction_text_output if "Prediction Text" in content_options_global else gr.update(visible=False),
        plot_images if ("Probabilities Plot" in content_options_global and plot_images) else gr.update(visible=False),
        gr.update(), # instruction_text unchanged
        gr.update(), # progress_text unchanged
    ]

    if "Feedback Questions" in content_options_global:
        outputs.extend([gr.update(interactive=True, visible=True) for _ in range(5)])
    else:
        outputs.extend([gr.update(visible=False) for _ in range(5)])

    outputs.append(gr.update(interactive=True))
    outputs.append(state)
    return tuple(outputs)

def process_single_model(img_array, uncertainty_methods, current_digit, draw_folder, state):
    model_options = ["Base Model", "MC-Dropout", "Ensemble Model"]
    selected_model_name = random.choice(model_options)
    model = uncertainty_models[selected_model_name]
    log(state, f"Selected model: {selected_model_name}")

    predicted_digit, confidence_value, probabilities_for_plot = predict_model(selected_model_name, model, img_array)
    save_prediction(current_digit, selected_model_name, predicted_digit, confidence_value, draw_folder)
    prediction_text_output = format_prediction_text(selected_model_name, predicted_digit, confidence_value, uncertainty_methods, state["content_options"])
    plot_images = generate_plots(probabilities_for_plot, selected_model_name, uncertainty_methods, state["content_options"], draw_folder)
    return prediction_text_output, plot_images

def process_all_models(img_array, uncertainty_methods, current_digit, draw_folder, state):
    models_to_use = ["Base Model", "MC-Dropout", "Ensemble Model"]
    prediction_text_output_list = []
    plot_images = []

    save_prediction(current_digit, "All Models", None, None, draw_folder)
    log(state, f"Using all models for digit {current_digit}")

    for model_name in models_to_use:
        model = uncertainty_models[model_name]
        predicted_digit, confidence_value, probabilities_for_plot = predict_model(model_name, model, img_array)
        text = format_prediction_text(model_name, predicted_digit, confidence_value, uncertainty_methods, state["content_options"])
        prediction_text_output_list.append(text)
        plots = generate_plots(probabilities_for_plot, model_name, uncertainty_methods, state["content_options"], draw_folder)
        plot_images.extend(plots)
        append_prediction(model_name, predicted_digit, confidence_value, draw_folder)

    return "\n\n".join(prediction_text_output_list), plot_images

def predict_model(model_name, model, img_array):
    predicted_labels, confidence, probabilities = None, None, None
    if model_name == "Base Model":
        predicted_labels, confidence, probabilities = predict_with_base_model(model, img_array)
    elif model_name == "MC-Dropout":
        predicted_labels, confidence, probabilities = predict_with_mc_dropout(model, img_array)
    else:
        predicted_labels, confidence, probabilities = predict_with_ensemble(model, img_array)

    return int(predicted_labels[0]), confidence[0], probabilities

def save_prediction(current_digit, model_name, predicted_digit, confidence_value, draw_folder):
    prediction_file = os.path.join(draw_folder, "prediction.txt")
    with open(prediction_file, 'w') as f:
        f.write(f"Intended Digit: {current_digit}\n")
        f.write(f"Model Used: {model_name}\n")
        if predicted_digit is not None:
            f.write(f"Predicted Digit: {predicted_digit}\n")
            f.write(f"Confidence: {confidence_value:.2f}%\n")

def append_prediction(model_name, predicted_digit, confidence_value, draw_folder):
    prediction_file = os.path.join(draw_folder, "prediction.txt")
    with open(prediction_file, 'a') as f:
        f.write(f"\n{model_name}:\nPredicted Digit: {predicted_digit}\nConfidence: {confidence_value:.2f}%\n")

def format_prediction_text(model_name, predicted_digit, confidence_value, uncertainty_methods, content_options):
    text = f"Predicted Digit: {predicted_digit}"
    if "Confidence %" in uncertainty_methods:
        text += f", Confidence: {confidence_value:.2f}%"
    if "Show Model Name" in content_options:
        text = f"{model_name} - {text}"
    return text

def generate_plots(probabilities, model_name, uncertainty_methods, content_options, draw_folder):
    if "Bar Plot" in uncertainty_methods:
        plot_image = plot_probabilities(probabilities, model_name, content_options)
        plot_file_path = os.path.join(draw_folder, f"probabilities_plot_{model_name.replace(' ', '_')}.png")
        plot_image.save(plot_file_path)
        return [plot_image]
    return []

def submit_feedback(q1_answer, q2_answer, q3_answer, q4_answer, q5_answer, subject_num_input, state):
    is_practice = state["is_practice"]
    practice_digits_to_draw = state["practice_digits_to_draw"]
    practice_current_index = state["practice_current_index"]
    digits_to_draw = state["digits_to_draw"]
    current_index = state["current_index"]
    subject_num = state["subject_num"]

    if is_practice:
        if practice_current_index >= len(practice_digits_to_draw):
            state["is_practice"] = False
        current_digit = practice_digits_to_draw[practice_current_index] if practice_current_index < len(practice_digits_to_draw) else None
        if current_digit is not None:
            log(state, f"Submitting feedback for practice digit {current_digit} (Practice Index: {practice_current_index})")
            subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}", "Practice")
        else:
            subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}", "Practice")
    else:
        if current_index >= len(digits_to_draw):
            log(state, "All digits have been drawn. Ending experiment after feedback.")
            return end_experiment_response(state)
        current_digit = digits_to_draw[current_index]
        log(state, f"Submitting feedback for digit {current_digit} (Index: {current_index})")
        subject_folder = os.path.join(BASE_RESULTS_DIR, f"Subject_{subject_num}")

    if current_digit is not None:
        digit_folder = os.path.join(subject_folder, f"digit_{current_digit}")
        draw_folders = sorted([d for d in os.listdir(digit_folder) if d.startswith("draw_")])
        if not draw_folders:
            return generate_error_response("No prediction file found.", state)
        last_draw_folder = draw_folders[-1]
        prediction_file = os.path.join(digit_folder, last_draw_folder, "prediction.txt")
        with open(prediction_file, 'a') as f:
            f.write(f"\nFeedback:\n")
            f.write(f"1. {q1_answer}\n")
            f.write(f"2. {q2_answer}\n")
            f.write(f"3. {q3_answer}\n")
            f.write(f"4. {q4_answer}\n")
            f.write(f"5. {q5_answer}\n")

    # Move to next digit
    if is_practice:
        state["practice_current_index"] += 1
        if state["practice_current_index"] >= len(practice_digits_to_draw):
            state["is_practice"] = False
            state["current_index"] = 0
            if state["current_index"] < len(digits_to_draw):
                next_digit = digits_to_draw[state["current_index"]]
                instruction = f"Please draw the digit {next_digit}"
                progress = f"{state['current_index'] + 1} / {len(digits_to_draw)}"
                log(state, "Practice runs completed. Proceeding to main experiment.")
                return continue_experiment_response(instruction, progress, state)
            else:
                return end_experiment_response(state)
        else:
            next_digit = practice_digits_to_draw[state["practice_current_index"]]
            instruction = f"Practice Run: Please draw the digit {next_digit}"
            progress = f"Practice Run {state['practice_current_index'] + 1} / {len(practice_digits_to_draw)}"
            log(state, f"Instruction updated: {instruction}")
            return continue_experiment_response(instruction, progress, state)
    else:
        state["current_index"] += 1
        if state["current_index"] >= len(digits_to_draw):
            log(state, "All digits have been drawn. Experiment completed.")
            return end_experiment_response(state)

        next_digit = digits_to_draw[state["current_index"]]
        instruction = f"Please draw the digit {next_digit}"
        progress = f"{state['current_index'] + 1} / {len(digits_to_draw)}"
        log(state, f"Instruction updated: {instruction}")
        return continue_experiment_response(instruction, progress, state)

def end_experiment_response(state):
    content_options = state["content_options"]
    log(state, "Ending experiment and showing thank you page.")
    return (
        gr.update(value=None),
        None if "Your Drawing" in content_options else gr.update(visible=False),
        None if "Processed Drawing" in content_options else gr.update(visible=False),
        "" if "Prediction Text" in content_options else gr.update(visible=False),
        None if "Probabilities Plot" in content_options else gr.update(visible=False),
        gr.update(value="Thank you for participating!"),
        gr.update(value=""),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(value=None, interactive=False, visible=False),
        gr.update(interactive=False),
        gr.update(visible=False),
        gr.update(visible=True),
        state
    )

def continue_experiment_response(instruction, progress, state):
    content_options = state["content_options"]
    log(state, f"Continuing experiment: {instruction}, progress: {progress}")
    return (
        gr.update(value=None),
        None if "Your Drawing" in content_options else gr.update(visible=False),
        None if "Processed Drawing" in content_options else gr.update(visible=False),
        "" if "Prediction Text" in content_options else gr.update(visible=False),
        None if "Probabilities Plot" in content_options else gr.update(visible=False),
        gr.update(value=instruction),
        gr.update(value=progress),
        *(gr.update(value=None, interactive=False, visible=True) if "Feedback Questions" in content_options else gr.update(visible=False) for _ in range(5)),
        gr.update(interactive=False),
        gr.update(),
        gr.update(),
        state
    )