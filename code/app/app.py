import gradio as gr
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import random

# Function to load the base model
def load_base_model(model_path="../../trained_models/base_model.keras"):
    print("Loading Base model...")
    model = load_model(model_path)
    print("Base model loaded.")
    return model

# Function to load the MC-Dropout model
def load_mc_dropout_model(model_path="../../trained_models/dropout_model.keras"):
    print("Loading MC-Dropout model...")
    model = load_model(model_path)
    print("MC-Dropout model loaded.")
    return model

# Function to load ensemble models
def load_ensemble_models(model_path_prefix="../../trained_models/ensemble_model", ensemble_size=5):
    print("Loading Ensemble models...")
    ensemble = []
    for i in range(ensemble_size):
        model_path = f"{model_path_prefix}_{i+1}.keras"
        model = load_model(model_path)
        ensemble.append(model)
    print("Ensemble models loaded.")
    return ensemble

# Prediction functions for each model type
def predict_with_base_model(model, x):
    probabilities = model.predict(x)
    predicted_labels = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1) * 100  # Convert to percentage
    return predicted_labels, confidence, probabilities

def predict_with_mc_dropout(model, x, num_samples=100):
    predictions = np.stack([model(x, training=True) for _ in range(num_samples)])
    mean_predictions = np.mean(predictions, axis=0)
    confidence = np.max(mean_predictions, axis=1) * 100  # Convert to percentage
    predicted_labels = np.argmax(mean_predictions, axis=1)
    return predicted_labels, confidence, mean_predictions

def predict_with_ensemble(ensemble, x):
    ensemble_predictions = [model.predict(x) for model in ensemble]
    mean_predictions = np.mean(ensemble_predictions, axis=0)
    predicted_labels = np.argmax(mean_predictions, axis=1)
    confidence = np.max(mean_predictions, axis=1) * 100  # Convert to percentage
    return predicted_labels, confidence, mean_predictions

# Function to plot probabilities as a bar chart
def plot_probabilities(probabilities, model_name):
    import matplotlib.pyplot as plt
    import io
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

# Load the models
base_model = load_base_model()
mc_dropout_model = load_mc_dropout_model()
ensemble_models = load_ensemble_models()

# Dictionary mapping model names to models
uncertainty_models = {
    "Base Model": base_model,
    "MC-Dropout": mc_dropout_model,
    "Ensemble Model": ensemble_models
}

# Global variables to track the state
current_digit = 0
draw_count = 0
max_draw_per_digit = 2
digits_drawn = 0  # Total digits drawn

def preprocess_image(img):
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
    uncertainty_methods,
    model_selection_mode  # New parameter
):
    global current_digit, draw_count, max_draw_per_digit, digits_drawn

    print("Processing drawing...")

    # Create the base directory for experiment results
    base_results_dir = "../../exp_results/"
    os.makedirs(base_results_dir, exist_ok=True)

    # Create a folder for the subject
    subject_folder = os.path.join(base_results_dir, f"Subject_{subject_num}")
    os.makedirs(subject_folder, exist_ok=True)

    # Create a folder for the current digit
    digit_folder = os.path.join(subject_folder, f"digit_{current_digit}")
    os.makedirs(digit_folder, exist_ok=True)

    # Create a folder for the current drawing attempt
    draw_folder = os.path.join(digit_folder, f"draw_{draw_count + 1}")
    os.makedirs(draw_folder, exist_ok=True)

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
        # Randomly select a model
        model_options = ["Base Model", "MC-Dropout", "Ensemble Model"]
        selected_model_name = random.choice(model_options)
        print(f"Selected model: {selected_model_name}")

        # Use the selected model for prediction
        if selected_model_name == "Base Model":
            model = uncertainty_models["Base Model"]
            predicted_labels, confidence, probabilities = predict_with_base_model(model, img_array)
            predicted_digit = predicted_labels[0]
            confidence_value = confidence[0]
            probabilities_for_plot = probabilities
        elif selected_model_name == "MC-Dropout":
            model = uncertainty_models["MC-Dropout"]
            predicted_labels, confidence, mean_predictions = predict_with_mc_dropout(model, img_array, num_samples=100)
            predicted_digit = predicted_labels[0]
            confidence_value = confidence[0]
            probabilities_for_plot = mean_predictions
        elif selected_model_name == "Ensemble Model":
            ensemble = uncertainty_models["Ensemble Model"]
            predicted_labels, confidence, mean_predictions = predict_with_ensemble(ensemble, img_array)
            predicted_digit = predicted_labels[0]
            confidence_value = confidence[0]
            probabilities_for_plot = mean_predictions
        else:
            print("Unknown model selected.")
            return (
                gr.update(),
                None,
                None,
                "Error: Unknown model selected.",
                None,
                gr.update(),
                gr.update(),
                gr.update()
            )

        print(f"Predicted Digit: {predicted_digit}, Confidence: {confidence_value:.2f}%")

        # Save prediction and uncertainty
        prediction_file = os.path.join(draw_folder, "prediction.txt")
        with open(prediction_file, 'w') as f:
            f.write(f"Intended Digit: {current_digit}\n")
            f.write(f"Model Used: {selected_model_name}\n")
            f.write(f"Predicted Digit: {predicted_digit}\n")
            f.write(f"Confidence: {confidence_value:.2f}%\n")
            # Feedback will be saved later

        print(f"Prediction saved at {prediction_file}")

        # Prepare prediction text
        prediction_text_output = (
            f"Model Used: {selected_model_name}\nPredicted Digit: {predicted_digit}, Confidence: {confidence_value:.2f}%"
        ) if "Confidence %" in uncertainty_methods else f"Model Used: {selected_model_name}\nPredicted Digit: {predicted_digit}"

        # Generate the bar plot if "Bar Plot" is selected
        if "Bar Plot" in uncertainty_methods:
            plot_image = plot_probabilities(probabilities_for_plot, selected_model_name)
            # Save the plot
            plot_file_path = os.path.join(draw_folder, f"probabilities_plot_{selected_model_name.replace(' ', '_')}.png")
            plot_image.save(plot_file_path)
            print(f"Probabilities plot saved at {plot_file_path}")
            plot_images = [plot_image]  # Wrap in a list for the gallery
        else:
            plot_images = []

    elif model_selection_mode == "Use all models for each digit":
        print("Using all models for prediction.")
        models_to_use = ["Base Model", "MC-Dropout", "Ensemble Model"]
        prediction_text_output_list = []

        # Prepare prediction file
        prediction_file = os.path.join(draw_folder, "prediction.txt")
        with open(prediction_file, 'w') as f:
            f.write(f"Intended Digit: {current_digit}\n")
            f.write(f"Models Used: All Models\n")

            for model_name in models_to_use:
                print(f"Processing with {model_name}")
                if model_name == "Base Model":
                    model = uncertainty_models["Base Model"]
                    predicted_labels, confidence, probabilities = predict_with_base_model(model, img_array)
                    predicted_digit = predicted_labels[0]
                    confidence_value = confidence[0]
                    probabilities_for_plot = probabilities
                elif model_name == "MC-Dropout":
                    model = uncertainty_models["MC-Dropout"]
                    predicted_labels, confidence, mean_predictions = predict_with_mc_dropout(model, img_array, num_samples=100)
                    predicted_digit = predicted_labels[0]
                    confidence_value = confidence[0]
                    probabilities_for_plot = mean_predictions
                elif model_name == "Ensemble Model":
                    ensemble = uncertainty_models["Ensemble Model"]
                    predicted_labels, confidence, mean_predictions = predict_with_ensemble(ensemble, img_array)
                    predicted_digit = predicted_labels[0]
                    confidence_value = confidence[0]
                    probabilities_for_plot = mean_predictions
                else:
                    continue

                # Append to prediction text
                prediction_text = (
                    f"{model_name} - Predicted Digit: {predicted_digit}, Confidence: {confidence_value:.2f}%"
                ) if "Confidence %" in uncertainty_methods else f"{model_name} - Predicted Digit: {predicted_digit}"
                prediction_text_output_list.append(prediction_text)

                # Save to prediction file
                f.write(f"\n{model_name}:\n")
                f.write(f"Predicted Digit: {predicted_digit}\n")
                f.write(f"Confidence: {confidence_value:.2f}%\n")

                # Generate bar plot if selected
                if "Bar Plot" in uncertainty_methods:
                    plot_image = plot_probabilities(probabilities_for_plot, model_name)
                    # Save the plot
                    plot_file_path = os.path.join(draw_folder, f"probabilities_plot_{model_name.replace(' ', '_')}.png")
                    plot_image.save(plot_file_path)
                    print(f"Probabilities plot for {model_name} saved at {plot_file_path}")
                    plot_images.append(plot_image)

        # Combine prediction texts
        prediction_text_output = "\n\n".join(prediction_text_output_list)

    else:
        print("Unknown model selection mode.")
        return (
            gr.update(),
            None,
            None,
            "Error: Unknown model selection mode.",
            None,
            gr.update(),
            gr.update(),
            gr.update()
        )

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

def submit_feedback(
    feedback,
    subject_num
):
    global current_digit, draw_count, max_draw_per_digit, digits_drawn

    print("Submitting feedback...")

    # Locate the prediction file
    base_results_dir = "../../exp_results/"
    prediction_file = os.path.join(
        base_results_dir,
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

def home_page(subject_num, uncertainty_methods, model_selection_mode):
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
        model_selection_mode = gr.Radio(
            choices=["Randomly pick one model per digit", "Use all models for each digit"],
            label="Model Selection Mode",
            value="Randomly pick one model per digit"  # Default value
        )
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
        probabilities_plot = gr.Gallery(label="Prediction Probabilities")  # Changed to Gallery
        feedback_text = gr.Textbox(label="Feedback on the Prediction", placeholder="Enter your feedback here...", interactive=False)
        next_digit_button = gr.Button("Next Digit", interactive=False)

    # Thank You Page
    with gr.Column(visible=False) as thank_you_page_container:
        gr.Markdown("Thank you for participating in the experiment!")

    # Home Page Button Click
    proceed_button.click(
        home_page,
        inputs=[subject_num, uncertainty_methods, model_selection_mode],
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
        inputs=[drawing, subject_num, uncertainty_methods, model_selection_mode],
        outputs=[
            drawing,                  # Keep drawing as is
            original_drawing_display, # Display original drawing
            processed_drawing_display,# Display processed drawing
            prediction_text,          # Update prediction_text
            probabilities_plot,       # Display probabilities_plot(s)
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