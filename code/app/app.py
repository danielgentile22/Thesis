# app.py

import gradio as gr
from interface import (
    process_drawing,
    submit_feedback,
    home_page,
    consent_page,
)
from config import MAX_DRAW_PER_DIGIT

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Handwritten Digit Recognition with Uncertainty Visualization")

    # Home Page
    with gr.Column(visible=True) as home_page_container:
        subject_num = gr.Textbox(label="Subject Number")
        uncertainty_methods = gr.CheckboxGroup(
            choices=["Confidence %", "Bar Plot"],
            label="Select Uncertainty Methods",
            value=["Confidence %", "Bar Plot"]  # Default to both options selected
        )
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
        probabilities_plot = gr.Gallery(label="Prediction Probabilities")
        
        # New Feedback Section with Likert Scale Questions
        feedback_instruction = gr.Markdown(
            "Evaluate the model with some sympathy. At times it will make mistakes if not too confident or it might be uncertain about a correct prediction if it is a difficult one. Please answer the following questions with this in mind."
        )
        q1 = gr.Radio(
            choices=["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
            label="1. Is the top prediction appropriate?",
            interactive=False
        )
        q2 = gr.Radio(
            choices=["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
            label="2. Are the alternative predictions appropriate?",
            interactive=False
        )
        q3 = gr.Radio(
            choices=["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
            label="3. In relation to how clear the drawing is, is the prediction too confident?",
            interactive=False
        )
        next_digit_button = gr.Button("Next Digit", interactive=False)

    # Thank You Page
    with gr.Column(visible=False) as thank_you_page_container:
        gr.Markdown("Thank you for participating in the experiment!")

    # Home Page Button Click
    proceed_button.click(
        home_page,
        inputs=[subject_num, uncertainty_methods, model_selection_mode],
        outputs=[
            home_page_container,
            consent_page_container,
            experiment_page_container,
            thank_you_page_container
        ]
    )

    # Consent Page Button Click
    start_experiment_button.click(
        consent_page,
        inputs=[agree_checkbox],
        outputs=[
            consent_page_container,
            experiment_page_container,
            instruction_text
        ]
    )

    # Submit Drawing Button Click
    submit_drawing_button.click(
        process_drawing,
        inputs=[
            drawing,
            subject_num,
            uncertainty_methods,
            model_selection_mode
        ],
        outputs=[
            drawing,                      # Keep drawing as is
            original_drawing_display,     # Display original drawing
            processed_drawing_display,    # Display processed drawing
            prediction_text,              # Update prediction_text
            probabilities_plot,           # Display probabilities_plot(s)
            instruction_text,             # Keep instruction_text
            q1,                           # Enable q1
            q2,                           # Enable q2
            q3,                           # Enable q3
            next_digit_button             # Enable next_digit_button
        ]
    )

    # Next Digit Button Click
    next_digit_button.click(
        submit_feedback,
        inputs=[
            q1,
            q2,
            q3,
            subject_num
        ],
        outputs=[
            drawing,                    # Clear drawing
            original_drawing_display,   # Clear original drawing display
            processed_drawing_display,  # Clear processed drawing display
            prediction_text,            # Clear prediction_text
            probabilities_plot,         # Clear probabilities_plot
            instruction_text,           # Update instruction_text
            q1,                         # Clear and disable q1
            q2,                         # Clear and disable q2
            q3,                         # Clear and disable q3
            next_digit_button,          # Disable next_digit_button
            experiment_page_container,  # Show/hide experiment page
            thank_you_page_container    # Show/hide thank you page
        ]
    )

# demo.launch(share=True)
demo.launch()