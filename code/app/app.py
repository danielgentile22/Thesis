# app.py

import gradio as gr
from interface import (
    process_drawing,
    submit_feedback,
    home_page,
)
from config import (
    MAX_DRAW_PER_DIGIT,
    BRUSH_DEFAULT_SIZE,
    BRUSH_COLORS,
    BRUSH_DEFAULT_COLOR,
    BRUSH_COLOR_MODE,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    CANVAS_SIZE
)

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Handwritten Digit Recognition with Uncertainty Visualization")

    # Home Page
    with gr.Column(visible=True) as home_page_container:
        subject_num = gr.Textbox(label="Subject Number")
        subject_name = gr.Textbox(label="Name")
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
        # New options to include/exclude content boxes
        content_options = gr.CheckboxGroup(
            choices=[
                "Your Drawing",
                "Processed Drawing",
                "Prediction Text",
                "Probabilities Plot",
                "Feedback Questions",
                "Show Model Name"  # New option added
            ],
            label="Select Content to Display",
            value=[
                "Processed Drawing",
                "Prediction Text",
                "Probabilities Plot",
                "Feedback Questions"
            ]
        )
        proceed_button = gr.Button("Start Experiment")

    # Experiment Page
    with gr.Column(visible=False) as experiment_page_container:
        with gr.Row():
            instruction_text = gr.Textbox(label="Instructions", interactive=False)
            progress_text = gr.Markdown(value="", visible=True)
        # Create a Brush instance with desired settings
        custom_brush = gr.Brush(
            default_size=BRUSH_DEFAULT_SIZE,
            colors=BRUSH_COLORS,
            default_color=BRUSH_DEFAULT_COLOR,
            color_mode=BRUSH_COLOR_MODE
        )
        # Use gr.ImageEditor with the brush parameter
        drawing = gr.ImageEditor(
            label="Draw a Digit",
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            canvas_size=CANVAS_SIZE,
            sources=(),
            show_download_button=False,
            brush=custom_brush,
        )
        submit_drawing_button = gr.Button("Submit Drawing")
        # Content boxes that can be included or excluded
        original_drawing_display = gr.Image(label="Your Drawing", visible=False)
        processed_drawing_display = gr.Image(label="Processed Drawing (28x28)", visible=False)
        prediction_text = gr.Textbox(label="Prediction", interactive=False, visible=False)
        probabilities_plot = gr.Gallery(label="Prediction Plot", visible=False)

        # Feedback Questions (Likert Scale)
        feedback_instruction = gr.Markdown(
            "Evaluate the model with some sympathy. At times it will make mistakes if not too confident or it might be uncertain about a correct prediction if it is a difficult one. Please answer the following questions with this in mind.",
            visible=False
        )

        likert_choices = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree", "Can't answer"]

        q1 = gr.Radio(
            choices=likert_choices,
            label="1. Is the top prediction appropriate?",
            interactive=False,
            visible=False
        )
        q2 = gr.Radio(
            choices=likert_choices,
            label="2. Is the top prediction's confidence appropriate?",
            interactive=False,
            visible=False
        )
        q3 = gr.Radio(
            choices=likert_choices,
            label="3. Are the alternative predictions appropriate?",
            interactive=False,
            visible=False
        )
        q4 = gr.Radio(
            choices=likert_choices,
            label="4. Are the alternative predictions' confidence appropriate?",
            interactive=False,
            visible=False
        )
        q5 = gr.Radio(
            choices=likert_choices,
            label="5. In relation to how clear the drawing is, is the prediction too confident?",
            interactive=False,
            visible=False
        )
        next_digit_button = gr.Button("Next Digit", interactive=False)

    # Thank You Page
    with gr.Column(visible=False) as thank_you_page_container:
        gr.Markdown("Thank you for participating in the experiment!")

    # Event Handlers
    # Home Page Button Click
    proceed_button.click(
        home_page,
        inputs=[subject_num, subject_name, uncertainty_methods, model_selection_mode, content_options],
        outputs=[
            home_page_container,
            experiment_page_container,
            instruction_text,
            progress_text,
            original_drawing_display,
            processed_drawing_display,
            prediction_text,
            probabilities_plot,
            feedback_instruction,
            q1,
            q2,
            q3,
            q4,
            q5,
        ]
    )

    # Submit Drawing Button Click
    submit_drawing_button.click(
        process_drawing,
        inputs=[
            drawing,
            subject_num,
            uncertainty_methods,
            model_selection_mode,
            content_options
        ],
        outputs=[
            drawing,
            original_drawing_display,
            processed_drawing_display,
            prediction_text,
            probabilities_plot,
            instruction_text,
            progress_text,
            q1,
            q2,
            q3,
            q4,
            q5,
            next_digit_button
        ]
    )

    # Next Digit Button Click
    next_digit_button.click(
        submit_feedback,
        inputs=[
            q1,
            q2,
            q3,
            q4,
            q5,
            subject_num
        ],
        outputs=[
            drawing,
            original_drawing_display,
            processed_drawing_display,
            prediction_text,
            probabilities_plot,
            instruction_text,
            progress_text,
            q1,
            q2,
            q3,
            q4,
            q5,
            next_digit_button,
            experiment_page_container,
            thank_you_page_container
        ]
    )

# Launch the Gradio app
demo.launch(share=True)
# demo.launch(server_name="145.90.176.189", server_port=7860)