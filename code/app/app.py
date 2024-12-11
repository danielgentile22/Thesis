import gradio as gr
from interface import (
    process_drawing,
    submit_feedback,
    home_page
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

with gr.Blocks() as demo:
    gr.Markdown("# Handwritten Digit Recognition with Uncertainty Visualization")

    # Global per-user state
    session_state = gr.State({
        "practice_digits_to_draw": [],
        "practice_current_index": 0,
        "digits_to_draw": [],
        "current_index": 0,
        "is_practice": True,
        "content_options": [],
        "subject_num": None,
        "skip_practice": False,
        "selected_digits": []
    })

    # Home Page
    with gr.Column(visible=True) as home_page_container:
        subject_num = gr.Textbox(label="Subject Number")
        subject_name = gr.Textbox(label="Name")
        uncertainty_methods = gr.CheckboxGroup(
            choices=["Confidence %", "Bar Plot"],
            label="Select Uncertainty Methods",
            value=["Confidence %", "Bar Plot"]
        )
        model_selection_mode = gr.Radio(
            choices=["Randomly pick one model per digit", "Use all models for each digit"],
            label="Model Selection Mode",
            value="Randomly pick one model per digit"
        )
        skip_practice = gr.Checkbox(label="Skip Practice Runs?", value=False)
        digit_choices = [str(i) for i in range(10)]
        selected_digits = gr.CheckboxGroup(
            choices=digit_choices,
            label="Select Which Digits to Draw",
            value=digit_choices
        )

        content_options = gr.CheckboxGroup(
            choices=[
                "Your Drawing",
                "Processed Drawing",
                "Prediction Text",
                "Probabilities Plot",
                "Feedback Questions",
                "Show Model Name"
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
        custom_brush = gr.Brush(
            default_size=BRUSH_DEFAULT_SIZE,
            colors=BRUSH_COLORS,
            default_color=BRUSH_DEFAULT_COLOR,
            color_mode=BRUSH_COLOR_MODE
        )
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

        original_drawing_display = gr.Image(label="Your Drawing", visible=False)
        processed_drawing_display = gr.Image(label="Processed Drawing (28x28)", visible=False)
        prediction_text = gr.Textbox(label="Prediction", interactive=False, visible=False)
        probabilities_plot = gr.Gallery(label="Prediction Plot", visible=False)

        feedback_instruction = gr.Markdown(
            "Evaluate the model with some sympathy...",
            visible=False
        )

        likert_choices = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree", "Can't answer"]

        q1 = gr.Radio(choices=likert_choices, label="1. Is the top prediction appropriate?", interactive=False, visible=False)
        q2 = gr.Radio(choices=likert_choices, label="2. Is the top prediction's confidence appropriate?", interactive=False, visible=False)
        q3 = gr.Radio(choices=likert_choices, label="3. Are the alternative predictions appropriate?", interactive=False, visible=False)
        q4 = gr.Radio(choices=likert_choices, label="4. Are the alternative predictions' confidences appropriate?", interactive=False, visible=False)
        q5 = gr.Radio(choices=likert_choices, label="5. In relation to how clear the drawing is, is the prediction too confident?", interactive=False, visible=False)

        next_digit_button = gr.Button("Next Digit", interactive=False)

    # Thank You Page
    with gr.Column(visible=False) as thank_you_page_container:
        gr.Markdown("Thank you for participating in the experiment!")

    proceed_button.click(
        home_page,
        inputs=[subject_num, subject_name, uncertainty_methods, model_selection_mode, content_options, skip_practice, selected_digits, session_state],
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
            session_state
        ]
    )

    submit_drawing_button.click(
        process_drawing,
        inputs=[drawing, subject_num, uncertainty_methods, model_selection_mode, content_options, session_state],
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
            session_state
        ]
    )

    next_digit_button.click(
        submit_feedback,
        inputs=[q1, q2, q3, q4, q5, subject_num, session_state],
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
            thank_you_page_container,
            session_state
        ]
    )

demo.queue().launch(share=True)