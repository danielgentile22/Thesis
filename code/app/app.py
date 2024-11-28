# app.py

import gradio as gr
from interface import (
    process_drawing,
    submit_feedback,
    home_page,
    consent_page,
    instructions_page,
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
        proceed_button = gr.Button("Proceed to Consent")

    # Consent Page
    with gr.Column(visible=False) as consent_page_container:
        gr.Markdown("This is an experiment. Do you agree to participate?")
        agree_checkbox = gr.Checkbox(label="I agree to participate in this experiment.")
        start_experiment_button = gr.Button("Proceed to Instructions")

    # Instructions Page
    with gr.Column(visible=False) as instructions_page_container:
        gr.Markdown("# Experiment Instructions")
        gr.Markdown("""
        ### 1. Number to Draw            
        At the top, you will receive instructions on which number to draw, as shown below:
        """)
        number_instructions_image = gr.Image(value="../../images/number_instruction.png", interactive=False, width=1500)
        
        gr.Markdown("""
        ### 2. How to Start Drawing
        Move to the canvas and start your drawing.\\
        The pen tool will be selected by default.\\
        Please use as much of the canvas as possible without altering the way you would want to draw the requested digit.
        """)
        drawing_instructions_image = gr.Image(value="../../images/draw_button.png", interactive=False, width=600)
        
        gr.Markdown("""
        ### 3. Brush Width
        The brush width is set to an appropriate size by default.\\
        You do not need to adjust it unless you wish to.
        """)
        brush_width_image = gr.Image(value="../../images/brush_size.png", interactive=False, width=600)
        
        gr.Markdown("""
        ### 4. Clearing the Canvas
        It is possible to clear the canvas.\\
        This can be done when the drawing is not satisfactory or when the previous drawing is still present.
        """)
        clear_canvas_image = gr.Image(value="../../images/clear_canvas.png", interactive=False, width=600)
        
        gr.Markdown("""
        ### 5. Submitting Drawing
        When done drawing, click "Submit Drawing".\\
        This will submit your drawing for processing. Please leave the drawing alone once this is done.
        """)
        submit_drawing_image = gr.Image(value="../../images/submit_drawing.png", interactive=False, width=600)
        
        gr.Markdown("""
        ### 6. Prediction Information
        After submitting your drawing you will be provided with the prediction and relevant information. \\
        1. Processed drawing: This is your drawing once it has been processed. If you need to refer to your drawing while on this page please refer to this image instead of your original drawing. \\
        2. Prediction: Here you will see the predicted digit and a value for confidence level. These will be needed when answering the questions. \\
        3. Prediction Plot: A bar plot of all the confidence levels for each possible digit. This will be used to evaluate the alternate predictions of the model.            
        """)
        prediction_info_image = gr.Image(value="../../images/prediction_info.png", interactive=False, width=600)
        
        gr.Markdown("""
        ### 7. Accessing Plot
        As can be seen in the image above the bar plot is hard to see initially.\\
        Please click the plot once to fix the sizing of it. If done correctly it will then look as follows:
        """)
        fixed_plot_image = gr.Image(value="../../images/fixed_plot.png", interactive=False, width=600)
        
        gr.Markdown("""
        ### 8. Questions Section
        Finally there are five questions to answer.\\
        Please answer these to the best of your ability. If you have any questions feel free to ask for help.\\
        The possible answers are as follows:\\
        - Strongly Disagree: Used when you disagree with the statement with near certainty.\\
        - Disagree: Used when you disagree with the statement with some uncertainty.\\
        - Neutral: Used when uncertain of whether you agree or disagree with the statement.\\
        - Agree: Used when you agree with the statement with some uncertainty.\\
        - Strongly Agree: Used when you agree with the statement with near certainty.\\
        - Can't Answer: Used when you cannot answer the question.
        """)
        question_section_image = gr.Image(value="../../images/question_section.png", interactive=False, width=600)
        
        gr.Markdown("""
        ### 9. Next Digit
        When done answering the questions please click the "Next Digit" button to clear all the results and move to the next digit.
        """)
        next_digit_image = gr.Image(value="../../images/next_digit.png", interactive=False, width=600)
        
        gr.Markdown("""
        ### 10. General Information
        You will be asked to draw each digit once for a total of 10 drawings after 3 practice runs.\\
        If any doubts or questions arise at any moment during the experiment please ask for clarification.\\
        There are no "correct" ways to draw or answer the questions. Please do your best with the drawings and be as accurate and honest as you can with the questions.\\\\
        ## Thank you and good luck!
        """)
        proceed_to_experiment_button = gr.Button("Start Experiment")

    # Experiment Page
    with gr.Column(visible=False) as experiment_page_container:
        with gr.Row():
            instruction_text = gr.Textbox(label="Instructions", interactive=False)
            progress_text = gr.Markdown(value="", visible=True)
        # Create a Brush instance with desired settings
        custom_brush = gr.Brush(
            default_size=BRUSH_DEFAULT_SIZE,      # Use constant from config.py
            colors=BRUSH_COLORS,                  # Use constant from config.py
            default_color=BRUSH_DEFAULT_COLOR,    # Use constant from config.py
            color_mode=BRUSH_COLOR_MODE           # Use constant from config.py
        )
        # Use gr.ImageEditor with the brush parameter
        drawing = gr.ImageEditor(
            label="Draw a Digit",
            height=CANVAS_HEIGHT,         # Use constant from config.py
            width=CANVAS_WIDTH,           # Use constant from config.py
            canvas_size=CANVAS_SIZE,      # Use constant from config.py
            sources=(),                   # Hide image features
            show_download_button=False,   # Hide download button
            brush=custom_brush,           # Set custom brush settings
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

        # Updated choices with "Can't answer"
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
        inputs=[subject_num, uncertainty_methods, model_selection_mode, content_options],
        outputs=[
            home_page_container,
            consent_page_container,
            instructions_page_container,
            experiment_page_container,
            thank_you_page_container,
            # Passing content_options to interface.py
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

    # Consent Page Button Click
    start_experiment_button.click(
        consent_page,
        inputs=[agree_checkbox],
        outputs=[
            consent_page_container,
            instructions_page_container
        ]
    )

    # Instructions Page Button Click
    proceed_to_experiment_button.click(
        instructions_page,
        outputs=[
            instructions_page_container,
            experiment_page_container,
            instruction_text,
            progress_text
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
            drawing,                      # Keep drawing as is
            original_drawing_display,     # Display original drawing
            processed_drawing_display,    # Display processed drawing
            prediction_text,              # Update prediction_text
            probabilities_plot,           # Display probabilities_plot(s)
            instruction_text,             # Keep instruction_text
            progress_text,                # Update progress_text
            q1,                           # Enable q1
            q2,                           # Enable q2
            q3,                           # Enable q3
            q4,                           # Enable q4
            q5,                           # Enable q5
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
            q4,
            q5,
            subject_num
        ],
        outputs=[
            drawing,                    # Clear drawing
            original_drawing_display,   # Clear original drawing display
            processed_drawing_display,  # Clear processed drawing display
            prediction_text,            # Clear prediction_text
            probabilities_plot,         # Clear probabilities_plot
            instruction_text,           # Update instruction_text
            progress_text,              # Update progress_text
            q1,                         # Clear and disable q1
            q2,                         # Clear and disable q2
            q3,                         # Clear and disable q3
            q4,                         # Clear and disable q4
            q5,                         # Clear and disable q5
            next_digit_button,          # Disable next_digit_button
            experiment_page_container,  # Show/hide experiment page
            thank_you_page_container    # Show/hide thank you page
        ]
    )

# Launch the Gradio app
demo.launch(server_name="145.90.176.189", server_port=7860)