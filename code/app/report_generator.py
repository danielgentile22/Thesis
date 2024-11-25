import os
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import io

def parse_prediction_file(file_path):
    data = {}
    feedback_data = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Intended Digit:"):
            data['Intended Digit'] = int(line.split(":")[1].strip())
        elif line.startswith("Model Used:"):
            data['Model Used'] = line.split(":")[1].strip()
        elif line.startswith("Predicted Digit:"):
            data['Predicted Digit'] = int(line.split(":")[1].strip())
        elif line.startswith("Confidence:"):
            conf_str = line.split(":")[1].strip().replace('%','')
            try:
                data['Confidence'] = float(conf_str)
            except ValueError:
                data['Confidence'] = None
        elif line == "Feedback:":
            # Read the next three lines for feedback questions
            q1_line = lines[idx+1].strip()
            q2_line = lines[idx+2].strip()
            q3_line = lines[idx+3].strip()

            if q1_line.startswith("1."):
                feedback_data['Question 1'] = q1_line.split("1. Is the top prediction appropriate?")[1].strip()
            if q2_line.startswith("2."):
                feedback_data['Question 2'] = q2_line.split("2. Are the alternative predictions appropriate?")[1].strip()
            if q3_line.startswith("3."):
                feedback_data['Question 3'] = q3_line.split("3. In relation to how clear the drawing is, is the prediction too confident?")[1].strip()

            data['Feedback'] = feedback_data
            break  # Exit after parsing feedback

    return data

def load_image(image_path):
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        return None

def main():
    st.title("Experiment Results Report")

    exp_results_dir = "../../exp_results"

    # Collect all subjects
    subjects = [d for d in os.listdir(exp_results_dir) if d.startswith('Subject_')]
    subjects.sort()  # Sort for consistent ordering

    data_list = []

    for subject in subjects:
        subject_path = os.path.join(exp_results_dir, subject)
        subject_num = subject.split('_')[1]

        digits = [d for d in os.listdir(subject_path) if d.startswith('digit_')]
        digits.sort(key=lambda x: int(x.split('_')[1]))  # Sort digits numerically

        for digit_dir in digits:
            digit = digit_dir.split('_')[1]
            digit_path = os.path.join(subject_path, digit_dir)

            draws = [d for d in os.listdir(digit_path) if d.startswith('draw_')]
            draws.sort(key=lambda x: int(x.split('_')[1]))  # Sort draws numerically

            for draw_dir in draws:
                draw_number = draw_dir.split('_')[1]
                draw_path = os.path.join(digit_path, draw_dir)

                # Load images
                original_drawing_path = os.path.join(draw_path, "original_drawing.png")
                processed_drawing_path = os.path.join(draw_path, "processed_drawing.png")

                original_drawing = load_image(original_drawing_path)
                processed_drawing = load_image(processed_drawing_path)

                # Load prediction data
                prediction_file_path = os.path.join(draw_path, "prediction.txt")
                if os.path.exists(prediction_file_path):
                    prediction_data = parse_prediction_file(prediction_file_path)
                else:
                    prediction_data = {}

                # Load probability plot
                probability_plot = None
                for file in os.listdir(draw_path):
                    if file.startswith('probabilities_plot_') and file.endswith('.png'):
                        plot_path = os.path.join(draw_path, file)
                        probability_plot = load_image(plot_path)
                        break  # Assuming only one plot per draw

                # Collect data
                data_entry = {
                    'Subject': subject_num,
                    'Digit': int(digit),
                    'Draw Number': int(draw_number),
                    'Original Drawing': original_drawing,
                    'Processed Drawing': processed_drawing,
                    'Probability Plot': probability_plot,
                    'Intended Digit': prediction_data.get('Intended Digit', None),
                    'Model Used': prediction_data.get('Model Used', None),
                    'Predicted Digit': prediction_data.get('Predicted Digit', None),
                    'Confidence': prediction_data.get('Confidence', None),
                    'Feedback': prediction_data.get('Feedback', {}),
                }

                data_list.append(data_entry)

    # Convert data_list to DataFrame for easier manipulation
    df = pd.DataFrame(data_list)

    # Identify entries with missing or invalid 'Feedback' data
    missing_feedback_entries = df[df['Feedback'].apply(lambda x: not isinstance(x, dict) or not x)]
    if not missing_feedback_entries.empty:
        st.write("### Entries with Missing or Invalid Feedback Data:")
        for idx, row in missing_feedback_entries.iterrows():
            st.write(f"Subject {row['Subject']}, Digit {row['Digit']}, Draw {row['Draw Number']}")
        st.write("Please check these entries for missing feedback.")

    # Ensure 'Feedback' column is always a dictionary
    df['Feedback'] = df['Feedback'].apply(lambda x: x if isinstance(x, dict) else {})

    # Define the desired order for feedback answers
    feedback_order = ['Strongly agree', 'Agree', 'Neutral', 'Disagree', 'Strongly disagree']

    # Sidebar filters
    st.sidebar.title("Filters")
    # Subjects filter
    subjects = sorted(df['Subject'].unique())
    selected_subjects = st.sidebar.multiselect("Select Subjects", subjects, default=subjects)
    # Digits filter
    digits = sorted(df['Digit'].unique())
    selected_digits = st.sidebar.multiselect("Select Digits", digits, default=digits)
    # Models filter
    models = df['Model Used'].dropna().unique()
    selected_models = st.sidebar.multiselect("Select Models", models, default=models)
    # Feedback filter
    st.sidebar.subheader("Filter by Feedback Answers")
    feedback_questions = ['Question 1', 'Question 2', 'Question 3']
    feedback_options = {}
    for q in feedback_questions:
        all_answers = df['Feedback'].apply(lambda x: x.get(q, 'N/A')).unique()
        all_answers = [ans for ans in feedback_order if ans in all_answers]
        selected_answers = st.sidebar.multiselect(f"{q} Options", options=all_answers, default=all_answers, key=q)
        feedback_options[q] = selected_answers

    # Display options
    st.sidebar.title("Display Options")
    show_original_drawing = st.sidebar.checkbox("Show Original Drawing", value=True)
    show_processed_drawing = st.sidebar.checkbox("Show Processed Image", value=True)
    show_probability_plot = st.sidebar.checkbox("Show Probability Plot", value=True)
    show_prediction_data = st.sidebar.checkbox("Show Prediction Data", value=True)
    show_feedback = st.sidebar.checkbox("Show Feedback", value=True)

    # Summary statistics options
    st.sidebar.title("Summary Statistics Options")
    show_avg_conf_per_model = st.sidebar.checkbox("Average Confidence per Model", value=True)
    show_conf_by_feedback = st.sidebar.checkbox("Confidence by Feedback Answer", value=False)
    show_confidence_distribution = st.sidebar.checkbox("Confidence Distribution", value=False)
    show_accuracy_by_feedback = st.sidebar.checkbox("Prediction Accuracy by Feedback Answer", value=False)
    show_correlation = st.sidebar.checkbox("Correlation Analysis", value=False)

    # Filter the DataFrame
    df_filtered = df[
        (df['Subject'].isin(selected_subjects)) &
        (df['Digit'].isin(selected_digits)) &
        (df['Model Used'].isin(selected_models))
    ]

    # Apply feedback filters
    for q in feedback_questions:
        df_filtered = df_filtered[df_filtered['Feedback'].apply(
            lambda x: x.get(q, 'N/A') in feedback_options[q] if isinstance(x, dict) else False)]

    if df_filtered.empty:
        st.write("No data available for the selected filters.")
        return

    # Display data
    st.header("Experiment Results")

    for idx, row in df_filtered.iterrows():
        st.subheader(f"Subject {row['Subject']} - Digit {row['Digit']} - Draw {row['Draw Number']}")

        cols = st.columns(3)
        if show_original_drawing and row['Original Drawing'] is not None:
            with cols[0]:
                st.image(row['Original Drawing'], caption="Original Drawing", width=150)
        if show_processed_drawing and row['Processed Drawing'] is not None:
            with cols[1]:
                st.image(row['Processed Drawing'], caption="Processed Image", width=150)
        if show_probability_plot and row['Probability Plot'] is not None:
            with cols[2]:
                st.image(row['Probability Plot'], caption="Probability Plot", use_container_width=True)

        if show_prediction_data:
            st.write(f"**Intended Digit:** {row.get('Intended Digit', 'N/A')}")
            st.write(f"**Model Used:** {row.get('Model Used', 'N/A')}")
            st.write(f"**Predicted Digit:** {row.get('Predicted Digit', 'N/A')}")
            confidence = row.get('Confidence', None)
            if confidence is not None:
                st.write(f"**Confidence:** {confidence:.2f}%")
            else:
                st.write("**Confidence:** N/A")

        if show_feedback:
            feedback = row.get('Feedback', {})
            if feedback:
                st.write("**Feedback Answers:**")
                st.write(f"1. Is the top prediction appropriate? **{feedback.get('Question 1', 'N/A')}**")
                st.write(f"2. Are the alternative predictions appropriate? **{feedback.get('Question 2', 'N/A')}**")
                st.write(f"3. Is the prediction too confident? **{feedback.get('Question 3', 'N/A')}**")
            else:
                st.write("**No feedback provided.**")

        st.markdown("---")

    # Summary Statistics
    st.header("Summary Statistics")

    # Add Correct Prediction column
    df_filtered['Correct Prediction'] = df_filtered.apply(
        lambda row: int(row.get('Predicted Digit', -1) == row.get('Intended Digit', None)), axis=1
    )

    # Prepare data for statistics
    stats_df = df_filtered.copy()
    feedback_expanded = stats_df['Feedback'].apply(pd.Series)
    stats_df = pd.concat([stats_df, feedback_expanded], axis=1)

    # Ensure feedback columns are categorical with specified order
    for q in feedback_questions:
        stats_df[q] = pd.Categorical(stats_df[q], categories=feedback_order, ordered=True)

    if not stats_df.empty:
        # Average confidence per model
        if show_avg_conf_per_model:
            st.subheader("Average Confidence per Model")
            avg_confidence = stats_df.groupby('Model Used')['Confidence'].mean().reset_index()
            st.table(avg_confidence)

            # Bar plot
            fig, ax = plt.subplots()
            sns.barplot(data=avg_confidence, x='Model Used', y='Confidence', ax=ax)
            ax.set_title('Average Confidence per Model')
            ax.set_ylabel('Average Confidence (%)')
            st.pyplot(fig)

        # Confidence by Feedback Answer
        if show_conf_by_feedback:
            st.subheader("Average Confidence per Model by Feedback Answer")
            for q in feedback_questions:
                st.write(f"### {q}")
                conf_by_feedback = stats_df.groupby(['Model Used', q])['Confidence'].mean().reset_index()

                # Ensure the 'q' column is categorical with the specified order
                conf_by_feedback[q] = pd.Categorical(conf_by_feedback[q], categories=feedback_order, ordered=True)

                # Sort the DataFrame
                conf_by_feedback = conf_by_feedback.sort_values(by=[q])

                st.table(conf_by_feedback)

                # Bar plot
                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(data=conf_by_feedback, x='Model Used', y='Confidence', hue=q, hue_order=feedback_order, ax=ax)
                ax.set_title(f'Average Confidence per Model by {q}')
                ax.set_ylabel('Average Confidence (%)')
                st.pyplot(fig)

        # Confidence Distribution
        if show_confidence_distribution:
            st.subheader("Confidence Distribution")
            fig, ax = plt.subplots(figsize=(10,6))
            sns.histplot(data=stats_df, x='Confidence', hue='Model Used', kde=True, ax=ax)
            ax.set_title('Confidence Distribution by Model')
            ax.set_xlabel('Confidence (%)')
            st.pyplot(fig)

        # Prediction Accuracy by Feedback Answer
        if show_accuracy_by_feedback:
            st.subheader("Prediction Accuracy by Feedback Answer")
            for q in feedback_questions:
                st.write(f"### {q}")
                accuracy_by_feedback = stats_df.groupby([q])['Correct Prediction'].mean().reset_index()
                accuracy_by_feedback['Accuracy (%)'] = accuracy_by_feedback['Correct Prediction'] * 100

                # Ensure the 'q' column is categorical with the specified order
                accuracy_by_feedback[q] = pd.Categorical(accuracy_by_feedback[q], categories=feedback_order, ordered=True)

                # Sort the DataFrame
                accuracy_by_feedback = accuracy_by_feedback.sort_values(by=[q])

                st.table(accuracy_by_feedback[[q, 'Accuracy (%)']])

                # Bar plot
                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(data=accuracy_by_feedback, x=q, y='Accuracy (%)', order=feedback_order, ax=ax)
                ax.set_title(f'Prediction Accuracy by {q}')
                ax.set_ylabel('Accuracy (%)')
                st.pyplot(fig)

        # Correlation Analysis
        if show_correlation:
            st.subheader("Correlation Analysis")
            # Encode feedback answers numerically
            feedback_mapping = {
                'Strongly disagree': 1,
                'Disagree': 2,
                'Neutral': 3,
                'Agree': 4,
                'Strongly agree': 5
            }
            for q in feedback_questions:
                stats_df[q + ' (Numeric)'] = stats_df[q].map(feedback_mapping)

            correlation_cols = ['Confidence'] + [q + ' (Numeric)' for q in feedback_questions]
            corr_matrix = stats_df[correlation_cols].corr()
            st.write("Correlation Matrix:")
            st.table(corr_matrix)

            # Heatmap
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
    else:
        st.write("No data available for summary statistics.")

if __name__ == "__main__":
    main()