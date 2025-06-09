import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.decompose_tasks import instruction_breakdown
from src.utils.constants import DEFAULT_MODEL

load_dotenv()

# Get API key from environment
API_KEY = os.getenv("TOGETHER_API_KEY")


def render():
    st.title("Step 5: Predict Prompt Parts")
    st.markdown("Use a few-shot LLM to automatically extract parts of your prompts.")

    # Initialize session state variables if they don't exist
    initialize_session_state()

    # Display the form with pre-filled values from session state
    display_model_form()

    # Process form submission if needed
    if st.session_state['submitted']:
        process_form_submission()


def initialize_session_state():
    """Initialize all required session state variables with default values"""
    defaults = {
        'output_dir': "tmp/",
        'platform': "TogetherAI",
        'api_key': API_KEY,
        'model_name': DEFAULT_MODEL,
        'submitted': False,
        'save_files': True  # Default to saving files
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def handle_form_submit():
    """Callback function for form submission"""
    st.session_state['submitted'] = True
    st.session_state['save_files'] = st.session_state.save_files_checkbox

    # Only update output_dir if save_files is enabled
    if st.session_state.save_files_checkbox and 'output_dir_input' in st.session_state:
        st.session_state['output_dir'] = st.session_state.output_dir_input


def display_model_form():
    """Display the form for model settings"""
    with st.form(key="prediction_form"):
        st.subheader("🧠 Choose Your Model Platform")
        st.selectbox("Platform", ["TogetherAI", "OpenAI"], key="platform")

        st.subheader(f"🔑 Enter your API Key")
        st.text_input("API Key", type="password", key="api_key", value=API_KEY)

        st.subheader("📦 Model Name")
        st.text_input("Model Name", key="model_name", value=DEFAULT_MODEL)

        st.subheader("📁 Data Storage Options")
        st.checkbox("Save files to disk (recommended for large datasets)",
                    key="save_files_checkbox",
                    value=st.session_state.save_files,
                    help="When enabled, intermediate files will be saved to disk. Otherwise, data is only kept in memory.")

        # Only show output directory if save_files is checked
        if st.session_state.get('save_files', True):
            st.text_input("Output Directory", key="output_dir_input", value=st.session_state['output_dir'])

        # Submit button with callback
        st.form_submit_button("🚀 Start Prediction", on_click=handle_form_submit)


def process_form_submission():
    """Process the form submission and run the prediction"""
    # Reset the submitted flag for next time
    st.session_state['submitted'] = False

    # Validate required fields
    missing_fields = validate_required_fields()

    if missing_fields:
        st.warning(f"Please fill in the following fields: {', '.join(missing_fields)}.")
    else:
        with st.spinner("Running prediction..."):
            try:
                run_prediction()
                st.success("✅ Prediction successful!")
                st.session_state.page = 6
                st.rerun()
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")


def validate_required_fields():
    """Validate that all required fields are filled"""
    missing_fields = []
    required_fields = ["platform", "api_key", "model_name"]

    # Only require output_dir if save_files is True
    if st.session_state.get('save_files', True):
        required_fields.append("output_dir_input")

    for field in required_fields:
        if not st.session_state.get(field):
            missing_fields.append(field.replace('_', ' ').title())
    return missing_fields


def run_prediction():
    """Run the prediction process using the selected model"""
    # Set API key if using TogetherAI
    if st.session_state.get("platform").lower() == "togetherai":
        os.environ["TOGETHER_API_KEY"] = st.session_state.api_key

    # Determine if we should save files to disk
    save_files = st.session_state.get('save_files', True)

    if save_files:
        # Create output directory and save files
        output_dir = st.session_state['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Save annotation files
        annotations_path, all_data_path = save_annotation_files(output_dir)
        out_path = os.path.join(output_dir, "predictions.csv")
    else:
        # Use memory-based approach with temporary paths
        annotations_path = "memory://annotations.json"
        all_data_path = "memory://all_data.csv"
        out_path = "memory://predictions.csv"

        # Keep data in memory
        st.session_state['annotations_data'] = st.session_state.final_annotations_output

    # Display information about the number of examples being used
    st.info(f"Using {len(st.session_state.final_annotations_output)} annotated examples for prediction")

    # Run instruction breakdown
    results_df = instruction_breakdown.main(
        annotation_file=annotations_path,
        input_csv=all_data_path,
        output_csv=out_path,
        input_column="prompt",
        model_id=st.session_state.model_name,
        delay=0.5,
        provider="together",
        memory_mode=not save_files,
        annotations_data=st.session_state.get('annotations_data') if not save_files else None,
        csv_data=st.session_state.csv_data if not save_files else None
    )

    # Store results in session state
    if not save_files and results_df is not None:
        st.session_state['predictions_df'] = results_df
    else:
        st.session_state['predictions_df'] = pd.read_csv(out_path)


def save_annotation_files(output_dir):
    """Save annotation files to the output directory"""
    # Save annotations to file
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, "w") as annotations_file:
        json.dump(st.session_state.final_annotations_output, annotations_file)

    # Save all data to CSV
    all_data_path = os.path.join(output_dir, "all_data.csv")
    st.session_state.csv_data.to_csv(all_data_path, index=False)

    return annotations_path, all_data_path
