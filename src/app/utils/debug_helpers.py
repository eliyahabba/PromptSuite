import pandas as pd
import json
import streamlit as st
import os
from pathlib import Path

# Base directory for demo data - use absolute path
DEMO_DATA_DIR = Path(__file__).resolve().parents[3] / "demo_data"
print(f"Demo data directory: {DEMO_DATA_DIR}")

def initialize_debug_mode():
    """Initialize session state variables for debug mode"""
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = True
    
    # Test directory creation
    try:
        os.makedirs(DEMO_DATA_DIR, exist_ok=True)
        st.sidebar.success(f"Debug directory created at {DEMO_DATA_DIR}")
    except Exception as e:
        print(f"Error creating directory or test file: {str(e)}")
        st.sidebar.error(f"Error creating debug directory: {str(e)}")
    
    # Add a sidebar indicator for debug mode
    st.sidebar.markdown("### Debug Mode")
    st.sidebar.checkbox("Active", value=st.session_state.debug_mode, key="debug_mode")
    
    if st.session_state.debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Load Demo Data")
        
        # Add buttons for loading demo data for each step
        for step in range(1, 8):
            if st.sidebar.button(f"Load demo data for step {step}"):
                load_demo_data_for_step(step)
        
        # Add a more visible button to save current state
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Save Current State")
        if st.sidebar.button("💾 SAVE CURRENT STATE AS DEMO DATA", use_container_width=True):
            save_current_state()

def save_current_state():
    """Save the current application state to demo data files"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(DEMO_DATA_DIR, exist_ok=True)
        
        # Print debug information
        print(f"Saving to directory: {DEMO_DATA_DIR}")
        print(f"Directory exists: {os.path.exists(DEMO_DATA_DIR)}")
        print(f"Current step: {st.session_state.get('page', 1)}")
        print(f"Session state keys: {list(st.session_state.keys())}")
        
        # Determine current step
        current_step = st.session_state.get('page', 1)
        
        # Save data based on current step
        if current_step >= 1 and 'csv_data' in st.session_state:
            csv_path = DEMO_DATA_DIR / "sample_prompts.csv"
            st.session_state.csv_data.to_csv(csv_path, index=False)
            print(f"Saved sample prompts")
            st.sidebar.success(f"Saved CSV data to {csv_path}")
            
        if current_step >= 2 and 'annotated_parts' in st.session_state:
            with open(DEMO_DATA_DIR / "annotated_parts.json", 'w') as f:
                json.dump(st.session_state.annotated_parts, f, indent=2)
            st.sidebar.success("Saved annotations")

        if current_step >= 3 and 'base_dimensions' in st.session_state:
            # Save dimensions
            dimensions_data = {
                "base_dimensions": st.session_state.base_dimensions,
                "custom_dimensions": st.session_state.get('custom_dimensions', [])
            }
            with open(DEMO_DATA_DIR / "dimensions.json", 'w') as f:
                json.dump(dimensions_data, f, indent=2)
            st.sidebar.success("Saved dimensions")
        if current_step >= 4 and 'sample_annotations' in st.session_state:
            with open(DEMO_DATA_DIR / "sample_annotations.json", 'w') as f:
                json.dump(st.session_state.sample_annotations, f, indent=2)
            st.sidebar.success("Saved annotations")

        if current_step >= 4 and 'dimension_assignments' in st.session_state:
            # Save dimension assignments
            assignments_data = {
                "dimension_assignments": st.session_state.dimension_assignments,
                "dimension_variant_counts": st.session_state.dimension_variant_counts
            }
            with open(DEMO_DATA_DIR / "sample_dimension_assignments.json", 'w') as f:
                json.dump(assignments_data, f, indent=2)
            st.sidebar.success("Saved dimension assignments")
            
        if current_step >= 5 and 'final_annotations_output' in st.session_state:
            with open(DEMO_DATA_DIR / "final_annotations.json", 'w') as f:
                json.dump(st.session_state.final_annotations_output, f, indent=2)
            st.sidebar.success("Saved final annotations")
            st.session_state.predictions_df.to_csv(DEMO_DATA_DIR / "predictions.csv", index=False)
            st.sidebar.success("Saved predictions data")
            annotations_path = os.path.join(DEMO_DATA_DIR, "annotations.json")
            with open(annotations_path, "w") as annotations_file:
                json.dump(st.session_state.final_annotations_output, annotations_file)

        if current_step >= 7 and 'augmented_data' in st.session_state:
            with open(DEMO_DATA_DIR / "augmented_data.json", 'w') as f:
                json.dump(st.session_state.augmented_data, f, indent=2)
            st.sidebar.success("Saved augmented data")
            
        # Save output data if it exists
        if 'output_data' in st.session_state:
            output_path = DEMO_DATA_DIR / "output_data.json"
            with open(output_path, 'w') as f:
                json.dump(st.session_state.output_data, f, indent=2)
            st.sidebar.success("Saved output data")
            
        st.sidebar.success("Successfully saved current state as demo data!")
        
    except Exception as e:
        st.sidebar.error(f"Error saving state: {str(e)}")

def load_demo_data_for_step(step_number):
    """Load demo data for a specific step"""
    try:
        if step_number == 1:
            # Demo data for CSV upload
            demo_csv_path = DEMO_DATA_DIR / "sample_prompts.csv"
            if demo_csv_path.exists():
                df = pd.read_csv(demo_csv_path)
                st.session_state.csv_data = df
                st.success(f"Loaded {len(df)} rows from demo data")
            else:
                st.error(f"Demo file not found: {demo_csv_path}")
                
        if step_number >= 2:
            # Demo data for prompt annotation
            if 'csv_data' not in st.session_state:
                load_demo_data_for_step(1)
                
            if 'csv_data_sampled' not in st.session_state:
                df = st.session_state.csv_data.sample(1, random_state=1)
                st.session_state.csv_data_sampled = df
                
            st.session_state.current_example_index = 0
            st.session_state.prompt = st.session_state.csv_data_sampled['prompt'].iloc[0]
            demo_json_path = DEMO_DATA_DIR / "annotated_parts.json"
            with open(demo_json_path, 'r') as f:
                st.session_state.annotated_parts = json.load(f)

        if step_number >= 3:
            # Demo data for dimensions
            dimensions_path = DEMO_DATA_DIR / "dimensions.json"
            if dimensions_path.exists():
                with open(dimensions_path, 'r') as f:
                    dimensions_data = json.load(f)
                    st.session_state.base_dimensions = dimensions_data.get("base_dimensions", [])
                    st.session_state.custom_dimensions = dimensions_data.get("custom_dimensions", [])
            else:
                # Load default dimensions if no saved dimensions
                from src.ui.add_dimensions import DEFAULT_DIMENSIONS
                st.session_state.base_dimensions = DEFAULT_DIMENSIONS
                st.session_state.custom_dimensions = []
                

        if step_number == 4:
            # First ensure we have dimensions
            if 'base_dimensions' not in st.session_state:
                load_demo_data_for_step(3)
                
            # Then load sample dimension assignments
            demo_json_path = DEMO_DATA_DIR / "sample_dimension_assignments.json"
            if demo_json_path.exists():
                with open(demo_json_path, 'r') as f:
                    data = json.load(f)
                    st.session_state.dimension_assignments = data["dimension_assignments"]
                    st.session_state.dimension_variant_counts = data["dimension_variant_counts"]
                st.success("Loaded dimension assignments")
            else:
                st.error(f"Demo file not found: {demo_json_path}")
                
        if step_number == 5:
            # First ensure we have prior state
            if 'annotated_parts' not in st.session_state:
                load_demo_data_for_step(4)
                
            # Load final annotations
            demo_json_path = DEMO_DATA_DIR / "final_annotations.json"
            if demo_json_path.exists():
                with open(demo_json_path, 'r') as f:
                    st.session_state.final_annotations_output = json.load(f)
                st.success("Loaded final annotations")
            else:
                st.error(f"Demo file not found: {demo_json_path}")
                
        if step_number == 6:
            # First ensure we have prior state
            if 'final_annotations_output' not in st.session_state:
                load_demo_data_for_step(5)
                
            # Load predictions data
            demo_csv_path = DEMO_DATA_DIR / "predictions.csv"
            if demo_csv_path.exists():
                df = pd.read_csv(demo_csv_path)
                st.session_state.predictions_df = df
                st.success("Loaded prediction data")
            else:
                st.error(f"Demo file not found: {demo_csv_path}")
            annotations_path = os.path.join(DEMO_DATA_DIR, "annotations.json")
            with open(annotations_path, "w") as annotations_file:
                json.dump(st.session_state.final_annotations_output, annotations_file)
            st.session_state.annotations_data = st.session_state.final_annotations_output
                
        if step_number == 7:
            # First ensure we have prior state
            if 'predictions_df' not in st.session_state:
                load_demo_data_for_step(6)
                
            # Load augmented data
            demo_json_path = DEMO_DATA_DIR / "augmented_data.json"
            if demo_json_path.exists():
                with open(demo_json_path, 'r') as f:
                    st.session_state.augmented_data = json.load(f)
                st.success("Loaded augmented data")
            else:
                st.error(f"Demo file not found: {demo_json_path}")
        
    except Exception as e:
        st.error(f"Error loading demo data: {str(e)}")

def ensure_required_state_for_step(step_number):
    """
    Checks if all required state variables for a step exist.
    If not, offers to load demo data.
    
    Returns True if the state is valid, False otherwise.
    """
    if step_number == 1:
        # Step 1 doesn't require previous state
        return True
        
    elif step_number == 2:
        if 'csv_data' not in st.session_state:
            st.warning("Missing data from previous step. Please load demo data.")
            if st.button("Load demo data for this step"):
                load_demo_data_for_step(step_number)
            return False
        return True
        
    elif step_number == 3:
        if 'annotated_parts' not in st.session_state:
            st.warning("Missing data from previous step. Please load demo data.")
            if st.button("Load demo data for this step"):
                load_demo_data_for_step(step_number)
            return False
        return True
    
    elif step_number == 4:
        missing_data = []
        if 'base_dimensions' not in st.session_state:
            missing_data.append("dimensions")
        if 'annotated_parts' not in st.session_state:
            missing_data.append("annotated parts")
            
        if missing_data:
            st.warning(f"Missing data from previous step: {', '.join(missing_data)}. Please load demo data.")
            if st.button("Load demo data for this step"):
                load_demo_data_for_step(step_number)
            return False
        return True
    
    elif step_number == 5:
        if 'final_annotations_output' not in st.session_state:
            if 'annotated_parts' in st.session_state:
                st.warning("Please save the annotations before proceeding.")
                return False
            else:
                st.warning("Missing data from previous step. Please load demo data.")
                if st.button("Load demo data for this step"):
                    load_demo_data_for_step(step_number)
                return False
        return True
    
    elif step_number == 6:
        if 'predictions_df' not in st.session_state and 'final_annotations_output' not in st.session_state:
            st.warning("Missing data from previous step. Please load demo data.")
            if st.button("Load demo data for this step"):
                load_demo_data_for_step(step_number)
            return False
        return True
    
    elif step_number == 7:
        if 'augmented_data' not in st.session_state:
            st.warning("Missing augmented data. Please run the augmentation step first.")
            if st.button("Load demo data for this step"):
                load_demo_data_for_step(step_number)
            return False
        return True
    
    return True  # Default case