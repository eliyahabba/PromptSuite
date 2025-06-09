import json
import os

import pandas as pd
import streamlit as st

from src.integration.simple_augmenter import main as simple_augmenter_main
from src.ui.utils.map_csv_to_json import map_csv_to_json


def render():
    st.title("Step 6: Run Augmentations")

    # Check if we're using files or session state
    using_files = st.session_state.get('save_files', False)
    
    try:
        if using_files:
            # Get data from files
            output_dir = st.session_state.get("output_dir")
            st.markdown(f"Using files stored in: `{output_dir}`")
            
            df = pd.read_csv(os.path.join(output_dir, "predictions.csv"))
            annotations_path = os.path.join(output_dir, "annotations.json")
            with open(annotations_path, "r") as f:
                annotations_data = json.load(f)
        else:
            # Get data from session state
            st.markdown("Using in-memory data storage")
            
            if 'predictions_df' not in st.session_state:
                st.error("Required prediction data not found in session state. Please go back to step 5.")
                return
                
            if 'annotations_data' not in st.session_state:
                st.error("Required annotation data not found in session state. Please go back to step 5.")
                return
                
            df = st.session_state.predictions_df
            annotations_data = st.session_state.annotations_data
        
        # Convert to the format needed for augmentation

        # Create a separate button for navigation to avoid conflicts
        col1, col2 = st.columns(2)

        # Only show the Run Augmentations button if no augmented data exists
        if "augmented_data" not in st.session_state:
            with col1:
                st.write(
                    "Click the button below to run the augmentation process "
                    "using the data from step 5."
                )
                if st.button("Run Augmentations"):
                    with st.spinner("Running augmentations..."):
                        final_json = map_csv_to_json(df, annotations_data)
                        data = simple_augmenter_main(final_json)
                        # Store in session state
                        st.session_state["augmented_data"] = data
                        st.success("Augmentations completed successfully!")
                        # Force a rerun to update the UI state
                        # st.rerun()

        # Display results if we have augmented data
        if "augmented_data" in st.session_state:
            st.subheader("Augmentation Results")
            st.json(st.session_state["augmented_data"])

            # Provide a download button
            st.download_button(
                label="Download Augmented Variations as JSON",
                data=json.dumps(st.session_state["augmented_data"], indent=2),
                file_name="augmented_variations.json",
                mime="application/json",
                key="download_augmented_file"
            )

            # Allow saving to disk even if using memory mode
            if not using_files and st.button("Save Results to Disk"):
                save_directory = st.text_input("Output Directory", value="results")
                if st.button("Confirm Save"):
                    os.makedirs(save_directory, exist_ok=True)
                    with open(os.path.join(save_directory, "augmented_variations.json"), "w") as f:
                        json.dump(st.session_state["augmented_data"], f, indent=2)
                    st.success(f"Results saved to {save_directory}")

            # Clear separation between download and navigation
            st.markdown("---")
            st.subheader("Continue to Next Step")

            # Navigation button in its own container to avoid conflicts
            if st.button("Continue to Step 7 (Show Variants)", key="continue_to_step7"):
                st.session_state.page = 7
                st.rerun()

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error("Please go back to step 5 and try again.")
        return