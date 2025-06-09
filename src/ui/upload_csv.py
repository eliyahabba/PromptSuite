# File: pages/upload_csv.py
import streamlit as st
import pandas as pd


def render():
    st.title("Step 1: Upload Your Prompt Dataset")
    st.write("Please upload a CSV file with a `prompt` column.")

    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ["prompt"]
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain these columns: {', '.join(required_columns)}")
                return

            if "output" in df.columns:
                st.session_state.output_data = df[["prompt", "output"]].set_index("prompt").to_dict()["output"]
                st.info("✅ Output column detected and saved")
            else:
                st.warning("No 'output' column found. Some features may not work properly.")
                st.session_state.output_data = {}

            st.session_state.csv_data = df
            st.success(f"Uploaded {len(df)} prompts.")
            st.write(df.head())

            if st.button("Continue to Annotation"):
                st.session_state.current_example_index = 0
                st.session_state.annotated_examples = []
                st.session_state.annotation_complete = False
                st.session_state.prompt = df['prompt'].iloc[0]
                st.session_state.page = 2
                js = '''
                <script>
                    var body = window.parent.document.querySelector(".main");
                    console.log(body);
                    body.scrollTop = 0;
                </script>
                '''

                st.components.v1.html(js)
                st.rerun()

        except Exception as e:
            st.error(f"Error reading file: {e}")
