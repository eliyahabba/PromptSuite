import streamlit as st

from src.ui import (
    ask_user_for_info,
    upload_csv,
    annotate_prompt,
    assign_dimensions,
    add_dimensions,
    run_augmentations,
    show_variants
)
from src.ui.utils.debug_helpers import (
    initialize_debug_mode,
    load_demo_data_for_step
)
from src.ui.utils.progress_indicator import show_progress_indicator


# from src.decompose_tasks import instruction_breakdown


def main():
    # Set up page configuration
    st.set_page_config(layout="wide", page_title="Multi-Prompt Evaluation Tool")
    # Retrieve the query parameters from the URL using st.query_params (new API)
    params = st.query_params  # This returns a dict, with each value as a list of strings.

    # Extract the 'step' parameter, defaulting to "1" if not present.
    # For example, if the URL is .../?step=3, then params == {"step": ["3"]}
    start_step = int(params.get("step", ["1"])[0])

    # Extract the 'debug' parameter, defaulting to "false" if not present.
    debug_mode = params.get("debug", ["False"]) == "true"

    st.write(f"Step: {start_step}, Debug mode: {debug_mode}")
    # Parse query parameters - updated to use non-experimental API
    # Initialize session state
    initialize_session_state(start_step, debug_mode)

    # Initialize debug mode UI if needed
    if st.session_state.debug_mode:
        initialize_debug_mode()

    # Total number of pages in the application
    total_pages = 7

    # Display the progress indicator at the top of every page
    current_page = st.session_state.page
    show_progress_indicator(current_page, total_pages)

    # Render the appropriate page based on the current state
    render_current_page(current_page)


def initialize_session_state(start_step=1, debug_mode=False):
    """Initialize the session state for navigation"""
    if 'page' not in st.session_state:
        st.session_state.page = start_step

    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = debug_mode

    # If in debug mode and starting from a specific step, load demo data
    if debug_mode and start_step > 1 and 'loaded_demo_data' not in st.session_state:
        st.session_state.loaded_demo_data = True
        load_demo_data_for_step(start_step)


def render_current_page(current_page):
    """Render the appropriate page based on the current state"""
    pages = {
        1: upload_csv.render,
        2: annotate_prompt.render,
        3: add_dimensions.render,
        4: assign_dimensions.render,
        5: ask_user_for_info.render,
        6: run_augmentations.render,
        7: show_variants.render
    }

    # Call the render function for the current page
    pages[current_page]()


if __name__ == '__main__':
    main()
