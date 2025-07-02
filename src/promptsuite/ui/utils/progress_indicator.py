import streamlit as st

# Page descriptions for the progress indicator
PAGE_DESCRIPTIONS = {
    1: "Upload Data",
    2: "Build Template", 
    3: "Generate Variations"
}

def show_progress_indicator(current_page, total_pages=3):
    """
    Display a progress indicator showing which page the user is on and how many remain.
    
    Args:
        current_page: The current page number (1-based)
        total_pages: The total number of pages in the process
    """
    # Calculate progress percentage
    progress_value = current_page / total_pages
    
    # Create a container with custom styling
    with st.container():
        # Add a separator line
        st.markdown('<hr style="margin-top: 0; margin-bottom: 10px;">', unsafe_allow_html=True)
        
        # Display progress text
        st.markdown(f'<p style="text-align: center; margin-bottom: 5px;">Step {current_page} of {total_pages}</p>', 
                  unsafe_allow_html=True)
        
        # Show progress bar
        st.progress(progress_value)
        
        # Show current page description
        if current_page in PAGE_DESCRIPTIONS:
            st.markdown(
                f'<p style="text-align: center; font-weight: bold;">{PAGE_DESCRIPTIONS[current_page]}</p>', 
                unsafe_allow_html=True
            )
        
        # Add a separator line
        st.markdown('<hr style="margin-top: 10px; margin-bottom: 20px;">', unsafe_allow_html=True)
