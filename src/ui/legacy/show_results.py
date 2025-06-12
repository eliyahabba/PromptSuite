"""
Step 4: Show Results for MultiPromptify 2.0
"""

import streamlit as st
from src.ui.components.results_display import display_full_results


def render():
    """Render the enhanced results display interface"""
    if not st.session_state.get('variations_generated', False):
        st.error("⚠️ Please generate variations first (Step 3)")
        if st.button("← Go to Step 3"):
            st.session_state.page = 3
            st.rerun()
        return

    # Enhanced header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; text-align: center;">🎉 Step 4: View Results</h1>
        <p style="color: rgba(255,255,255,0.9); text-align: center; margin: 0.5rem 0 0 0;">
            Explore your generated prompt variations with enhanced visualization
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get results data
    variations = st.session_state.generated_variations
    stats = st.session_state.generation_stats
    generation_time = st.session_state.generation_time
    original_data = st.session_state.uploaded_data

    # Use the shared display function
    display_full_results(
        variations=variations,
        original_data=original_data,
        stats=stats,
        generation_time=generation_time,
        show_export=True,
        show_header=False  # Don't show header since we have our own above
    ) 