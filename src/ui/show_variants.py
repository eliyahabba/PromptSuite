import pandas as pd
import streamlit as st

# Define background colors for each part of the prompt
PART_COLORS = {
    "task_description": "#FFD580",  # light orange
    "context": "#BAE7FF",          # light blue
    "examples": "#D3F261",         # light green
    "choices": "#FF9CDD",          # light pink
}

def render():
    st.title("Step 7: Final Variations Display")
    st.markdown("This step shows all generated variations for each example, with each part highlighted in a different color.")
    
    # Get augmented data from session state
    data = st.session_state.get("augmented_data")
    
    if data is None:
        st.error("No augmented data found. Please run the augmentation step first.")
        return
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display a legend for the color coding
    display_color_legend()
    
    # Display each example and its variations
    display_all_examples(data)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.success("All variations are displayed above.") 

def display_color_legend():
    """Display a legend for the color coding of prompt parts"""
    st.markdown("### Color Legend")
    legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">'
    
    for part_name, color in PART_COLORS.items():
        legend_html += f'<div style="padding: 5px 10px; background-color: {color}; border-radius: 4px;">{part_name.replace("_", " ").title()}</div>'
    
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

def highlight_parts(final_prompt, parts):
    """
    Replace each occurrence of parts[part_name] in final_prompt with a colored span
    to highlight it according to the part_name, while preserving newlines.
    """
    highlighted = final_prompt
    for part_name, part_text in parts.items():
        if not part_text:
            continue
        color = PART_COLORS.get(part_name, "#FFFFB8")  # default light yellow
        if pd.isna(part_text):
            continue
        formatted_text = part_text.replace("\n", "<br>")
        highlighted = highlighted.replace(
            part_text,
            f'<span style="background-color: {color}; padding: 2px; margin: 1px; border-radius: 3px; white-space: pre-wrap;">{formatted_text}</span>'
        )
    return highlighted


def display_all_examples(data):
    """Display all examples and their variations"""
    for example_index, example_item in enumerate(data):
        original_prompt = example_item.get("original_prompt", "")
        variations = example_item.get("variations", [])

        st.markdown(f"### Example {example_index + 1}")
        st.markdown("**Original Prompt:**")
        st.code(original_prompt, language="")

        display_variations(variations)

def display_variations(variations):
    """Display all variations for a single example"""
    for var_index, var_item in enumerate(variations):
        final_prompt = var_item.get("final_prompt", "")
        parts = var_item.get("parts", {})

        highlighted_html = highlight_parts(final_prompt, parts)
        
        st.markdown(f"**Variation {var_index + 1}:**", unsafe_allow_html=True)
        st.markdown(
            f'<div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px;">'
            f'{highlighted_html}'
            f'</div>',
            unsafe_allow_html=True
        ) 