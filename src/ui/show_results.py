"""
Step 4: Show Results for MultiPromptify 2.0
"""

import streamlit as st
import pandas as pd
import json
import io

# Define colors for highlighting different parts
HIGHLIGHT_COLORS = {
    "original": "#E8F5E8",      # Light green for original values
    "variation": "#FFF2CC",     # Light yellow for variations
    "field": "#E3F2FD",         # Light blue for field names
    "template": "#F3E5F5"       # Light purple for template parts
}


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

    # Display summary metrics
    display_enhanced_summary_metrics(variations, stats, generation_time)

    # Simplified tabbed interface - only keep All Variations and Export
    tab1, tab2 = st.tabs(["📋 All Variations", "💾 Export"])

    with tab1:
        display_enhanced_variations(variations, original_data)

    with tab2:
        export_interface(variations)


def display_enhanced_summary_metrics(variations, stats, generation_time):
    """Display enhanced summary metrics with cards"""
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #007bff; margin-bottom: 2rem;">
        <h3 style="color: #007bff; margin-top: 0;">📊 Generation Summary</h3>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced metrics with gradient cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{len(variations):,}</h2>
            <p style="margin: 0; opacity: 0.8;">Total Variations</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{stats.get('original_rows', 0)}</h2>
            <p style="margin: 0; opacity: 0.8;">Original Rows</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{stats.get('avg_variations_per_row', 0):.1f}</h2>
            <p style="margin: 0; opacity: 0.8;">Avg per Row</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: #333;">
            <h2 style="margin: 0; font-size: 2rem;">{generation_time:.1f}s</h2>
            <p style="margin: 0; opacity: 0.7;">Generation Time</p>
        </div>
        """, unsafe_allow_html=True)

    # Quick insights
    if variations:
        avg_length = sum(len(v['prompt']) for v in variations) / len(variations)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 20%); 
                    padding: 1rem; border-radius: 8px; text-align: center; color: white; margin-top: 1rem;">
            <p style="margin: 0; font-size: 1.1rem;">📏 Average prompt length: {avg_length:.0f} characters</p>
        </div>
        """, unsafe_allow_html=True)


def display_enhanced_variations(variations, original_data):
    """Display variations with enhanced visual presentation"""
    st.markdown("""
    <div style="background-color: #e8f5e8; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #28a745; margin-bottom: 2rem;">
        <h3 style="color: #28a745; margin-top: 0;">🎨 Enhanced Variations Display</h3>
        <p style="margin-bottom: 0; color: #155724;">Each variation shows the original data row and highlights the generated content</p>
    </div>
    """, unsafe_allow_html=True)

    if not variations:
        st.warning("No variations to display")
        return

    # Color legend
    display_color_legend()

    # Pagination controls with better styling
    total_variations = len(variations)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📄 Navigation")
        page_options = [5, 10, 20, 50]
        items_per_page = st.selectbox("Variations per page", page_options, index=1)

        total_pages = (total_variations - 1) // items_per_page + 1
        page = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1)
        if page is None:
            page = 1

    # Calculate range
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_variations)

    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
        <strong>Showing variations {start_idx + 1}-{end_idx} of {total_variations:,}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Display variations with enhanced formatting
    for i in range(start_idx, end_idx):
        variation = variations[i]
        display_single_variation(variation, i + 1, original_data)


def display_color_legend():
    """Display a color legend for the highlighting"""
    pass
    # st.markdown("### 🎨 Color Legend")
    # legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; padding: 1rem; background: #f8f9fa; border-radius: 8px;">'
    #
    # legend_items = {
    #     "Original Data": HIGHLIGHT_COLORS["original"],
    #     "Generated Variations": HIGHLIGHT_COLORS["variation"],
    #     "Field Names": HIGHLIGHT_COLORS["field"],
    #     "Template Structure": HIGHLIGHT_COLORS["template"]
    # }
    #
    # for label, color in legend_items.items():
    #     legend_html += f'''
    #     <div style="display: flex; align-items: center; gap: 8px;">
    #         <div style="width: 20px; height: 20px; background-color: {color}; border-radius: 4px; border: 1px solid #ccc;"></div>
    #         <span style="font-weight: 500;">{label}</span>
    #     </div>
    #     '''
    #
    # legend_html += '</div>'
    # st.markdown(legend_html, unsafe_allow_html=True)
    #

def display_single_variation(variation, variation_num, original_data):
    """Display a single variation with enhanced visualization"""
    original_row_index = variation.get('original_row_index', 0)

    # Create expandable card for each variation
    with st.expander(f"🔍 Variation {variation_num} (from row {original_row_index + 1})", expanded=True):

        # Two column layout
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            # Field values used in generation
            st.markdown("**🔧 Field Changes**")

            field_values = variation.get('field_values', {})
            original_values = variation.get('original_values', {})

            for field, value in field_values.items():
                original_val = original_values.get(field, value)

                # Check if value was modified
                is_modified = str(value) != str(original_val)

                if is_modified:
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #ff6b6b;">
                        <strong style="color: #1976d2;">{field}:</strong><br>
                        <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px; text-decoration: line-through; opacity: 0.7;">{original_val}</span><br>
                        <span style="background: {HIGHLIGHT_COLORS['variation']}; padding: 2px 6px; border-radius: 3px; font-weight: bold;">→ {value}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #28a745;">
                        <strong style="color: #1976d2;">{field}:</strong> <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            # Generated prompt display
            st.markdown("**✨ Generated Prompt**")

            # Highlight the prompt with field values
            highlighted_prompt = highlight_prompt_fields(variation['prompt'], field_values)

            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid #dee2e6; font-family: 'Courier New', monospace; line-height: 1.6; font-size: 14px;">
                {highlighted_prompt}
            </div>
            """, unsafe_allow_html=True)


def highlight_prompt_fields(prompt, field_values):
    """Highlight field values within the generated prompt"""
    highlighted = prompt

    # Sort field values by length (longest first) to avoid partial replacements
    sorted_fields = sorted(field_values.items(), key=lambda x: len(str(x[1])), reverse=True)

    for field, value in sorted_fields:
        if value and str(value).strip():
            # Escape HTML in the value
            escaped_value = str(value).replace('<', '&lt;').replace('>', '&gt;')

            # Create highlighted version
            highlighted_value = f'<span style="background: {HIGHLIGHT_COLORS["variation"]}; padding: 1px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #ffc107;">{escaped_value}</span>'

            # Replace in prompt (case-sensitive)
            highlighted = highlighted.replace(str(value), highlighted_value)

    # Convert newlines to HTML breaks and preserve spaces
    highlighted = highlighted.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')

    return highlighted


def export_interface(variations):
    """Interface for exporting results in various formats"""
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196f3; margin-bottom: 2rem;">
        <h3 style="color: #1976d2; margin-top: 0;">💾 Export Your Results</h3>
        <p style="margin-bottom: 0; color: #0d47a1;">Choose your preferred format to download the generated variations</p>
    </div>
    """, unsafe_allow_html=True)

    if not variations:
        st.warning("No variations to export")
        return

    # Enhanced export options with cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; margin-bottom: 1rem;">
            <h4 style="color: #ff9800; margin: 0;">📋 JSON Format</h4>
            <p style="margin: 0.5rem 0; color: #666; font-size: 0.9rem;">Complete data with metadata</p>
        </div>
        """, unsafe_allow_html=True)

        json_data = json.dumps(variations, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="prompt_variations.json",
            mime="application/json",
            use_container_width=True
        )

    with col2:
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; margin-bottom: 1rem;">
            <h4 style="color: #4caf50; margin: 0;">📊 CSV Format</h4>
            <p style="margin: 0.5rem 0; color: #666; font-size: 0.9rem;">Spreadsheet compatible</p>
        </div>
        """, unsafe_allow_html=True)

        # Flatten for CSV
        flattened = []
        for var in variations:
            flat_var = {
                'prompt': var['prompt'],
                'original_row_index': var.get('original_row_index', ''),
                'variation_count': var.get('variation_count', ''),
                'prompt_length': len(var['prompt'])
            }
            # Add field values
            for key, value in var.get('field_values', {}).items():
                flat_var[f'field_{key}'] = value
            flattened.append(flat_var)

        csv_df = pd.DataFrame(flattened)
        csv_data = csv_df.to_csv(index=False)

        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="prompt_variations.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col3:
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; margin-bottom: 1rem;">
            <h4 style="color: #9c27b0; margin: 0;">📝 Text Format</h4>
            <p style="margin: 0.5rem 0; color: #666; font-size: 0.9rem;">Plain text prompts only</p>
        </div>
        """, unsafe_allow_html=True)

        text_data = "\n\n--- VARIATION ---\n\n".join([var['prompt'] for var in variations])

        st.download_button(
            label="📥 Download TXT",
            data=text_data,
            file_name="prompt_variations.txt",
            mime="text/plain",
            use_container_width=True
        )