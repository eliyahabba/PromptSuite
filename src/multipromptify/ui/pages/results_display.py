"""
Shared Results Display Module for MultiPromptify 2.0
Contains all the display functions for generated variations
"""

import json

import pandas as pd
import streamlit as st

from multipromptify.core.engine import MultiPromptify
from multipromptify.core.template_keys import (
    PROMPT_FORMAT_VARIATIONS, FEW_SHOT_KEY, PROMPT_FORMAT
)


# Helper function to inject JavaScript for clipboard functionality
def _copy_to_clipboard_js():
    js_code = """
    function copyTextToClipboard(text) {
        var textArea = document.createElement("textarea");
        textArea.value = text;
        
        // Avoid scrolling to bottom
        textArea.style.top = "0";
        textArea.style.left = "0";
        textArea.style.position = "fixed";

        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        try {
            var successful = document.execCommand('copy');
            var msg = successful ? 'successful' : 'unsuccessful';
            console.log('Copying text command was ' + msg);
        } catch (err) {
            console.error('Oops, unable to copy', err);
        }

        document.body.removeChild(textArea);
    }
    """
    st.components.v1.html(f"<script>{js_code}</script>", height=0, width=0)


# Define colors for highlighting different parts
HIGHLIGHT_COLORS = {
    "original": "#E8F5E8",  # Light green for original values
    "variation": "#FFF2CC",  # Light yellow for variations
    "field": "#E3F2FD",  # Light blue for field names
    "template": "#F3E5F5"  # Light purple for template parts
}


def display_full_results(variations, original_data, stats, generation_time, show_export=True, show_header=True):
    """
    Main function to display the complete results with all features
    
    Args:
        variations: List of generated variations
        original_data: Original DataFrame
        stats: Generation statistics
        generation_time: Time taken for generation
        show_export: Whether to show export functionality
        show_header: Whether to show the success header
    """
    if not variations:
        st.warning("No variations to display")
        return

    if show_header:
        # Success header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                    padding: 2rem; border-radius: 10px; margin: 2rem 0;">
            <h2 style="color: white; margin: 0; text-align: center;">🎉 Generation Complete!</h2>
            <p style="color: rgba(255,255,255,0.9); text-align: center; margin: 0.5rem 0 0 0;">
                Your prompt variations have been successfully generated
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Display summary metrics
    display_enhanced_summary_metrics(variations, stats, generation_time)

    # Tabbed interface
    tab1, tab2, tab3 = st.tabs(["📋 All Variations", "💬 Conversation Format", "💾 Export"])

    with tab1:
        display_enhanced_variations(variations, original_data)

    with tab2:
        display_conversation_format(variations, original_data)

    with tab3:
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

        total_pages = (total_variations - 1) // items_per_page + 1 if total_variations > 0 else 1
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
    # Commenting out for now as mentioned in original code
    pass


def display_single_conversation(conversation, conversation_num, original_variation):
    """Display a single conversation in clean JSON format with correct answer when available"""

    # Handle both old and new conversation formats
    if isinstance(conversation, dict) and 'conversation' in conversation:
        # New enhanced format with metadata
        actual_conversation = conversation['conversation']
        metadata = conversation['metadata']
        original_row_index = metadata.get('original_row_index', 0)
    else:
        # Old format - conversation is just the messages
        actual_conversation = conversation
        metadata = {}
        original_row_index = original_variation.get('original_row_index', 0)

    # Create expandable card for each conversation
    with st.expander(f"💬 Conversation {conversation_num} (from row {original_row_index + 1})",
                     expanded=(conversation_num <= 3)):

        # Two column layout for conversation display
        col1, col2 = st.columns([2, 1])

        with col1:
            # JSON representation of just the conversation
            conversation_json = json.dumps(actual_conversation, indent=2, ensure_ascii=False)

            # הצגת כפתור העתקה מעוצב, textarea מוסתר, ותצוגה יפה עם st.code
            st.components.v1.html(f'''
                <div style="position: relative; margin-bottom: 0.5rem;">
                    <textarea id="conv_{conversation_num}" style="display:none;">{conversation_json}</textarea>
                    <button onclick="navigator.clipboard.writeText(document.getElementById('conv_{conversation_num}').value); var btn=this; btn.innerText='Copied!'; setTimeout(()=>btn.innerText='Copy Conversation',1200);" style="position:absolute;top:0;right:0;padding:6px 16px;background:linear-gradient(135deg,#2196f3 0%,#21cbf3 100%);color:white;border:none;border-radius:5px;cursor:pointer;font-weight:bold;box-shadow:0 2px 6px rgba(33,150,243,0.08);transition:background 0.2s;z-index:10;">Copy Conversation</button>
                </div>
            ''', height=38)
            st.code(conversation_json, language="json")

        with col2:
            # Get correct answer from gold_updates (always exists)
            correct_answer = None
            gold_updates = original_variation['gold_updates']
            if gold_updates:
                # Get the first (and usually only) gold field update
                correct_answer = next(iter(gold_updates.values()))

            # Display correct answer (should always be available now)
            if correct_answer is not None and str(correct_answer).strip():
                st.markdown("**🏆 Correct Answer:**")
                st.markdown(f"""
                <div style="background: #e8f5e8; padding: 0.75rem; border-radius: 6px; border-left: 4px solid #4caf50; margin: 0.5rem 0;">
                    <span style="color: #2e7d32; font-weight: bold; font-size: 1.1rem;">{correct_answer}</span>
                </div>
                """, unsafe_allow_html=True)
            elif correct_answer is not None and not str(correct_answer).strip():
                # Empty answer - this is valid for templates without gold field
                st.markdown("**🏆 Correct Answer:** No answer field configured in template")
            else:
                # This should never happen - if it does, it's a bug
                st.error("🐛 Bug: No correct answer found - please report this issue")


def display_single_variation(variation, variation_num, original_data):
    """Display a single variation with enhanced visualization"""
    original_row_index = variation.get('original_row_index', 0)

    # Get original row data for comparison
    original_row = original_data.iloc[original_row_index] if original_row_index < len(original_data) else None

    # Create expandable card for each variation
    with st.expander(f"🔍 Variation {variation_num} (from row {original_row_index + 1})", expanded=(variation_num <= 3)):

        # Two column layout
        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            # Field values used in generation with comparison
            st.markdown("**🔧 Field Changes**")

            field_values = variation.get('field_values', {})

            # Get original template for comparison
            original_template = st.session_state.get('selected_template', {})
            # Also check template_config from the variation itself
            template_config = variation.get('template_config', {})

            for field, value in field_values.items():
                if field == PROMPT_FORMAT_VARIATIONS:
                    # Get the original prompt_format template value
                    original_val = None
                    operation_type = None

                    # Try to get the original prompt_format_template from template_config first, then original_template
                    for template_source in [template_config, original_template]:
                        if isinstance(template_source, dict):
                            if PROMPT_FORMAT in template_source:
                                original_val = template_source[PROMPT_FORMAT]
                                break
                            elif 'template' in template_source:
                                # If it's a nested template structure
                                template_content = template_source['template']
                                if isinstance(template_content, dict) and PROMPT_FORMAT in template_content:
                                    original_val = template_content[PROMPT_FORMAT]
                                    break

                    # Check what operation was applied to prompt_format
                    for template_source in [template_config, original_template]:
                        if isinstance(template_source, dict) and PROMPT_FORMAT_VARIATIONS in template_source:
                            prompt_format_config = template_source[PROMPT_FORMAT_VARIATIONS]
                            if isinstance(prompt_format_config, list) and prompt_format_config:
                                operation_type = prompt_format_config[0]  # e.g., 'paraphrase'
                                break

                    # If we couldn't find the original, use the current value
                    if original_val is None:
                        original_val = value

                    # Check if prompt_format was modified
                    is_modified = str(value) != str(original_val) and original_val != value

                    if is_modified:
                        # Create operation description
                        operation_desc = f" ({operation_type})" if operation_type else ""

                        st.markdown(f"""
                        <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #667eea;">
                            <strong style="color: #1976d2;">{field}{operation_desc}:</strong><br>
                            <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px; text-decoration: line-through; opacity: 0.7;">{original_val}</span><br>
                            <span style="background: {HIGHLIGHT_COLORS['variation']}; padding: 2px 6px; border-radius: 3px; font-weight: bold;">→ {value}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        operation_desc = f" ({operation_type})" if operation_type else ""
                        st.markdown(f"""
                        <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #667eea;">
                            <strong style="color: #1976d2;">{field}{operation_desc}:</strong> <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px;">{value}</span>
                        </div>
                        """, unsafe_allow_html=True)

                elif field == FEW_SHOT_KEY:
                    # Show few-shot info
                    if value:
                        few_shot_count = value.count('\n\n') + 1 if value else 0
                        st.markdown(f"""
                        <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #667eea;">
                            <strong style="color: #1976d2;">{field}:</strong> <span style="background: {HIGHLIGHT_COLORS['field']}; padding: 2px 6px; border-radius: 3px;">{few_shot_count} examples</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # For other fields, compare with original row data if available
                    if original_row is not None and field in original_row.index:
                        original_val = str(original_row[field])
                        is_modified = str(value) != str(original_val)

                        # Check what operations were applied to this field
                        operation_types = []
                        for template_source in [template_config, original_template]:
                            if isinstance(template_source, dict) and field in template_source:
                                field_config = template_source[field]
                                if isinstance(field_config, list):
                                    operation_types = field_config
                                    break

                        operation_desc = f" ({', '.join(operation_types)})" if operation_types else ""

                        if is_modified:
                            st.markdown(f"""
                            <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #667eea;">
                                <strong style="color: #1976d2;">{field}{operation_desc}:</strong><br>
                                <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px; text-decoration: line-through; opacity: 0.7;">{original_val}</span><br>
                                <span style="background: {HIGHLIGHT_COLORS['variation']}; padding: 2px 6px; border-radius: 3px; font-weight: bold;">→ {value}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #667eea;">
                                <strong style="color: #1976d2;">{field}{operation_desc}:</strong> <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px;">{value}</span>
                            </div>
                            """, unsafe_allow_html=True)

            # Show gold field updates (when fields like options are shuffled)
            gold_updates = variation.get('gold_updates', {})
            for gold_field, new_value in gold_updates.items():
                original_val = str(
                    original_row[gold_field]) if original_row is not None and gold_field in original_row.index else None
                if original_val is not None and str(new_value) != str(original_val):
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #e74c3c;">
                        <strong style="color: #c0392b;">{gold_field} (updated):</strong><br>
                        <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px; text-decoration: line-through; opacity: 0.7;">{original_val}</span><br>
                        <span style="background: #ffebee; padding: 2px 6px; border-radius: 3px; font-weight: bold; color: #e74c3c;">→ {new_value}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #4caf50;">
                        <strong style="color: #388e3c;">{gold_field}:</strong>
                        <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px;">{new_value}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Show any additional fields from original data that weren't used in field_values
            if original_row is not None:
                gold_updates = variation.get('gold_updates', {})
                for col in original_row.index:
                    if col not in field_values and col not in gold_updates and str(original_row[col]).strip():
                        original_val = str(original_row[col])
                        st.markdown(f"""
                        <div style="margin: 0.5rem 0; padding: 0.5rem; background: white; border-radius: 4px; border-left: 3px solid #667eea;">
                            <strong style="color: #1976d2;">{col}:</strong> <span style="background: {HIGHLIGHT_COLORS['original']}; padding: 2px 6px; border-radius: 3px;">{original_val}</span>
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

    # Enhanced export options with cards - 2x2 layout
    col1, col2 = st.columns(2)

    with col1:
        # Row 1 - JSON and CSV
        subcol1, subcol2 = st.columns(2)

        with subcol1:
            st.markdown("""
            <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; margin-bottom: 1rem;">
                <h4 style="color: #ff9800; margin: 0;">📋 JSON Format</h4>
                <p style="margin: 0.5rem 0; color: #666; font-size: 0.9rem;">Complete data with metadata</p>
            </div>
            """, unsafe_allow_html=True)

            # Enhance variations with conversation field to match API format
            enhanced_variations = MultiPromptify._prepare_variations_for_conversation_export(variations)
            json_data = json.dumps(enhanced_variations, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="prompt_variations.json",
                mime="application/json",
                use_container_width=True
            )

        with subcol2:
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
                    # For CSV, we need to handle different data types appropriately
                    if isinstance(value, list):
                        # Convert lists to JSON string for CSV compatibility
                        flat_var[f'field_{key}'] = json.dumps(value)
                        flat_var[f'field_{key}_type'] = 'list'
                    else:
                        flat_var[f'field_{key}'] = value
                        flat_var[f'field_{key}_type'] = type(value).__name__

                # Add gold updates if present
                gold_updates = var.get('gold_updates')
                if gold_updates:
                    for gold_field, gold_value in gold_updates.items():
                        flat_var[f'gold_update_{gold_field}'] = gold_value

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

    with col2:
        # Row 2 - Text and Conversation formats
        subcol1, subcol2 = st.columns(2)

        with subcol1:
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


def convert_to_conversation_format(variations, original_data=None):
    """
    Convert variations to conversation format with user/assistant messages.
    
    Args:
        variations: List of generated variations
        original_data: Optional DataFrame with original data for extracting original answers
        
    Returns:
        List of conversation objects with metadata
    """
    conversations = []

    for variation in variations:
        # Parse conversation from existing conversation field or prompt
        if 'conversation' in variation and variation['conversation']:
            conversation = variation['conversation']
        else:
            # Build conversation from prompt
            prompt = variation.get('prompt', '')

            # Split prompt into conversation parts if it contains few-shot examples
            parts = prompt.split('\n\n')
            conversation = []

            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue

                # Check if this is the last part (incomplete question)
                if i == len(parts) - 1:
                    # Last part - this is the question without answer
                    conversation.append({
                        "role": "user",
                        "content": part
                    })
                else:
                    # This is a complete Q&A pair
                    # Split by the last occurrence of newline to separate question and answer
                    lines = part.split('\n')
                    if len(lines) >= 2:
                        # Assume the last line is the answer
                        answer = lines[-1].strip()
                        question = '\n'.join(lines[:-1]).strip()

                        conversation.append({
                            "role": "user",
                            "content": question
                        })
                        conversation.append({
                            "role": "assistant",
                            "content": answer
                        })
                    else:
                        # Single line - treat as user message
                        conversation.append({
                            "role": "user",
                            "content": part
                        })

        # Get the correct answer after updates
        correct_answer = None
        gold_updates = variation.get('gold_updates')
        if gold_updates:
            # Get the first (and usually only) gold field update
            for gold_field, gold_value in gold_updates.items():
                correct_answer = gold_value
                break
        else:
            # No updates, try to extract original answer from original data
            if original_data is not None:
                original_row_index = variation.get('original_row_index', 0)
                template_config = variation.get('template_config', {})
                gold_config = template_config.get('gold')

                if gold_config and original_row_index < len(original_data):
                    # Extract gold field name
                    if isinstance(gold_config, str):
                        gold_field = gold_config
                    elif isinstance(gold_config, dict) and 'field' in gold_config:
                        gold_field = gold_config['field']
                    else:
                        gold_field = None

                    if gold_field and gold_field in original_data.columns:
                        original_answer = original_data.iloc[original_row_index][gold_field]
                        correct_answer = str(original_answer)

        # Create conversation with minimal metadata
        conversation_item = {
            "conversation": conversation,
            "metadata": {
                "original_row_index": variation.get('original_row_index', 0),
                "variation_count": variation.get('variation_count', 1),
            }
        }

        # Always add correct_answer if we have it
        if correct_answer is not None:
            conversation_item["metadata"]["correct_answer"] = correct_answer

        conversations.append(conversation_item)

    return conversations


def display_conversation_format(variations, original_data=None):
    """Display variations in conversation format within the UI"""
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #2196f3; margin-bottom: 2rem;">
        <h3 style="color: #1976d2; margin-top: 0;">💬 Conversation Format Display</h3>
        <p style="margin-bottom: 0; color: #0d47a1;">Each variation displayed as a conversation with role and content structure</p>
    </div>
    """, unsafe_allow_html=True)

    if not variations:
        st.warning("No variations to display")
        return

    # Convert to conversation format with original data for correct answers
    conversations = convert_to_conversation_format(variations, original_data)

    st.markdown("---")

    # Pagination controls
    total_conversations = len(conversations)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📄 Navigation")
        page_options = [5, 10, 20, 50]
        items_per_page = st.selectbox("Conversations per page", page_options, index=1, key="conv_per_page")

        total_pages = (total_conversations - 1) // items_per_page + 1 if total_conversations > 0 else 1
        page = st.number_input(f"Page (1-{total_pages})", min_value=1, max_value=total_pages, value=1, key="conv_page")
        if page is None:
            page = 1

    # Calculate range
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_conversations)

    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
        <strong>Showing conversations {start_idx + 1}-{end_idx} of {total_conversations:,}</strong>
    </div>
    """, unsafe_allow_html=True)

    # Display conversations
    for i in range(start_idx, end_idx):
        conversation = conversations[i]
        original_variation = variations[i]
        display_single_conversation(conversation, i + 1, original_variation)
