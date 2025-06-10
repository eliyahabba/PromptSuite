"""
Step 4: Show Results for MultiPromptify 2.0
"""

import streamlit as st
import json
import pandas as pd
from typing import Dict, List, Any


def render():
    """Render the results viewing interface"""
    if not st.session_state.get('variations_generated', False):
        st.error("⚠️ Please generate variations first (Step 3)")
        if st.button("← Go to Step 3"):
            st.session_state.page = 3
            st.rerun()
        return
    
    # Enhanced header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; text-align: center;">
            📊 Step 4: View Results
        </h1>
        <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 0.5rem 0 0 0;">
            Explore and analyze your generated prompt variations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    variations = st.session_state.generated_variations
    stats = st.session_state.generation_stats
    generation_time = st.session_state.generation_time
    
    # Main metrics display
    display_metrics(variations, stats, generation_time)
    
    # Add visual separator
    st.markdown("---")
    
    # Navigation tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Browse Variations", 
        "📊 Analysis & Stats", 
        "🎯 Few-shot Examples", 
        "💾 Export Data"
    ])
    
    with tab1:
        browse_variations(variations)
    
    with tab2:
        show_analysis(variations, stats)
    
    with tab3:
        analyze_few_shot_examples(variations)
    
    with tab4:
        export_options(variations)


def display_metrics(variations: List[Dict], stats: Dict, generation_time: float):
    """Display key metrics in an attractive format"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{}</h2>
            <p style="margin: 0; opacity: 0.8;">Total Variations</p>
        </div>
        """.format(len(variations)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{}</h2>
            <p style="margin: 0; opacity: 0.8;">Original Rows</p>
        </div>
        """.format(stats.get('original_rows', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;">
            <h2 style="margin: 0; font-size: 2rem;">{:.1f}</h2>
            <p style="margin: 0; opacity: 0.8;">Avg per Row</p>
        </div>
        """.format(stats.get('avg_variations_per_row', 0)), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: #333;">
            <h2 style="margin: 0; font-size: 2rem;">{:.1f}s</h2>
            <p style="margin: 0; opacity: 0.7;">Generation Time</p>
        </div>
        """.format(generation_time), unsafe_allow_html=True)


def browse_variations(variations: List[Dict]):
    """Browse through generated variations with search and filter"""
    st.subheader("🔍 Browse Variations")
    
    # Search and filter controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input(
            "🔍 Search in prompts",
            placeholder="Enter search term...",
            help="Search for specific content in the generated prompts"
        )
    
    with col2:
        # Filter by original row
        available_rows = sorted(set(var.get('original_row_index', 0) for var in variations))
        selected_row = st.selectbox(
            "📊 Filter by original row",
            options=["All"] + [f"Row {i}" for i in available_rows],
            help="Show variations from specific original data row"
        )
    
    with col3:
        # Items per page
        items_per_page = st.selectbox(
            "📄 Items per page",
            options=[10, 25, 50, 100],
            index=1,
            help="Number of variations to display per page"
        )
    
    # Filter variations based on search and selection
    filtered_variations = variations
    
    if search_term:
        filtered_variations = [
            var for var in filtered_variations 
            if search_term.lower() in var['prompt'].lower()
        ]
    
    if selected_row != "All":
        row_idx = int(selected_row.split()[1])
        filtered_variations = [
            var for var in filtered_variations 
            if var.get('original_row_index') == row_idx
        ]
    
    st.info(f"📊 Showing {len(filtered_variations)} variations (filtered from {len(variations)} total)")
    
    # Pagination
    total_pages = (len(filtered_variations) - 1) // items_per_page + 1 if filtered_variations else 0
    
    if total_pages > 1:
        page = st.selectbox(
            f"Page (1-{total_pages})",
            options=list(range(1, total_pages + 1)),
            index=0
        )
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_variations = filtered_variations[start_idx:end_idx]
    else:
        page_variations = filtered_variations[:items_per_page]
    
    # Display variations
    if page_variations:
        for i, variation in enumerate(page_variations):
            with st.expander(f"🔍 Variation {i+1} (Row {variation.get('original_row_index', 'N/A')})", expanded=(i == 0)):
                
                # Prompt display with syntax highlighting
                st.markdown("**📝 Generated Prompt:**")
                st.code(variation['prompt'], language="text")
                
                # Metadata
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📋 Metadata:**")
                    metadata_display = {
                        "Original Row": variation.get('original_row_index', 'N/A'),
                        "Variation #": variation.get('variation_count', 'N/A'),
                        "Varied Fields": variation.get('varied_fields', [])
                    }
                    for key, value in metadata_display.items():
                        st.markdown(f"- **{key}**: {value}")
                
                with col2:
                    st.markdown("**🔧 Field Values:**")
                    field_values = variation.get('field_values', {})
                    
                    # Show few-shot separately if present
                    for field, value in field_values.items():
                        if field == 'few_shot' and value:
                            st.markdown(f"- **{field}**: [Few-shot examples - see Few-shot tab]")
                        elif isinstance(value, str) and len(value) > 100:
                            st.markdown(f"- **{field}**: {value[:100]}...")
                        else:
                            st.markdown(f"- **{field}**: {value}")
    else:
        st.info("No variations match your search criteria.")


def show_analysis(variations: List[Dict], stats: Dict):
    """Show detailed analysis and statistics"""
    st.subheader("📊 Analysis & Statistics")
    
    # Basic statistics
    st.markdown("### 📈 Generation Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Variation Distribution:**")
        
        # Count variations per original row
        row_counts = {}
        for var in variations:
            row_idx = var.get('original_row_index', 0)
            row_counts[row_idx] = row_counts.get(row_idx, 0) + 1
        
        if row_counts:
            df_dist = pd.DataFrame([
                {"Row": f"Row {k}", "Variations": v} 
                for k, v in sorted(row_counts.items())
            ])
            st.dataframe(df_dist, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Field Analysis:**")
        
        # Analyze which fields were varied
        varied_fields_count = {}
        for var in variations:
            varied_fields = var.get('varied_fields', [])
            for field in varied_fields:
                varied_fields_count[field] = varied_fields_count.get(field, 0) + 1
        
        if varied_fields_count:
            df_fields = pd.DataFrame([
                {"Field": k, "Occurrences": v} 
                for k, v in sorted(varied_fields_count.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_fields, use_container_width=True)
    
    # Detailed statistics
    st.markdown("### 📋 Detailed Statistics")
    
    detailed_stats = {
        "Total Variations": len(variations),
        "Unique Original Rows": len(set(var.get('original_row_index', 0) for var in variations)),
        "Average Variations per Row": stats.get('avg_variations_per_row', 0),
        "Min Variations per Row": stats.get('min_variations_per_row', 0),
        "Max Variations per Row": stats.get('max_variations_per_row', 0),
        "Total Fields Used": len(stats.get('field_names', [])),
        "Fields with Variations": len(set().union(*[var.get('varied_fields', []) for var in variations]))
    }
    
    df_stats = pd.DataFrame([
        {"Metric": k, "Value": v} 
        for k, v in detailed_stats.items()
    ])
    st.dataframe(df_stats, use_container_width=True)


def analyze_few_shot_examples(variations: List[Dict]):
    """Analyze and display few-shot examples usage"""
    st.subheader("🎯 Few-shot Examples Analysis")
    
    # Find variations with few-shot examples
    few_shot_variations = [
        var for var in variations 
        if var.get('field_values', {}).get('few_shot')
    ]
    
    if not few_shot_variations:
        st.info("🔍 No few-shot examples found in the generated variations.")
        return
    
    st.success(f"✅ Found {len(few_shot_variations)} variations with few-shot examples")
    
    # Analyze few-shot usage patterns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Few-shot Usage Statistics:**")
        
        # Check if same examples for all or different per row
        few_shot_examples = [var['field_values']['few_shot'] for var in few_shot_variations]
        unique_examples = set(few_shot_examples)
        
        stats_data = {
            "Total with few-shot": len(few_shot_variations),
            "Unique example sets": len(unique_examples),
            "Pattern": "Same for all" if len(unique_examples) == 1 else "Different per row"
        }
        
        for key, value in stats_data.items():
            st.markdown(f"- **{key}**: {value}")
    
    with col2:
        st.markdown("**🔍 Example Length Analysis:**")
        
        # Analyze length of examples
        example_lengths = [len(var['field_values']['few_shot']) for var in few_shot_variations]
        
        if example_lengths:
            length_stats = {
                "Average length": f"{sum(example_lengths) / len(example_lengths):.0f} chars",
                "Min length": f"{min(example_lengths)} chars", 
                "Max length": f"{max(example_lengths)} chars"
            }
            
            for key, value in length_stats.items():
                st.markdown(f"- **{key}**: {value}")
    
    # Display sample few-shot examples
    st.markdown("### 👀 Sample Few-shot Examples")
    
    # Show a few examples
    sample_size = min(3, len(few_shot_variations))
    
    for i in range(sample_size):
        var = few_shot_variations[i]
        few_shot_content = var['field_values']['few_shot']
        row_idx = var.get('original_row_index', 'N/A')
        
        with st.expander(f"📋 Few-shot Examples for Row {row_idx}", expanded=(i == 0)):
            st.markdown("**Examples content:**")
            st.text(few_shot_content)
            
            # Count number of examples
            example_count = few_shot_content.count('\n\n') + 1 if few_shot_content else 0
            st.info(f"📊 Contains approximately {example_count} examples")
    
    if len(few_shot_variations) > sample_size:
        st.info(f"💡 Showing {sample_size} samples. Total {len(few_shot_variations)} variations have few-shot examples.")


def export_options(variations: List[Dict]):
    """Provide various export options"""
    st.subheader("💾 Export Data")
    
    st.markdown("""
    <div style="background-color: #e3f2fd; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <h4 style="color: #1976d2; margin-top: 0;">📁 Export Your Generated Variations</h4>
        <p style="margin-bottom: 0; color: #0d47a1;">Choose your preferred format to download the results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Export format selection
    col1, col2, col3 = st.columns(3)
    
    # JSON Export
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color: #ff9800;">📋 JSON Format</h4>
            <p style="color: #666; font-size: 0.9rem;">Complete data with all metadata</p>
        </div>
        """, unsafe_allow_html=True)
        
        json_data = json.dumps(variations, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="multipromptify_variations.json",
            mime="application/json",
            use_container_width=True,
            help="Downloads all variations with complete metadata"
        )
    
    # CSV Export
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color: #4caf50;">📊 CSV Format</h4>
            <p style="color: #666; font-size: 0.9rem;">Spreadsheet compatible format</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Flatten data for CSV
        flattened_data = []
        for var in variations:
            flat_var = {
                'prompt': var['prompt'],
                'original_row_index': var.get('original_row_index', ''),
                'variation_count': var.get('variation_count', ''),
            }
            
            # Add field values
            field_values = var.get('field_values', {})
            for key, value in field_values.items():
                if key == 'few_shot' and value:
                    # Truncate long few-shot examples for CSV
                    flat_var[f'field_{key}'] = value[:200] + "..." if len(str(value)) > 200 else value
                else:
                    flat_var[f'field_{key}'] = value
            
            flattened_data.append(flat_var)
        
        csv_df = pd.DataFrame(flattened_data)
        csv_data = csv_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="multipromptify_variations.csv",
            mime="text/csv",
            use_container_width=True,
            help="Downloads flattened data suitable for spreadsheets"
        )
    
    # Text Export
    with col3:
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;">
            <h4 style="color: #9c27b0;">📝 Text Format</h4>
            <p style="color: #666; font-size: 0.9rem;">Plain text prompts only</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create text with separators
        text_content = []
        for i, var in enumerate(variations):
            text_content.append(f"=== VARIATION {i+1} (Row {var.get('original_row_index', 'N/A')}) ===")
            text_content.append(var['prompt'])
            text_content.append("")  # Empty line separator
        
        text_data = "\n".join(text_content)
        
        st.download_button(
            label="📥 Download TXT",
            data=text_data,
            file_name="multipromptify_variations.txt",
            mime="text/plain",
            use_container_width=True,
            help="Downloads just the prompts in plain text format"
        )
    
    # Additional export options
    st.markdown("---")
    st.markdown("### 🔧 Advanced Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Custom filtering for export
        st.markdown("**🎯 Custom Export Filtering:**")
        
        # Filter by original row
        available_rows = sorted(set(var.get('original_row_index', 0) for var in variations))
        selected_rows = st.multiselect(
            "Select specific rows to export",
            options=[f"Row {i}" for i in available_rows],
            help="Leave empty to export all rows"
        )
        
        # Filter by field presence
        all_fields = set()
        for var in variations:
            all_fields.update(var.get('field_values', {}).keys())
        
        required_fields = st.multiselect(
            "Export only variations containing these fields",
            options=sorted(all_fields),
            help="Leave empty to export all variations"
        )
        
        if st.button("📤 Apply Filters & Export"):
            filtered_vars = variations
            
            # Apply row filter
            if selected_rows:
                selected_indices = [int(row.split()[1]) for row in selected_rows]
                filtered_vars = [var for var in filtered_vars if var.get('original_row_index') in selected_indices]
            
            # Apply field filter
            if required_fields:
                filtered_vars = [
                    var for var in filtered_vars 
                    if all(field in var.get('field_values', {}) for field in required_fields)
                ]
            
            if filtered_vars:
                st.success(f"✅ Filtered to {len(filtered_vars)} variations")
                
                # Provide download for filtered data
                filtered_json = json.dumps(filtered_vars, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download Filtered JSON",
                    data=filtered_json,
                    file_name="multipromptify_filtered_variations.json",
                    mime="application/json"
                )
            else:
                st.warning("⚠️ No variations match the selected filters")
    
    with col2:
        # Export statistics
        st.markdown("**📊 Export Summary:**")
        export_stats = {
            "Total variations available": len(variations),
            "Estimated JSON size": f"{len(json.dumps(variations)) / 1024:.1f} KB",
            "Estimated CSV size": f"{len(csv_data) / 1024:.1f} KB",
            "Variations with few-shot": len([v for v in variations if v.get('field_values', {}).get('few_shot')]),
            "Unique original rows": len(set(var.get('original_row_index', 0) for var in variations))
        }
        
        for key, value in export_stats.items():
            st.markdown(f"- **{key}**: {value}")
    
    # Tips for using exported data
    with st.expander("💡 Tips for Using Exported Data"):
        st.markdown("""
        **JSON Format:**
        - Contains complete metadata and field values
        - Best for programmatic use and analysis
        - Can be easily loaded back into Python/MultiPromptify
        
        **CSV Format:**
        - Easy to open in Excel, Google Sheets, etc.
        - Good for manual inspection and basic analysis
        - Few-shot examples may be truncated for readability
        
        **Text Format:**
        - Just the generated prompts, no metadata
        - Perfect for direct use in other AI systems
        - Smallest file size
        
        **Programming Example:**
        ```python
        import json
        
        # Load variations back into Python
        with open('multipromptify_variations.json', 'r') as f:
            variations = json.load(f)
        
        # Extract just the prompts
        prompts = [var['prompt'] for var in variations]
        ```
        """) 