"""
Step 2: Template Builder for MultiPromptify 2.0
"""

import streamlit as st
import re
from src.multipromptify import MultiPromptify


def render():
    """Render the template builder interface"""
    if not st.session_state.get('data_loaded', False):
        st.error("⚠️ Please upload data first (Step 1)")
        if st.button("← Go to Step 1"):
            st.session_state.page = 1
            st.rerun()
        return
    
    st.markdown('<div class="step-header"><h2>🔧 Step 2: Build Your Template</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Templates</strong> define how your prompts are structured and which parts should be varied.
        Use <code>{field_name:variation_type}</code> syntax to specify variations.
    </div>
    """, unsafe_allow_html=True)
    
    # Get the uploaded data
    df = st.session_state.uploaded_data
    available_columns = df.columns.tolist()
    
    # Template building interface - simplified to 2 tabs only
    tab1, tab2 = st.tabs(["🎯 Template Suggestions", "✏️ Custom Template"])
    
    with tab1:
        template_suggestions_interface(available_columns)
    
    with tab2:
        custom_template_interface(available_columns)
    
    # Show selected template details prominently at the bottom
    if st.session_state.get('template_ready', False):
        display_selected_template_details(available_columns)


def template_suggestions_interface(available_columns):
    """Interface for selecting template suggestions"""
    st.subheader("Choose a Template Suggestion")
    st.write("Select a pre-built template that matches your data structure and task type")
    
    # Show currently selected template at the top
    if st.session_state.get('template_ready', False):
        selected_name = st.session_state.get('template_name', 'Unknown')
        selected_template = st.session_state.get('selected_template', '')
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">✅ Currently Selected: {selected_name}</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-family: monospace; font-size: 0.9rem;">
                {selected_template[:100]}{"..." if len(selected_template) > 100 else ""}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    suggestions = st.session_state.template_suggestions
    
    # Create tabs for each category
    category_tabs = []
    category_data = []
    
    for category_key, category_info in suggestions.items():
        category_tabs.append(f"📋 {category_info['category_name']}")
        category_data.append((category_key, category_info))
    
    # Create tabs for categories
    tabs = st.tabs(category_tabs)
    
    for i, (tab, (category_key, category_info)) in enumerate(zip(tabs, category_data)):
        with tab:
            st.write(f"**{category_info['description']}**")
            
            # Filter templates based on available columns for this category
            compatible_templates = []
            incompatible_templates = []
            
            for template in category_info['templates']:
                # Extract field names from template (without variation types)
                field_pattern = r'\{([^:}]+)(?::[^}]+)?\}'
                template_fields = re.findall(field_pattern, template['template'])
                required_fields = [f for f in template_fields if f not in ['instruction', 'few_shot']]
                
                # Check if we have the required columns
                missing_fields = set(required_fields) - set(available_columns)
                if not missing_fields:
                    compatible_templates.append(template)
                else:
                    template['missing_fields'] = missing_fields
                    incompatible_templates.append(template)
            
            if compatible_templates:
                st.success(f"✅ Found {len(compatible_templates)} compatible {category_info['category_name']} templates")
                
                for template in compatible_templates:
                    # Check if this is the currently selected template
                    is_selected = (st.session_state.get('template_ready', False) and 
                                  st.session_state.get('selected_template', '') == template['template'])
                    
                    # Style the expander differently if selected
                    if is_selected:
                        expander_label = f"✅ {template['name']} (Currently Selected)"
                    else:
                        expander_label = f"📋 {template['name']}"
                    
                    with st.expander(expander_label, expanded=is_selected):
                        st.write(f"**Description:** {template['description']}")
                        st.code(template['template'], language="text")
                        
                        # Show which columns will be used
                        field_pattern = r'\{([^:}]+)(?::([^}]+))?\}'
                        matches = re.findall(field_pattern, template['template'])
                        
                        st.write("**Template fields:**")
                        for field, variation_type in matches:
                            if field in available_columns:
                                status = "✅ Available"
                                color = "green"
                            elif field in ['instruction', 'few_shot']:
                                status = "⚙️ User input"
                                color = "blue"
                            else:
                                status = "❌ Missing"
                                color = "red"
                            
                            variation_info = f" (variation: {variation_type})" if variation_type else ""
                            st.markdown(f"- **{field}**{variation_info}: <span style='color: {color}'>{status}</span>", unsafe_allow_html=True)
                        
                        # Button styling based on selection
                        button_key = f"template_{category_key}_{template['name'].lower().replace(' ', '_')}"
                        if is_selected:
                            if st.button(f"🔄 Re-select {template['name']}", key=f"re_{button_key}"):
                                st.session_state.selected_template = template['template']
                                st.session_state.template_name = template['name']
                                st.session_state.template_ready = True
                                st.success(f"✅ Re-selected {template['name']} template")
                                st.rerun()
                        else:
                            if st.button(f"✨ Select {template['name']}", key=button_key, type="primary"):
                                st.session_state.selected_template = template['template']
                                st.session_state.template_name = template['name']
                                st.session_state.template_ready = True
                                st.success(f"✅ Selected {template['name']} template")
                                st.rerun()
            
            # Show incompatible templates if any
            if incompatible_templates:
                with st.expander(f"⚠️ {category_info['category_name']} templates requiring additional columns"):
                    for template in incompatible_templates:
                        st.write(f"**{template['name']}**: Missing columns: {', '.join(template.get('missing_fields', []))}")
                        st.code(template['template'], language="text")


def custom_template_interface(available_columns):
    """Interface for building custom templates"""
    st.subheader("Build a Custom Template")
    
    # Template explanation
    with st.expander("📚 Template Syntax Guide", expanded=False):
        st.markdown("""
        **Template Syntax:**
        - `{field_name}` - Use field as-is (no variations)
        - `{field_name:paraphrase}` - Generate paraphrase variations  (meaning-preserving text variations)
        - `{field_name:non-semantic}` - Generate non-semantic/structural formatting variations
        - `{field_name:surface}` - Generate surface-level formatting variations (spacing, punctuation)
        - `{field_name:context}` - Add contextual variations
        - `{field_name:multiple-choice}` - Generate multiple choice formatting variations
        - `{field_name:multidoc}` - Generate multi-document order variations
        
        **Special fields:**
        - `{instruction}` - Static instruction text you'll provide
        
        **Few-shot Examples (New Simplified Syntax):**
        - `{few_shot:3}` - 3 few-shot examples (automatically selected from data)
        - `{few_shot:[5]}` - 5 different examples per data row (list format)
        - `{few_shot:(2)}` - 2 same examples for all rows (tuple format)
        - `{few_shot:train[4]}` - 4 examples from train split, different per row
        - `{few_shot:test(3)}` - 3 examples from test split, same for all
        
        **Few-shot Data Sources:**
        The system will automatically use examples from your data:
        - If you have a 'split' column (train/test), it will respect the split specification
        - Otherwise, it will sample from all available data
        - Examples will exclude the current row to avoid data leakage
        
        **Note:** For paraphrase variations, you'll need to provide an API key in the generation step.
        """)
    
    # Available columns display
    st.write("**Available columns in your data:**")
    col_display = st.columns(min(len(available_columns), 4))
    for i, col in enumerate(available_columns):
        with col_display[i % 4]:
            st.code(f"{{{col}}}")
    
    # Show available variation types
    st.write("**Available variation types:**")
    st.markdown("""
    - **paraphrase** - Meaning-preserving text variations (requires API key) - **recommended for instructions**
    - **surface/non-semantic** - Formatting variations: spacing, typos, punctuation - **recommended for questions/text**
    - **context** - Contextual additions and modifications
    - **few-shot** - Few-shot example variations
    - **multiple-choice** - Multiple choice question formatting
    - **multidoc** - Document order variations
    """)
    
    st.info("💡 **Recommended:** Use `paraphrase` for instructions and `surface` for questions/text content")
    
    # Template builder
    st.subheader("Template Editor")
    
    # Provide a starting template based on common patterns
    default_template = ""
    if 'question' in available_columns and 'answer' in available_columns:
        default_template = "{instruction:paraphrase}: {question:surface}\nAnswer: {answer}"
    elif 'text' in available_columns and 'label' in available_columns:
        default_template = "{instruction:paraphrase}: \"{text:surface}\"\nLabel: {label}"
    elif len(available_columns) >= 2:
        col1, col2 = available_columns[:2]
        default_template = f"{{instruction:paraphrase}}: {{{col1}:surface}}\nOutput: {{{col2}}}"
    else:
        default_template = "{instruction:paraphrase}: {" + available_columns[0] + ":surface}"
    
    template = st.text_area(
        "Template",
        placeholder=st.session_state.get('selected_template', default_template),
        height=150,
        help="Use curly braces to reference columns from your data. Add :variation_type to specify how each field should be varied."
    )
    # Set default template if empty
    if not template:
        template = st.session_state.get('selected_template', default_template)
    
    # Template validation
    if template:
        validate_template(template, available_columns)
    
    # Save template button
    if st.button("Save Template"):
        if template.strip():
            st.session_state.selected_template = template
            st.session_state.template_name = "Custom Template"
            st.session_state.template_ready = True
            st.success("✅ Template saved successfully!")
            st.rerun()
        else:
            st.error("Please enter a template")

    # Show current custom template status
    if (st.session_state.get('template_ready', False) and 
        st.session_state.get('template_name', '') == "Custom Template"):
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                    padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <h4 style="color: white; margin: 0;">✅ Your Custom Template is Active</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                This template is ready for use in the next step
            </p>
        </div>
        """, unsafe_allow_html=True)


def display_selected_template_details(available_columns):
    """Display selected template details prominently at the bottom"""
    st.markdown("---")
    
    template = st.session_state.selected_template
    template_name = st.session_state.get('template_name', 'Custom Template')
    df = st.session_state.uploaded_data
    
    # Main selected template display with enhanced styling
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h2 style="color: white; margin: 0; text-align: center;">🎯 Selected Template: {template_name}</h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; margin: 0.5rem 0 0 0;">
            Your template is ready for generating variations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Template content in a styled container
    st.markdown("### 📝 Template Content")
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid #dee2e6;">
    """, unsafe_allow_html=True)
    st.code(template, language="text")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Template analysis
    mp = MultiPromptify()
    try:
        is_valid, errors = mp.template_parser.validate_template(template)
        
        if is_valid:
            # Parse template to show fields
            fields = mp.template_parser.parse(template)
            variation_fields = mp.template_parser.get_variation_fields()
            required_columns = mp.template_parser.get_required_columns()
            
            # Analysis in two columns with enhanced styling
            st.markdown("### 🔍 Template Analysis")
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #007bff;">
                    <h4 style="color: #007bff; margin-top: 0;">📋 Template Fields</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for field in fields:
                    if field.variation_type:
                        st.markdown(f"- **`{field.name}`** → {field.variation_type} variations")
                    else:
                        st.markdown(f"- **`{field.name}`** → no variations")
            
            with col2:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h4 style="color: #28a745; margin-top: 0;">✅ Data Requirements</h4>
                </div>
                """, unsafe_allow_html=True)
                
                missing_cols = required_columns - set(df.columns)
                if missing_cols:
                    st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                else:
                    st.success("✅ All required columns available")
                    
                # Show available columns count
                st.info(f"📊 Using {len(required_columns)} data columns from your dataset")
            
            # Variation summary
            if variation_fields:
                # API key requirement check
                if any(var_type in ['paraphrase'] for var_type in variation_fields.values()):
                    st.info("🔑 This template uses paraphrase variations - you'll need to provide an API key in the next step.")
            
        else:
            st.error("❌ Template has validation errors:")
            for error in errors:
                st.error(f"- {error}")
            return
            
    except Exception as e:
        st.error(f"Error analyzing template: {str(e)}")
        return
    
    # Continue button with enhanced styling
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h4 style="color: #495057;">🚀 Ready to generate variations?</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Continue to Generate Variations →", type="primary", use_container_width=True):
            st.session_state.page = 3
            st.rerun()


def validate_template(template, available_columns):
    """Validate template and show feedback"""
    try:
        mp = MultiPromptify()
        is_valid, errors = mp.template_parser.validate_template(template)
        
        if is_valid:
            # Check column availability
            required_columns = mp.template_parser.get_required_columns()
            missing_cols = required_columns - set(available_columns)
            
            if missing_cols:
                st.warning(f"⚠️ Template references missing columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ Template is valid and all columns are available")
                
            # Show field analysis
            variation_fields = mp.template_parser.get_variation_fields()
            if variation_fields:
                st.info(f"Fields with variations: {', '.join(f'{k}:{v}' for k, v in variation_fields.items())}")
        else:
            st.error("❌ Template validation errors:")
            for error in errors:
                st.error(f"- {error}")
                
    except Exception as e:
        st.error(f"Template validation error: {str(e)}") 