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
        
        # Handle template display for new format
        if isinstance(selected_template, dict):
            # New format - show brief preview
            template_preview = f"Instruction: {selected_template.get('instruction', '')[:50]}..."
        else:
            # Old format - show first 100 characters
            template_preview = f"{selected_template[:100]}{'...' if len(selected_template) > 100 else ''}"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">✅ Currently Selected: {selected_name}</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-family: monospace; font-size: 0.9rem;">
                {template_preview}
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
                
                # Handle new template format (dictionary) vs old format (string)
                if isinstance(template['template'], dict):
                    # New format - check both instruction and template parts
                    instruction_text = template['template'].get('instruction', '')
                    template_text = template['template'].get('template', '')
                    
                    # Get fields from both parts
                    instruction_fields = re.findall(field_pattern, instruction_text)
                    template_fields = re.findall(field_pattern, template_text)
                    
                    # Combine and get unique fields, excluding 'instruction' and 'few_shot'
                    all_template_fields = list(set(instruction_fields + template_fields))
                    required_fields = [f for f in all_template_fields if f not in ['instruction', 'few_shot']]
                else:
                    # Old format - single template string
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
                    current_selected = st.session_state.get('selected_template', '')
                    
                    # Compare templates properly - handle both old and new formats
                    is_selected = False
                    if st.session_state.get('template_ready', False):
                        if isinstance(template['template'], dict) and isinstance(current_selected, dict):
                            # Both are new format - compare structure
                            is_selected = (template['template'].get('instruction') == current_selected.get('instruction') and
                                         template['template'].get('template') == current_selected.get('template'))
                        elif not isinstance(template['template'], dict) and not isinstance(current_selected, dict):
                            # Both are old format - direct comparison
                            is_selected = (template['template'] == current_selected)
                        # If formats don't match, they're not the same
                    
                    # Style the expander differently if selected
                    if is_selected:
                        expander_label = f"✅ {template['name']} (Currently Selected)"
                    else:
                        expander_label = f"📋 {template['name']}"
                    
                    with st.expander(expander_label, expanded=is_selected):
                        st.write(f"**Description:** {template['description']}")
                        
                        # Display template based on format
                        if isinstance(template['template'], dict):
                            # New format - show instruction and template separately
                            st.markdown("**Instruction:**")
                            st.code(template['template'].get('instruction', ''), language="text")
                            st.markdown("**Processing Template:**")
                            st.code(template['template'].get('template', ''), language="text")
                            
                            # For field analysis, check both parts
                            instruction_text = template['template'].get('instruction', '')
                            template_text = template['template'].get('template', '')
                            
                            # Get fields from both parts for analysis
                            field_pattern = r'\{([^:}]+)(?::([^}]+))?\}'
                            instruction_matches = re.findall(field_pattern, instruction_text)
                            template_matches = re.findall(field_pattern, template_text)
                            
                            # Combine matches
                            matches = list(set(instruction_matches + template_matches))
                        else:
                            # Old format - single template
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
    with st.expander("📚 Template System Guide", expanded=False):
        st.markdown("""
        **Two-Part Template System:**
        
        **1. Instruction (Template with Data Placeholders):**
        - Define your instruction with placeholders for your data columns
        - Example: `"Please answer: Question:{question}\\nOptions:{options}\\nAnswer:{answer}"`
        - Use `{field_name}` to reference columns from your data
        - This creates the base structure of your prompt
        
        **2. Template (Processing Rules):**
        - Define how to process the instruction and individual fields
        - Example: `{instruction:paraphrase}\\n\\n{few_shot:(2)}\\n\\n{question:surface}`
        - `{instruction:paraphrase}` - Apply paraphrase variations to your entire instruction
        - `{field_name:variation_type}` - Apply specific variations to individual fields
        
        **Complete Example:**
        ```
        Instruction: "Please select the correct answer: Question:{question}\\nOptions:{options}\\nAnswer:{answer}"
        Template: {instruction:paraphrase}\\n\\n{few_shot:(2)}\\n\\n{question:surface}
        ```
        
        **Available Variations:**
        - **paraphrase** - Meaning-preserving text variations (requires API key) - **recommended for instructions**
        - **surface/non-semantic** - Formatting variations: spacing, typos, punctuation - **recommended for questions/text**
        - **context** - Contextual additions and modifications
        - **multiple-choice** - Multiple choice question formatting
        - **multidoc** - Document order variations
        
        **Few-shot Examples:**
        - `{few_shot:3}` - 3 few-shot examples (automatically selected from data)
        - `{few_shot:[5]}` - 5 different examples per data row (list format)
        - `{few_shot:(2)}` - 2 same examples for all rows (tuple format)
        - `{few_shot:train[4]}` - 4 examples from train split, different per row
        - `{few_shot:test(3)}` - 3 examples from test split, same for all
        
        **Note:** For paraphrase variations, you'll need to provide an API key in the generation step.
        """)

    # Available columns display
    st.write("**Available columns in your data:**")
    col_display = st.columns(min(len(available_columns), 4))
    for i, col in enumerate(available_columns):
        with col_display[i % 4]:
            st.code(f"{{{col}}}")

    # Instruction input
    st.subheader("1. Define Your Instruction")
    st.write("Create an instruction template with placeholders for your data columns:")
    
    # Provide example instructions based on data
    default_instruction = ""
    if 'question' in available_columns and 'answer' in available_columns:
        if 'options' in available_columns:
            default_instruction = "Please select the correct answer: Question:{question}\\nOptions:{options}\\nAnswer:{answer}"
        else:
            default_instruction = "Please answer the following question: Question:{question}\\nAnswer:{answer}"
    elif 'text' in available_columns and 'label' in available_columns:
        default_instruction = "Classify the sentiment of the following text: \"{text}\"\\nSentiment: {label}"
    elif len(available_columns) >= 2:
        col1, col2 = available_columns[:2]
        default_instruction = f"Process the following: {{{col1}}}\\nOutput: {{{col2}}}"
    else:
        default_instruction = f"Process: {{{available_columns[0]}}}"

    instruction_text = st.text_area(
        "Instruction (with data placeholders)",
        placeholder=st.session_state.get('custom_instruction', default_instruction),
        height=100,
        help="Define your instruction with {field_name} placeholders for data columns. Use \\n for line breaks."
    )
    
    if not instruction_text:
        instruction_text = st.session_state.get('custom_instruction', default_instruction)

    # Template input
    st.subheader("2. Define Your Processing Template")
    st.write("Define how to process the instruction and individual fields:")
    
    # Default template based on instruction
    default_template = "{instruction:paraphrase}"
    if 'few_shot' in instruction_text or any(col in instruction_text for col in available_columns):
        # Add few-shot if the instruction suggests it could be useful
        if 'question' in available_columns and 'answer' in available_columns:
            default_template = "{instruction:paraphrase}\\n\\n{few_shot:(2)}"
        # Add field variations if they're referenced in instruction
        field_variations = []
        for col in available_columns:
            if f"{{{col}}}" in instruction_text:
                if col in ['question', 'text']:
                    field_variations.append(f"{{{col}:surface}}")
                elif col in ['options']:
                    field_variations.append(f"{{{col}:multiple-choice}}")
        
        if field_variations:
            default_template += "\\n\\n" + "\\n".join(field_variations)

    template_text = st.text_area(
        "Template (processing rules)",
        placeholder=st.session_state.get('custom_template', default_template),
        height=100,
        help="Define how to process your instruction and fields. Use :variation_type to specify variations."
    )
    
    if not template_text:
        template_text = st.session_state.get('custom_template', default_template)

    # Validation and preview
    if instruction_text and template_text:
        st.subheader("3. Preview")
        validate_custom_template(instruction_text, template_text, available_columns)

    # Save template button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Create Template", type="primary", use_container_width=True):
            if instruction_text.strip() and template_text.strip():
                # Store both instruction and template separately but combine for the system
                combined_template = create_combined_template(instruction_text, template_text)
                
                st.session_state.selected_template = combined_template
                st.session_state.template_name = "Custom Template"
                st.session_state.template_ready = True
                st.session_state.custom_instruction = instruction_text
                st.session_state.custom_template = template_text
                st.success("✅ Template created successfully!")
                st.rerun()
            else:
                st.error("Please enter both instruction and template")

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
    
    # Template content and analysis in columns
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📝 Template Content")
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; border: 1px solid #dee2e6;">
        """, unsafe_allow_html=True)
        
        # Show instruction and template separately if it's a custom template
        if (template_name == "Custom Template" and 
            hasattr(st.session_state, 'custom_instruction') and 
            hasattr(st.session_state, 'custom_template')):
            st.markdown("**Instruction:**")
            st.code(st.session_state.custom_instruction, language="text")
            st.markdown("**Template:**")
            st.code(st.session_state.custom_template, language="text")
        elif isinstance(template, dict) and 'instruction' in template and 'template' in template:
            # Template suggestion in new format
            st.markdown("**Instruction:**")
            st.code(template['instruction'], language="text")
            st.markdown("**Processing Template:**")
            st.code(template['template'], language="text")
        else:
            # Old format template (suggestion or legacy)
            template_str = template.get('combined', str(template)) if isinstance(template, dict) else template
            st.code(template_str, language="text")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Template analysis
    mp = MultiPromptify()
    try:
        # For combined templates, we need to parse the template part
        template_to_analyze = template
        instruction_to_analyze = None
        
        if template_name == "Custom Template" and hasattr(st.session_state, 'custom_template'):
            # Custom template - use the stored template and instruction
            template_to_analyze = st.session_state.custom_template
            instruction_to_analyze = st.session_state.custom_instruction if hasattr(st.session_state, 'custom_instruction') else None
        elif isinstance(template, dict) and 'instruction' in template and 'template' in template:
            # Template suggestion in new format
            template_to_analyze = template['template']
            instruction_to_analyze = template['instruction']
        elif isinstance(template, dict) and 'combined' in template:
            # Combined format - use as is
            template_to_analyze = template['combined']
        else:
            # Old format - use as is
            template_to_analyze = template
            
        is_valid, errors = mp.template_parser.validate_template(template_to_analyze)
        
        if is_valid:
            # Parse template to show fields
            fields = mp.template_parser.parse(template_to_analyze)
            variation_fields = mp.template_parser.get_variation_fields()
            required_columns = mp.template_parser.get_required_columns()
            
            # Also check instruction for required columns if we have one
            if instruction_to_analyze:
                instruction_fields = mp.template_parser.parse(instruction_to_analyze)
                instruction_required = {
                    field.name for field in instruction_fields 
                    if not field.is_literal and field.name not in {'instruction', 'few_shot'}
                }
                required_columns.update(instruction_required)

            with col2:
                st.markdown("""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #007bff; margin-top: 3rem;">
                    <h4 style="color: #007bff; margin-top: 0;">📋 Template Fields</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for field in fields:
                    if field.variation_type:
                        st.markdown(f"- **`{field.name}`** → {field.variation_type} variations")
                    else:
                        st.markdown(f"- **`{field.name}`** → no variations")
            
            # Check for missing columns and show error if any
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            else:
                st.success("✅ All required columns available")
            
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


def validate_custom_template(instruction_text, template_text, available_columns):
    """Validate custom template and show feedback"""
    try:
        mp = MultiPromptify()
        is_valid, errors = mp.template_parser.validate_template(template_text)
        
        if is_valid:
            # Check column availability from both instruction and template
            template_required = mp.template_parser.get_required_columns()
            
            # Also check instruction for required columns
            instruction_fields = mp.template_parser.parse(instruction_text)
            instruction_required = {
                field.name for field in instruction_fields 
                if not field.is_literal and field.name not in {'instruction', 'few_shot'}
            }
            
            all_required = template_required.union(instruction_required)
            missing_cols = all_required - set(available_columns)
            
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


def create_combined_template(instruction_text, template_text):
    """Create a combined template from instruction and template"""
    # Convert literal \n to actual newlines
    instruction_clean = instruction_text.replace('\\n', '\n')
    template_clean = template_text.replace('\\n', '\n')
    
    # Store both parts for the system to use
    # The system will process instruction as a separate component
    return {
        'instruction': instruction_clean,
        'template': template_clean,
        'combined': f"{instruction_clean}\\n\\n{template_clean}"
    } 