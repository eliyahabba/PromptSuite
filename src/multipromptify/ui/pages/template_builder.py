"""
Template Builder for MultiPromptify - Dictionary Format Only
"""

import streamlit as st

from multipromptify.core import MultiPromptify
from multipromptify.core.template_keys import (
    PROMPT_FORMAT, PROMPT_FORMAT_VARIATIONS, GOLD_KEY, FEW_SHOT_KEY, PARAPHRASE_WITH_LLM, CONTEXT_VARIATION,
    SHUFFLE_VARIATION, MULTIDOC_VARIATION, ENUMERATE_VARIATION,
    INSTRUCTION, INSTRUCTION_VARIATIONS, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION
)
from multipromptify.shared.constants import FEW_SHOT_DYNAMIC_DEFAULT


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
        <strong>Dictionary Templates</strong> allow precise control over how your prompts are structured and varied.
        Define which fields to vary and how many variations to generate for each field.
    </div>
    """, unsafe_allow_html=True)

    # Get the uploaded data
    df = st.session_state.uploaded_data
    available_columns = df.columns.tolist()

    # Template interface - two tabs: suggestions and custom builder
    tab1, tab2 = st.tabs(["🎯 Template Suggestions", "🔧 Custom Builder"])

    with tab1:
        template_suggestions_interface(available_columns)

    with tab2:
        template_builder_interface(available_columns)

    # Show selected template details at the bottom
    if st.session_state.get('template_ready', False):
        display_selected_template_details(available_columns)


def template_suggestions_interface(available_columns):
    """Interface for selecting template suggestions"""
    st.subheader("Choose a Template Suggestion")
    st.write("Select a pre-built template that matches your data structure and task type")

    # Show currently selected template at the top
    if st.session_state.get('template_ready', False):
        selected_name = st.session_state.get('template_name', 'Unknown')
        selected_template = st.session_state.get('selected_template', {})

        # Show template preview
        field_count = len([k for k in selected_template.keys() if k != 'few_shot'])
        few_shot_info = ""
        if 'few_shot' in selected_template:
            fs_config = selected_template['few_shot']
            few_shot_info = f" + {fs_config.get('count', 2)} few-shot examples"
        template_preview = f"Dictionary format: {field_count} fields{few_shot_info}"

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
                # Get required fields from new dictionary format
                template_dict = template['template']
                required_fields = [
                    k for k in template_dict.keys()
                    if k not in [
                        INSTRUCTION,
                        INSTRUCTION_VARIATIONS, PROMPT_FORMAT, PROMPT_FORMAT_VARIATIONS,
                        FEW_SHOT_KEY, GOLD_KEY, ENUMERATE_VARIATION
                    ]
                ]

                # Check if gold field value exists in columns
                if GOLD_KEY in template_dict:
                    gold_config = template_dict[GOLD_KEY]
                    if isinstance(gold_config, str):
                        # Old format: gold field is just the column name
                        if gold_config not in available_columns:
                            required_fields.append(gold_config)
                    elif isinstance(gold_config, dict) and 'field' in gold_config:
                        # New format: gold field is a dict with 'field' key
                        gold_field = gold_config['field']
                        if gold_field not in available_columns:
                            required_fields.append(gold_field)
                        # If there's an options_field specified, check it too
                        if 'options_field' in gold_config:
                            options_field = gold_config['options_field']
                            if options_field not in available_columns:
                                required_fields.append(options_field)

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
                    current_selected = st.session_state.get('selected_template', {})
                    is_selected = (template['template'] == current_selected)

                    # Style the expander differently if selected
                    if is_selected:
                        expander_label = f"✅ {template['name']} (Currently Selected)"
                    else:
                        expander_label = f"📋 {template['name']}"

                    with st.expander(expander_label, expanded=is_selected):
                        st.write(f"**Description:** {template['description']}")

                        # Display dictionary template
                        st.markdown("**Template Configuration (Dictionary Format):**")

                        # Display as formatted JSON code block to avoid Streamlit's array indexing
                        import json
                        formatted_json = json.dumps(template['template'], indent=2, ensure_ascii=False)
                        st.code(formatted_json, language="json")

                        # Button styling based on selection
                        button_key = f"template_{category_key}_{template['name'].lower().replace(' ', '_')}"
                        if is_selected:
                            if st.button(f"🔄 Re-select {template['name']}", key=f"re_{button_key}"):
                                st.session_state.selected_template = template['template']
                                st.session_state.template_name = template['name']
                                st.session_state.template_ready = True
                                st.rerun()
                        else:
                            if st.button(f"✅ Select {template['name']}", key=button_key, type="primary"):
                                st.session_state.selected_template = template['template']
                                st.session_state.template_name = template['name']
                                st.session_state.template_ready = True
                                st.success(f"Template '{template['name']}' selected!")
                                st.rerun()

            if incompatible_templates:
                st.warning(f"⚠️ {len(incompatible_templates)} templates require additional columns")

                with st.expander("Show incompatible templates"):
                    for template in incompatible_templates:
                        missing = ', '.join(template['missing_fields'])
                        st.write(f"**{template['name']}**: Missing columns: {missing}")


def template_builder_interface(available_columns):
    """Main template builder interface using dictionary format"""
    st.subheader("🔧 Custom Template Builder")

    # Show current template status
    if st.session_state.get('template_ready', False):
        template_name = st.session_state.get('template_name', 'Custom Template')
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); 
                    padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="color: white; margin: 0;">✅ Active Template: {template_name}</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
                Template is ready for generating variations
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Available variation types
    variation_types = [PARAPHRASE_WITH_LLM, CONTEXT_VARIATION, SHUFFLE_VARIATION, MULTIDOC_VARIATION,
                       FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION]

    # Add explanation of new variation types
    with st.expander("ℹ️ Variation Types Guide", expanded=False):
        st.markdown("""
        **Available Variation Types:**
        
        - **paraphrase_with_llm**: Uses AI to rephrase text while preserving meaning
        - **rewording**: Backward compatibility - maps to typos_and noise
        - **context**: Adds background context to questions
        - **shuffle**: Reorders items in lists (e.g., multiple choice options)
        - **multidoc**: Handles multiple document variations
        - **format structure**: Semantic-preserving format changes (separators, connectors, casing)
        - **typos and noise**: Robustness testing with noise injection (typos, character swaps, etc.)
        
        **When to use which:**
        - Use **format structure** for testing model robustness to different prompt formats
        - Use **typos and noise** for testing model robustness to noisy input
        - Use **rewording** for backward compatibility (same as typos_and noise)
        """)

    # Initialize template state
    if 'template_config' not in st.session_state:
        st.session_state.template_config = {}

    st.markdown("### 1. Configure Fields")
    st.write("Select which fields to include and which variations to apply:")

    # Available columns display
    st.markdown("**Available data columns:**")
    filtered_columns = [col for col in available_columns if col not in [INSTRUCTION, INSTRUCTION_VARIATIONS]]
    cols = st.columns(min(len(filtered_columns), 4))
    for i, col in enumerate(filtered_columns):
        with cols[i % 4]:
            st.code(col, language="text")

    # Field configuration interface
    configured_fields = {}

    # Use tabs for better organization
    field_tabs = st.tabs(
        ["📝 Instruction", "📊 Data Fields", "🏆 Gold Field", "🔢 Enumerate", "🎯 Few-shot", "⚙️ System Prompt"])

    with field_tabs[0]:
        # Instruction configuration
        st.markdown("**Instruction Field Configuration**")
        st.write("The prompt_format field defines the overall prompt structure.")

        # Instruction template input
        st.markdown("**1. Instruction Template (Required)**")
        prompt_format_template = st.text_area(
            "Enter your prompt_format template with placeholders:",
            value=st.session_state.template_config.get(PROMPT_FORMAT, ''),
            key=PROMPT_FORMAT,
            help="Use {field_name} for placeholders. Example: 'Answer the following question: {question}\nAnswer: {answer}'. Remember to specify the 'gold' field for the answer column.",
            placeholder="Answer the following question: {question}\nAnswer: {answer}"
        )

        if prompt_format_template:
            # Show preview of placeholders
            import re
            placeholders = re.findall(r'\{([^}]+)\}', prompt_format_template)
            if placeholders:
                st.info(f"📋 Found placeholders: {', '.join(set(placeholders))}")
            else:
                st.warning("⚠️ No placeholders found in template")

        if prompt_format_template:
            configured_fields[PROMPT_FORMAT] = prompt_format_template

        st.markdown("**2. Instruction Variations (Optional)**")
        selected_variations = st.multiselect(
            "Variation types for prompt_format",
            options=variation_types,
            default=st.session_state.template_config.get(PROMPT_FORMAT_VARIATIONS, []),
            key="prompt_format_variations",
            help="Select variation types to apply to the prompt_format"
        )

        if selected_variations:
            configured_fields[PROMPT_FORMAT_VARIATIONS] = selected_variations

    with field_tabs[1]:
        # Data fields configuration
        st.markdown("**Data Fields Configuration**")
        st.write("Configure variations for your data columns:")

        for field_name in filtered_columns:
            with st.expander(f"Configure '{field_name}' field"):
                selected_variations = st.multiselect(
                    f"Variations for {field_name}",
                    options=variation_types,
                    default=st.session_state.template_config.get(field_name, []),
                    key=f"variations_{field_name}",
                    help=f"Select variation types for {field_name}"
                )

                if selected_variations:
                    configured_fields[field_name] = selected_variations

                # Show sample data for this field
                df = st.session_state.uploaded_data
                if not df[field_name].dropna().empty:
                    sample_value = str(df[field_name].dropna().iloc[0])
                    st.code(f"Sample: {sample_value[:100]}{'...' if len(sample_value) > 100 else ''}")

    with field_tabs[2]:
        # Gold field configuration
        st.markdown("**Gold Field Configuration**")
        st.write("Configure the gold field (correct answer/output column):")

        # Check if few-shot will be enabled to determine if gold is required
        # We need to check the current state of few-shot configuration from the UI
        st.info("💡 Gold field is required when using few-shot examples")

        # Gold field selection
        gold_field = st.selectbox(
            "Select gold field (correct answer column):",
            options=["None"] + available_columns,
            index=0,
            key="gold_field",
            help="Choose the column that contains the correct answers/outputs"
        )

        if gold_field and gold_field != "None":
            # Gold field configuration format
            gold_format = st.selectbox(
                "Gold field format:",
                options=["Simple", "Advanced"],
                index=0,
                key="gold_format",
                help="Simple: just specify the column name. Advanced: specify type and options"
            )

            if gold_format == "Simple":
                # Simple format - just the field name
                configured_fields[GOLD_KEY] = gold_field

                # Show preview
                df = st.session_state.uploaded_data
                if not df[gold_field].dropna().empty:
                    sample_value = df[gold_field].dropna().iloc[0]
                    st.markdown("**Preview:**")
                    st.code(f"Sample gold value: {sample_value}")
                    st.info("💡 Using simple format: `'gold': '{}'`".format(gold_field))

            else:
                # Advanced format - full configuration
                gold_type = st.selectbox(
                    "Gold field type:",
                    options=["value", "index"],
                    index=0,
                    key="gold_type",
                    help="'value' for text answers, 'index' for position-based answers (like multiple choice)"
                )

                # If index type, need options field
                options_field = None
                if gold_type == "index":
                    options_field = st.selectbox(
                        "Options field (for index-based answers):",
                        options=["None"] + available_columns,
                        index=0,
                        key="gold_options_field",
                        help="Choose the column that contains the list of options"
                    )

                    if options_field == "None":
                        options_field = None

                # Preview gold configuration
                df = st.session_state.uploaded_data
                if not df[gold_field].dropna().empty:
                    sample_value = df[gold_field].dropna().iloc[0]
                    st.markdown("**Preview:**")
                    st.code(f"Sample gold value: {sample_value}")

                    if gold_type == "index" and options_field and options_field in df.columns:
                        if not df[options_field].dropna().empty:
                            sample_options = df[options_field].dropna().iloc[0]
                            st.code(f"Sample options: {sample_options}")

                # Add to configuration
                if gold_type == "value":
                    # Simple format for value type
                    if options_field:
                        configured_fields[GOLD_KEY] = {
                            'field': gold_field,
                            'type': gold_type,
                            'options_field': options_field
                        }
                    else:
                        configured_fields[GOLD_KEY] = {
                            'field': gold_field,
                            'type': gold_type
                        }
                else:
                    # Index type requires options field
                    if options_field:
                        configured_fields[GOLD_KEY] = {
                            'field': gold_field,
                            'type': gold_type,
                            'options_field': options_field
                        }
                    else:
                        st.warning("⚠️ Index type requires an options field")

    with field_tabs[3]:
        # Enumerate configuration
        st.markdown("**Enumerate Field Configuration**")
        st.write("Configure automatic enumeration for list fields (e.g., multiple choice options):")

        # Check for fields that contain lists
        df = st.session_state.uploaded_data
        list_like_fields = []
        for field_name in available_columns:
            if not df[field_name].dropna().empty:
                sample_value = df[field_name].dropna().iloc[0]
                # Check if it's actually a list type
                if isinstance(sample_value, list):
                    list_like_fields.append(field_name)

        if list_like_fields:
            st.info(f"💡 Detected list-like fields: {', '.join(list_like_fields)}")

        # Field selection for enumeration
        enumerate_field = st.selectbox(
            "Select field to enumerate:",
            options=["None"] + available_columns,
            index=0,
            key="enumerate_field",
            help="Choose a field that contains lists or comma-separated values to enumerate"
        )

        if enumerate_field and enumerate_field != "None":
            # Enumeration type selection
            enumerate_types = {
                "1234": "Numbers (1. 2. 3. 4.)",
                "ABCD": "Uppercase letters (A. B. C. D.)",
                "abcd": "Lowercase letters (a. b. c. d.)",
                "greek": "Greek letters (α. β. γ. δ.)",
                "roman": "Roman numerals (I. II. III. IV.)"
            }

            enumerate_type = st.selectbox(
                "Select enumeration type:",
                options=list(enumerate_types.keys()),
                format_func=lambda x: enumerate_types[x],
                index=0,
                key="enumerate_type",
                help="Choose how to enumerate the items in the selected field"
            )

            # Preview enumeration
            df = st.session_state.uploaded_data
            if not df[enumerate_field].dropna().empty:
                sample_value = str(df[enumerate_field].dropna().iloc[0])
                st.markdown("**Preview:**")

                # Show original
                st.code(f"Original: {sample_value}")

                # Show enumerated preview (simulate the enumeration)
                try:
                    from multipromptify.augmentations.structure.enumerate import EnumeratorAugmenter
                    enumerator = EnumeratorAugmenter()
                    preview_result = enumerator.enumerate_field(sample_value, enumerate_type)
                    st.code(f"Enumerated: {preview_result}")
                except Exception as e:
                    st.warning(f"Preview error: {e}")

            # Add to configuration
            configured_fields['enumerate'] = {
                'field': enumerate_field,
                'type': enumerate_type
            }

    with field_tabs[4]:
        # Few-shot configuration
        st.markdown("**Few-shot Examples Configuration**")
        st.write("Configure few-shot learning examples:")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Calculate dynamic default based on available data
            df = st.session_state.uploaded_data
            available_rows = len(df) if df is not None else 0
            # Set default to min(2, max(0, available_rows - 1)) to ensure we have at least 1 row left for generation
            dynamic_default = FEW_SHOT_DYNAMIC_DEFAULT(available_rows)

            few_shot_count = st.number_input(
                "Number of examples",
                min_value=0,
                max_value=min(10, available_rows - 1) if available_rows > 1 else 0,
                value=st.session_state.template_config.get(FEW_SHOT_KEY, {}).get('count', dynamic_default),
                key="few_shot_count",
                help=f"Set to 0 to disable few-shot examples. Max available: {available_rows - 1 if available_rows > 1 else 0} (keeping 1 row for generation)"
            )

        with col2:
            few_shot_format = st.selectbox(
                "Example selection",
                options=["rotating", "fixed"],
                index=0 if st.session_state.template_config.get(FEW_SHOT_KEY, {}).get('format',
                                                                                      'rotating') == 'rotating' else 1,
                key="few_shot_format",
                help="Rotating: different examples per row, Fixed: same examples for all rows"
            )

        with col3:
            few_shot_split = st.selectbox(
                "Data split",
                options=["all", "train", "test"],
                index=0,
                key="few_shot_split",
                help="Which portion of data to use for examples"
            )

        if few_shot_count > 0:
            configured_fields[FEW_SHOT_KEY] = {
                'count': few_shot_count,
                'format': few_shot_format,
                'split': few_shot_split
            }

    # System prompt configuration (new tab)
    with field_tabs[5]:
        st.markdown("**System Prompt Configuration**")
        st.write("Configure the system prompt for your template (optional):")

        instruction = st.text_area(
            "System prompt template (optional):",
            value=st.session_state.template_config.get(INSTRUCTION, ''),
            key=INSTRUCTION,
            help="System prompt for the model, e.g. 'You are a helpful assistant.'"
        )
        if instruction:
            configured_fields[INSTRUCTION] = instruction

        instruction_variations = st.multiselect(
            "Variation types for system prompt (optional):",
            options=variation_types,
            default=st.session_state.template_config.get(INSTRUCTION_VARIATIONS, []),
            key="instruction_variations",
            help="Select variation types to apply to the system prompt"
        )
        if instruction_variations:
            configured_fields[INSTRUCTION_VARIATIONS] = instruction_variations

    # Template preview and validation
    st.markdown("### 2. Template Preview")

    if configured_fields:
        # Validate template
        mp = MultiPromptify()
        try:
            parsed_fields = mp.parse_template(configured_fields)

            # Show template structure
            st.success(f"✅ Template is valid! Configured {len(parsed_fields)} fields.")

            # Display configuration in a nice format
            st.markdown("**Template Configuration:**")

            # Display as formatted JSON code block to avoid Streamlit's array indexing
            import json
            formatted_json = json.dumps(configured_fields, indent=2, ensure_ascii=False)
            st.code(formatted_json, language="json")

            # Show field summary
            field_summary = []
            for field_name, config in configured_fields.items():
                if field_name == FEW_SHOT_KEY:
                    summary = f"**{field_name}**: {config['count']} {config['format']} examples from {config['split']} data"
                elif field_name == GOLD_KEY:
                    if isinstance(config, str):
                        summary = f"**{field_name}**: {config} (simple format)"
                    elif isinstance(config, dict):
                        gold_type = config.get('type', 'value')
                        if 'options_field' in config:
                            summary = f"**{field_name}**: {config['field']} ({gold_type} type, options from {config['options_field']})"
                        else:
                            summary = f"**{field_name}**: {config['field']} ({gold_type} type)"
                    else:
                        summary = f"**{field_name}**: {config}"
                elif field_name == 'enumerate':
                    summary = f"**{field_name}**: {config['field']} field with {config['type']} enumeration"
                elif field_name == INSTRUCTION:
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0; padding: 0.75rem; background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%); 
                                border-radius: 6px; border-left: 3px solid #1976d2;">
                        <strong style="color: #1976d2;">⚙️ {field_name}:</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    quoted_value = f'"{config}"'
                    st.code(quoted_value, language="text")
                else:
                    variations = ', '.join(config) if isinstance(config, list) else str(config)
                    summary = f"**{field_name}**: {variations}"
                field_summary.append(summary)

            for summary in field_summary:
                st.markdown(f"- {summary}")

        except Exception as e:
            st.error(f"❌ Template validation error: {str(e)}")
            configured_fields = {}
    else:
        st.warning("⚠️ Configure at least one field to preview the template")

    # Save template
    st.markdown("### 3. Save Template")

    # Check if template is valid
    template_errors = []

    # Check if few-shot is configured and gold is required
    if FEW_SHOT_KEY in configured_fields:
        few_shot_count = configured_fields[FEW_SHOT_KEY].get('count', 0)
        if few_shot_count > 0 and GOLD_KEY not in configured_fields:
            template_errors.append("Gold field is required when using few-shot examples")

    # Check if prompt_format template is provided
    if PROMPT_FORMAT not in configured_fields:
        template_errors.append("Instruction template is required")

    # Display validation errors
    if template_errors:
        st.error("❌ Template validation errors:")
        for error in template_errors:
            st.write(f"• {error}")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Create Template", type="primary", use_container_width=True):
            if configured_fields and not template_errors:
                st.session_state.selected_template = configured_fields
                st.session_state.template_name = "Custom Dictionary Template"
                st.session_state.template_ready = True
                st.session_state.template_config = configured_fields
                st.success("✅ Template created successfully!")
                st.rerun()
            elif template_errors:
                st.error("❌ Please fix validation errors before creating template")
            else:
                st.error("❌ Please configure at least one field")


def display_selected_template_details(available_columns):
    """Display selected template details"""
    st.markdown("---")

    template = st.session_state.selected_template
    template_name = st.session_state.get('template_name', 'Dictionary Template')

    # Main template display
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h2 style="color: white; margin: 0; text-align: center;">🎯 Selected Template: {template_name}</h2>
        <p style="color: rgba(255,255,255,0.9); text-align: center; margin: 0.5rem 0 0 0;">
            Ready for generating variations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Template configuration with enhanced styling
    st.markdown("### 📝 Template Configuration")

    # Display template in a clean, expanded format with colors
    for key, value in template.items():
        if key == PROMPT_FORMAT:
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.75rem; background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%); 
                        border-radius: 6px; border-left: 3px solid #2196f3;">
                <strong style="color: #1976d2;">📝 {key}:</strong>
            </div>
            """, unsafe_allow_html=True)
            # Add quotes around the prompt_format template for clarity
            quoted_value = f'"{value}"'
            st.code(quoted_value, language="text")

        elif key == FEW_SHOT_KEY and isinstance(value, dict):
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.75rem; background: linear-gradient(135deg, #f3e5f5 0%, #f8f9fa 100%); 
                        border-radius: 6px; border-left: 3px solid #9c27b0;">
                <strong style="color: #7b1fa2;">🎯 {key}:</strong> 
                <span style="color: #4a148c;">{value['count']} {value['format']} examples from {value['split']} data</span>
            </div>
            """, unsafe_allow_html=True)

        elif key == GOLD_KEY:
            if isinstance(value, str):
                gold_text = f"{value} (simple format)"
            elif isinstance(value, dict):
                gold_type = value.get('type', 'value')
                if 'options_field' in value:
                    gold_text = f"{value['field']} ({gold_type} type, options from {value['options_field']})"
                else:
                    gold_text = f"{value['field']} ({gold_type} type)"
            else:
                gold_text = str(value)

            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.75rem; background: linear-gradient(135deg, #fff8e1 0%, #f8f9fa 100%); 
                        border-radius: 6px; border-left: 3px solid #ffc107;">
                <strong style="color: #f57c00;">🏆 {key}:</strong> 
                <span style="color: #e65100;">{gold_text}</span>
            </div>
            """, unsafe_allow_html=True)

        elif key == 'enumerate':
            enumerate_text = f"{value['field']} field with {value['type']} enumeration"
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.75rem; background: linear-gradient(135deg, #f1f8e9 0%, #f8f9fa 100%); 
                        border-radius: 6px; border-left: 3px solid #8bc34a;">
                <strong style="color: #689f38;">🔢 {key}:</strong> 
                <span style="color: #33691e;">{enumerate_text}</span>
            </div>
            """, unsafe_allow_html=True)

        elif key == INSTRUCTION:
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.75rem; background: linear-gradient(135deg, #e3f2fd 0%, #f8f9fa 100%); 
                        border-radius: 6px; border-left: 3px solid #1976d2;">
                <strong style="color: #1976d2;">⚙️ {key}:</strong>
            </div>
            """, unsafe_allow_html=True)
            quoted_value = f'"{value}"'
            st.code(quoted_value, language="text")
        elif key == INSTRUCTION_VARIATIONS:
            variations_text = ', '.join(value)
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.75rem; background: linear-gradient(135deg, #e8f5e8 0%, #f8f9fa 100%); 
                        border-radius: 6px; border-left: 3px solid #4caf50;">
                <strong style="color: #2e7d32;">🔄 {key}:</strong> 
                <span style="color: #1b5e20;">{variations_text}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin: 0.5rem 0; padding: 0.75rem; background: linear-gradient(135deg, #fff3e0 0%, #f8f9fa 100%); 
                        border-radius: 6px; border-left: 3px solid #ff9800;">
                <strong style="color: #f57c00;">⚙️ {key}:</strong> 
                <span style="color: #e65100;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    # Continue button
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
