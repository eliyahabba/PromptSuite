"""
Step 3: Generate Variations for MultiPromptify 2.0
"""
import os
import time

import streamlit as st
from dotenv import load_dotenv

from multipromptify import MultiPromptify
from multipromptify.core.template_keys import (
    PROMPT_FORMAT, PARAPHRASE_WITH_LLM, CONTEXT_VARIATION, INSTRUCTION
)
from multipromptify.shared.constants import GenerationInterfaceConstants, GenerationDefaults
from .results_display import display_full_results

# Load environment variables
load_dotenv()

# Get API key from environment
API_KEY = os.getenv("TOGETHER_API_KEY")


def render():
    """Render the variations generation interface"""
    if not st.session_state.get('template_ready', False):
        st.error("⚠️ Please complete the template setup first (Step 2)")
        if st.button("← Go to Step 2"):
            st.session_state.page = 2
            st.rerun()
        return

    # Enhanced header with better styling
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; text-align: center;">
            ⚡ Step 3: Generate Variations
        </h1>
        <p style="color: rgba(255,255,255,0.8); text-align: center; margin: 0.5rem 0 0 0;">
            Configure settings and generate your prompt variations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get data and template
    df = st.session_state.uploaded_data
    template = st.session_state.selected_template
    template_name = st.session_state.get('template_name', 'Custom Template')

    # Display current setup
    display_current_setup(df, template, template_name)

    # Add visual separator
    st.markdown("---")

    # Generation configuration
    configure_generation()

    # Add visual separator
    st.markdown("---")

    # Generate variations
    generate_variations_interface()


def display_current_setup(df, template, template_name):
    """Display the current data and template setup with enhanced cards"""
    st.subheader("📋 Current Setup Overview")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**📊 Data Summary**")

        # Metrics in a more visual way
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("📝 Rows", len(df))
        with metric_col2:
            st.metric("🗂️ Columns", len(df.columns))

    with col2:
        st.markdown(f"**📝 Template: {template_name}**")

        if isinstance(template, dict):
            # Display instruction and prompt format separately if they exist
            if INSTRUCTION in template:
                st.markdown("**Instruction:**")
                st.code(template[INSTRUCTION], language="text")
            if PROMPT_FORMAT in template:
                st.markdown("**Prompt Format:**")
                st.code(template[PROMPT_FORMAT], language="text")
            
            # Display the rest of the template (excluding INSTRUCTION and PROMPT_FORMAT)
            template_parts = {k: v for k, v in template.items() 
                            if k not in [INSTRUCTION, PROMPT_FORMAT]}
            
            if template_parts:
                st.markdown("**Template Variables:**")
                # Format the dictionary nicely
                template_str = "{\n"
                for key, value in template_parts.items():
                    if isinstance(value, list):
                        template_str += f"    '{key}': {value},\n"
                    elif isinstance(value, dict):
                        template_str += f"    '{key}': {{\n"
                        for sub_key, sub_value in value.items():
                            template_str += f"        '{sub_key}': {sub_value},\n"
                        template_str += "    },\n"
            else:
                        template_str += f"    '{key}': {value},\n"
                template_str += "}"
                st.code(template_str, language="python")
        else:
            # Old format - just display as string
            st.code(template, language="text")


def configure_generation():
    """Configure generation settings with enhanced visual design"""
    st.subheader("⚙️ Generation Configuration")

    # Main settings in cards
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("**🔢 Quantity Settings**")

        # Max variations setting - support None for unlimited
        if 'max_variations_per_row' not in st.session_state:
            st.session_state.max_variations_per_row = GenerationDefaults.MAX_VARIATIONS_PER_ROW

        # Create options for max variations including unlimited
        max_variations_per_row_options = [
            ("Unlimited (default)", None),
            ("50 variations", 50),
            ("100 variations", 100),
            ("200 variations", 200),
            ("500 variations", 500),
            ("1000 variations", 1000)
        ]
        
        # Add custom option if current value is not in the list
        current_max_variations_per_row = st.session_state.max_variations_per_row
        if current_max_variations_per_row is not None and current_max_variations_per_row not in [50, 100, 200, 500, 1000]:
            max_variations_per_row_options.append((f"{current_max_variations_per_row} variations", current_max_variations_per_row))
        
        max_variations_per_row_labels = [label for label, _ in max_variations_per_row_options]
        max_variations_per_row_values = [value for _, value in max_variations_per_row_options]

        if current_max_variations_per_row is None:
            max_variations_per_row_index = 0
        else:
            try:
                max_variations_per_row_index = max_variations_per_row_values.index(current_max_variations_per_row)
            except ValueError:
                max_variations_per_row_index = 0

        selected_max_variations_per_row_label = st.selectbox(
            "Maximum variations per row",
            options=max_variations_per_row_labels,
            index=max_variations_per_row_index,
            key='max_variations_per_row_label',
            help="Maximum number of variations to generate per data row. If a row has more variations than this limit, the same subset will be selected for all rows (None = unlimited)"
        )
        st.session_state.max_variations_per_row = max_variations_per_row_values[max_variations_per_row_labels.index(selected_max_variations_per_row_label)]

        # Max rows setting
        df = st.session_state.uploaded_data
        # Ensure max_rows is initialized before use
        if 'max_rows' not in st.session_state:
            st.session_state.max_rows = None
        max_rows_options = [("All rows (default)", None)] + [(str(i), i) for i in range(1, len(df) + 1)]
        max_rows_labels = [label for label, _ in max_rows_options]
        max_rows_values = [value for _, value in max_rows_options]

        if st.session_state.max_rows is None:
            max_rows_index = 0
        else:
            max_rows_index = max_rows_values.index(st.session_state.max_rows)

        selected_label = st.selectbox(
            "📊 Maximum rows from data to use",
            options=max_rows_labels,
            index=max_rows_index,
            key='max_rows_label',
            help=f"Select how many rows to use (total: {len(df)} rows)."
        )
        st.session_state.max_rows = max_rows_values[max_rows_labels.index(selected_label)]

    with col2:
        st.markdown("**⚙️ Generation Settings**")

        # Variations per field
        if 'variations_per_field' not in st.session_state:
            st.session_state.variations_per_field = GenerationDefaults.VARIATIONS_PER_FIELD

        variations_per_field = st.number_input(
            "🔄 Variations per field",
            min_value=GenerationInterfaceConstants.MIN_VARIATIONS_PER_FIELD,
            max_value=GenerationInterfaceConstants.MAX_VARIATIONS_PER_FIELD,
            key='variations_per_field',
            help="Number of variations to generate for each field with variation annotations"
        )

        # Random seed for reproducibility
        st.markdown("**🎲 Reproducibility Options**")
        use_seed = st.checkbox("🔒 Use random seed for reproducible results")
        if use_seed:
            if 'random_seed' not in st.session_state:
                st.session_state.random_seed = GenerationDefaults.RANDOM_SEED
            seed = st.number_input("🌱 Random seed", min_value=0, value=st.session_state.random_seed, key='random_seed')
        else:
            st.session_state.random_seed = None

    # Check if template uses paraphrase variations
    template = st.session_state.get('selected_template', '')

    needs_api_key = False
    for k, v in template.items():
        if isinstance(v, list) and (PARAPHRASE_WITH_LLM in v or CONTEXT_VARIATION in v):
            needs_api_key = True

    if needs_api_key:
        # Enhanced API Configuration in sidebar
        with st.sidebar:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                <h3 style="color: white; margin: 0;">🔑 API Configuration</h3>
                <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Required for advanced variations</p>
            </div>
            """, unsafe_allow_html=True)

            st.info("🤖 Your template uses paraphrase variations which require an API key.")

            # Platform selection
            platform = st.selectbox(
                "🌐 Platform",
                [GenerationDefaults.API_PLATFORM, "OpenAI"],
                index=0,
                help="Choose the AI platform for paraphrase generation"
            )
            st.session_state.api_platform = platform

            # Model name with default value directly in the text box
            default_model = GenerationDefaults.MODEL_NAME
            current_model = st.session_state.get('model_name', default_model)
            model_name = st.text_input(
                "🧠 Model Name",
                value=current_model,
                help="Name of the model to use for paraphrase generation"
            )
            st.session_state.model_name = model_name

            # API Key input
            api_key = st.text_input(
                f"🔐 API Key for {platform}",
                type="password",
                value=st.session_state.get('api_key', API_KEY or ''),
                help=f"Required for generating paraphrase variations using {platform}"
            )
            # Use environment API key as default if nothing entered
            st.session_state.api_key = api_key

            if not api_key:
                st.warning("⚠️ API key is required for paraphrase variations. Generation may not work without it.")
    else:
        # Clear API key if not needed
        for key in ['api_key', 'api_platform', 'model_name']:
            if key in st.session_state:
                del st.session_state[key]

    # Remove the old few-shot configuration interface
    st.session_state.generation_few_shot = None


def generate_variations_interface():
    """Enhanced interface for generating variations"""
    st.subheader("🚀 Generate Variations")

    # Estimation in a compact info box
    df = st.session_state.uploaded_data
    max_variations_per_row = st.session_state.max_variations_per_row
    variations_per_field = st.session_state.variations_per_field
    max_rows = st.session_state.max_rows

    # Use only the selected number of rows for estimation
    effective_rows = len(df) if max_rows is None else min(max_rows, len(df))

    # Estimate total variations
    mp = MultiPromptify()
    try:
        variation_fields = mp.parse_template(st.session_state.selected_template)
        num_variation_fields = len([f for f, v in variation_fields.items() if v is not None])

        if num_variation_fields > 0:
            if max_variations_per_row is None:
                # No limit on variations
                estimated_per_row = variations_per_field ** num_variation_fields
                estimated_total = estimated_per_row * effective_rows
            else:
                # Limited variations per row
                estimated_per_row = min(variations_per_field ** num_variation_fields, max_variations_per_row)
                estimated_total = estimated_per_row * effective_rows
        else:
            estimated_total = effective_rows  # No variations, just one prompt per row

        # Compact estimation display
        st.info(
            f"📊 **Generation Estimate:** ~{estimated_total:,} variations from {effective_rows:,} rows • ~{estimated_total // effective_rows if effective_rows > 0 else 0} variations per row")

    except Exception as e:
        error_message = str(e)
        if "Not enough data for few-shot examples" in error_message:
            st.info(
                "⚠️ Not enough data for few-shot examples - please increase data size or reduce the number of examples")
        else:
            st.warning(f"❌ Could not estimate variations: {str(e)}")

    # Enhanced generation button
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("🚀 Generate All Variations", type="primary", use_container_width=True):
            generate_all_variations()

    # Show existing results if available
    if st.session_state.get('variations_generated', False):
        display_generation_results()


def generate_all_variations():
    """Generate all variations with progress tracking"""

    # Create an expandable progress container
    with st.expander("📊 Generation Progress & Details", expanded=True):
        progress_container = st.container()

        with progress_container:
            st.markdown("### 🔄 Generation in Progress...")

            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            details_text = st.empty()

            try:
                start_time = time.time()

                # Step 1: Initialize
                status_text.text("🔄 Step 1/5: Initializing MultiPromptify...")
                details_text.info("Setting up the generation engine with your configuration")
                progress_bar.progress(0.1)

                mp = MultiPromptify(max_variations_per_row=st.session_state.max_variations_per_row)

                # Set random seed if specified
                if st.session_state.get('random_seed') is not None:
                    import random
                    random.seed(st.session_state.random_seed)
                    details_text.info(f"🌱 Random seed set to: {st.session_state.random_seed}")

                # Step 2: Prepare data
                status_text.text("📊 Step 2/5: Preparing data...")
                progress_bar.progress(0.2)

                df = st.session_state.uploaded_data
                max_rows = st.session_state.max_rows

                # Limit data to selected number of rows
                if max_rows is not None and max_rows < len(df):
                    df = df.head(max_rows)
                    details_text.info(
                        f"📊 Using first {max_rows} rows out of {len(st.session_state.uploaded_data)} total rows")
                else:
                    details_text.info(f"📊 Using all {len(df)} rows from your data")

                # Step 3: Configure parameters
                status_text.text("⚙️ Step 3/5: Configuring generation parameters...")
                progress_bar.progress(0.3)

                template = st.session_state.selected_template
                variations_per_field = st.session_state.variations_per_field
                api_key = st.session_state.get('api_key')

                # Show configuration details
                config_details = []
                # Template prompt_format is already part of the template, no need for separate prompt_format
                config_details.append(f"🔄 Variations per field: {variations_per_field}")
                if api_key:
                    config_details.append("🔑 API key configured for advanced variations")

                details_text.info(" | ".join(config_details))

                # Step 4: Generate variations
                status_text.text("⚡ Step 4/5: Generating variations...")
                details_text.warning("🤖 AI is working hard to create your prompt variations...")
                progress_bar.progress(0.4)

                variations = mp.generate_variations(
                    template=template,
                    data=df,
                    variations_per_field=variations_per_field,
                    api_key=api_key
                )

                # Step 5: Computing statistics
                status_text.text("📈 Step 5/5: Computing statistics...")
                progress_bar.progress(0.8)
                details_text.info(f"✨ Generated {len(variations)} variations successfully!")

                stats = mp.get_stats(variations)

                # Complete
                progress_bar.progress(1.0)
                end_time = time.time()
                generation_time = end_time - start_time

                # Store results
                st.session_state.generated_variations = variations
                st.session_state.generation_stats = stats
                st.session_state.generation_time = generation_time
                st.session_state.variations_generated = True

                # Final success message
                status_text.text("✅ Generation Complete!")
                details_text.success(
                    f"🎉 Successfully generated {len(variations)} variations in {generation_time:.1f} seconds!")

                # Add summary statistics
                st.markdown("#### 📊 Quick Summary:")
                summary_col1, summary_col2, summary_col3 = st.columns(3)

                with summary_col1:
                    st.metric("Total Variations", len(variations))
                with summary_col2:
                    st.metric("Processing Time", f"{generation_time:.1f}s")
                with summary_col3:
                    avg_per_row = len(variations) / len(df) if len(df) > 0 else 0
                    st.metric("Avg per Row", f"{avg_per_row:.1f}")

                # Auto-scroll to results after a short delay
                time.sleep(1)
                st.rerun()

            except Exception as e:
                # Check if this is the few-shot examples error
                error_message = str(e)
                if "Not enough data for few-shot examples" in error_message:
                    # Handle few-shot error gracefully with single clear message
                    status_text.text("⚠️ Data Configuration Issue")
                    details_text.error("Cannot proceed - insufficient data for few-shot examples")
                    st.error(
                        "⚠️ **Cannot create few-shot examples:** Not enough data rows available. Please increase your data size or reduce the number of few-shot examples in the template configuration.")
                    return  # Stop execution for few-shot error
                else:
                    # Error handling with details
                    status_text.text("❌ Generation Failed")
                    details_text.error(f"❌ Error: {str(e)}")
                    st.error(f"❌ Error generating variations: {str(e)}")

                    # Show debug info outside the expander to avoid nesting
                    import traceback
                    st.text("🔍 Debug Information:")
                    st.code(traceback.format_exc())


def display_generation_results():
    """Display the full results using the shared display module"""
    if not st.session_state.get('variations_generated', False):
        return

    variations = st.session_state.generated_variations
    stats = st.session_state.generation_stats
    generation_time = st.session_state.generation_time
    original_data = st.session_state.uploaded_data

    # Use the shared display function with collapsible option
    with st.container():
        # Add collapsible container for the results
        display_full_results(
            variations=variations,
            original_data=original_data,
            stats=stats,
            generation_time=generation_time,
            show_export=True,
            show_header=True
        )

    # Generation complete - no more navigation needed
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                padding: 1.5rem; border-radius: 10px; text-align: center; color: white; margin: 2rem 0;">
        <h3 style="margin: 0;">🎉 Generation Complete!</h3>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Your prompt variations are ready above. You can download them using the export options.
        </p>
    </div>
    """, unsafe_allow_html=True)
