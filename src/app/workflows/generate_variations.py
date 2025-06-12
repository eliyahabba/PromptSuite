"""
Step 3: Generate Variations for MultiPromptify 2.0
"""
import pandas as pd
import streamlit as st
import json
import time
import os
from dotenv import load_dotenv
from src.multipromptify import MultiPromptify
from src.shared.constants import DEFAULT_MODEL
from src.app.components.results_display import display_full_results

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
        
        st.markdown("**📋 Available Columns:**")
        columns_text = ", ".join(df.columns.tolist())
        st.markdown(f'<div style="background: #e9ecef; padding: 0.5rem; border-radius: 5px; font-family: monospace;">{columns_text}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**📝 Template: {template_name}**")
        
        # Handle new template format (dictionary) vs old format (string)
        if isinstance(template, dict):
            if 'instruction' in template and 'template' in template:
                st.markdown("**Instruction:**")
                st.code(template['instruction'], language="text")
                st.markdown("**Processing Template:**")
                st.code(template['template'], language="text")
            else:
                # Fallback to combined or string representation
                template_str = template.get('combined', str(template))
                st.code(template_str, language="text")
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
        
        # Max variations setting
        max_variations = st.number_input(
            "🎯 Maximum variations to generate",
            min_value=1,
            max_value=1000,
            value=st.session_state.get('max_variations', 50),
            help="Total number of prompt variations to generate across all data rows"
        )
        st.session_state.max_variations = max_variations
        
        # Max rows setting
        df = st.session_state.uploaded_data
        max_rows = st.number_input(
            "📊 Maximum rows from data to use",
            min_value=1,
            max_value=len(df),
            value=st.session_state.get('max_rows', len(df)),
            help=f"Use only the first N rows from your data (total: {len(df)} rows)"
        )
        st.session_state.max_rows = max_rows
    
    with col2:
        st.markdown("**⚙️ Generation Settings**")
        
        # Variations per field
        variations_per_field = st.number_input(
            "🔄 Variations per field",
            min_value=1,
            max_value=10,
            value=st.session_state.get('variations_per_field', 3),
            help="Number of variations to generate for each field with variation annotations"
        )
        st.session_state.variations_per_field = variations_per_field
        
        # Random seed for reproducibility
        st.markdown("**🎲 Reproducibility Options**")
        use_seed = st.checkbox("🔒 Use random seed for reproducible results")
        if use_seed:
            seed = st.number_input("🌱 Random seed", min_value=0, value=42)
            if seed is None:
                seed = 42
            st.session_state.random_seed = seed
        else:
            st.session_state.random_seed = None
    
    # Check if template uses paraphrase variations
    template = st.session_state.get('selected_template', '')
    
    # Handle new template format (dictionary) vs old format (string)
    template_text = ''
    if isinstance(template, dict):
        if 'template' in template:
            template_text = template['template']
        elif 'combined' in template:
            template_text = template['combined']
        else:
            template_text = str(template)
    else:
        template_text = template
    
    needs_api_key = ':paraphrase' in template_text
    
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
                ["TogetherAI", "OpenAI"], 
                index=0,
                help="Choose the AI platform for paraphrase generation"
            )
            st.session_state.api_platform = platform
            
            # Model name with default value directly in the text box
            default_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
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
    max_variations = st.session_state.get('max_variations', 50)
    variations_per_field = st.session_state.get('variations_per_field', 3)
    max_rows = st.session_state.get('max_rows', len(df))
    
    # Use only the selected number of rows for estimation
    effective_rows = min(max_rows, len(df))
    
    # Estimate total variations
    mp = MultiPromptify()
    try:
        variation_fields = mp.parse_template(st.session_state.selected_template)
        num_variation_fields = len([f for f, v in variation_fields.items() if v is not None])
        
        if num_variation_fields > 0:
            estimated_per_row = min(variations_per_field ** num_variation_fields, max_variations // effective_rows)
            estimated_total = min(estimated_per_row * effective_rows, max_variations)
        else:
            estimated_total = effective_rows  # No variations, just one prompt per row
        
        # Compact estimation display
        st.info(f"📊 **Generation Estimate:** ~{estimated_total:,} variations from {effective_rows:,} rows • ~{estimated_total//effective_rows if effective_rows > 0 else 0} variations per row")
    
    except Exception as e:
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
                
                mp = MultiPromptify(max_variations=st.session_state.max_variations)
                
                # Set random seed if specified
                if st.session_state.get('random_seed') is not None:
                    import random
                    random.seed(st.session_state.random_seed)
                    details_text.info(f"🌱 Random seed set to: {st.session_state.random_seed}")
                
                # Step 2: Prepare data
                status_text.text("📊 Step 2/5: Preparing data...")
                progress_bar.progress(0.2)
                
                df = st.session_state.uploaded_data
                max_rows = st.session_state.get('max_rows', len(df))
                
                # Limit data to selected number of rows
                if max_rows < len(df):
                    df = df.head(max_rows)
                    details_text.info(f"📊 Using first {max_rows} rows out of {len(st.session_state.uploaded_data)} total rows")
                else:
                    details_text.info(f"📊 Using all {len(df)} rows from your data")
                
                # Step 3: Configure parameters
                status_text.text("⚙️ Step 3/5: Configuring generation parameters...")
                progress_bar.progress(0.3)
                
                template = st.session_state.selected_template
                variations_per_field = st.session_state.get('variations_per_field', 3)
                api_key = st.session_state.get('api_key')
                
                # Show configuration details
                config_details = []
                # Template instruction is already part of the template, no need for separate instruction
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
                details_text.success(f"🎉 Successfully generated {len(variations)} variations in {generation_time:.1f} seconds!")
                
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
                # Error handling with details
                status_text.text("❌ Generation Failed")
                details_text.error(f"❌ Error: {str(e)}")
                
                st.error(f"❌ Error generating variations: {str(e)}")
                import traceback
                with st.expander("🔍 Debug Information"):
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