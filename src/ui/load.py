import streamlit as st
import sys
import os

# Add the src directory to the path to import multipromptify
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ui import (
    upload_data,
    template_builder,
    generate_variations,
    show_results
)
from src.ui.utils.debug_helpers import (
    initialize_debug_mode,
    load_demo_data_for_step
)
from src.ui.utils.progress_indicator import show_progress_indicator


def main():
    """Main Streamlit app for MultiPromptify 2.0"""
    # Set up page configuration
    st.set_page_config(
        layout="wide", 
        page_title="MultiPromptify 2.0 - Multi-Prompt Dataset Generator",
        page_icon="🚀"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stProgress .st-bo {
        background-color: #f0f2f6;
    }
    .step-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🚀 MultiPromptify 2.0</h1>
        <h3>Generate Multi-Prompt Datasets from Single-Prompt Datasets</h3>
        <p style="color: #666;">Create variations of your prompts using template-based transformations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Retrieve query parameters
    params = st.query_params
    start_step = int(params.get("step", ["1"])[0])
    debug_mode = params.get("debug", ["False"])[0].lower() == "true"
    
    # Initialize session state
    initialize_session_state(start_step, debug_mode)
    
    # Initialize debug mode UI if needed
    if st.session_state.debug_mode:
        initialize_debug_mode()
    
    # Total number of pages in the simplified application
    total_pages = 4
    
    # Display the progress indicator
    current_page = st.session_state.page
    show_progress_indicator(current_page, total_pages)
    
    # Render the appropriate page
    render_current_page(current_page)


def initialize_session_state(start_step=1, debug_mode=False):
    """Initialize the session state for navigation"""
    defaults = {
        'page': start_step,
        'debug_mode': debug_mode,
        'data_loaded': False,
        'template_ready': False,
        'variations_generated': False,
        'template_suggestions': [
            # Sentiment Analysis
            {
                'name': 'Sentiment Analysis',
                'template': '{instruction:paraphrase}: "{text:surface}"\nSentiment: {label}',
                'description': 'Classify sentiment with paraphrase instructions and surface text variations (spacing, typos)',
                'sample_data': {
                    'text': ['I love this movie!', 'This book is terrible.'],
                    'label': ['positive', 'negative']
                }
            },
            # Question Answering
            {
                'name': 'Question Answering',
                'template': '{instruction:paraphrase}: {question:surface}\nAnswer: {answer}',
                'description': 'Q&A with paraphrase instructions and surface question variations (spacing, typos)',
                'sample_data': {
                    'question': ['What is the capital of France?', 'How many days in a week?'],
                    'answer': ['Paris', '7']
                }
            },
            # Multiple Choice
            {
                'name': 'Multiple Choice',
                'template': '{instruction:paraphrase}:\n\nQuestion: {question:surface}\nOptions: {options:multiple-choice}\n\nAnswer: {answer}',
                'description': 'Multiple choice with paraphrase instructions, surface question variations, and choice formatting',
                'sample_data': {
                    'question': ['What is the largest planet?', 'Which element has symbol O?'],
                    'options': ['A) Earth B) Jupiter C) Mars', 'A) Oxygen B) Gold C) Silver'],
                    'answer': ['B', 'A']
                }
            },
            # Basic few-shot format
            {
                'name': 'Few-shot Basic (Same Examples)',
                'template': '{instruction:paraphrase}\n\n{few_shot:3}\n\nQuestion: {question:surface}\nAnswer: {answer}',
                'description': 'Basic few-shot learning with 3 examples (same examples for all rows)',
                'sample_data': {
                    'question': ['What is 12+8?', 'What is 15-7?', 'What is 6*4?', 'What is 20/5?'],
                    'answer': ['20', '8', '24', '4']
                }
            },
            # List format few-shot (different per row)
            {
                'name': 'Few-shot List (Different Examples Per Row)',
                'template': '{instruction:paraphrase}\n\n{few_shot:[2]}\n\nQuestion: {question:surface}\nAnswer: {answer}',
                'description': 'Few-shot with 2 different examples per data row (list format)',
                'sample_data': {
                    'question': ['What is 12+8?', 'What is 15-7?', 'What is 6*4?', 'What is 20/5?'],
                    'answer': ['20', '8', '24', '4']
                }
            },
            # Tuple format few-shot (same for all)
            {
                'name': 'Few-shot Tuple (Same Examples For All)',
                'template': '{instruction:paraphrase}\n\n{few_shot:(3)}\n\nQuestion: {question:surface}\nAnswer: {answer}',
                'description': 'Few-shot with 3 same examples for all rows (tuple format)',
                'sample_data': {
                    'question': ['What is 12+8?', 'What is 15-7?', 'What is 6*4?', 'What is 20/5?'],
                    'answer': ['20', '8', '24', '4']
                }
            },
            # Train split few-shot
            {
                'name': 'Few-shot with Train Split',
                'template': '{instruction:paraphrase}\n\n{few_shot:train[4]}\n\nQuestion: {question:surface}\nAnswer: {answer}',
                'description': 'Few-shot with 4 examples from train split, different per row',
                'sample_data': {
                    'question': ['What is 12+8?', 'What is 15-7?', 'What is 6*4?', 'What is 20/5?'],
                    'answer': ['20', '8', '24', '4'],
                    'split': ['train', 'train', 'test', 'test']
                }
            },
            # Test split few-shot with tuple
            {
                'name': 'Few-shot with Test Split (Tuple)',
                'template': '{instruction:paraphrase}\n\n{few_shot:test(2)}\n\nQuestion: {question:surface}\nAnswer: {answer}',
                'description': 'Few-shot with 2 examples from test split, same for all rows (tuple format)',
                'sample_data': {
                    'question': ['What is 12+8?', 'What is 15-7?', 'What is 6*4?', 'What is 20/5?'],
                    'answer': ['20', '8', '24', '4'],
                    'split': ['train', 'train', 'test', 'test']
                }
            },
            # Text Classification with Context
            {
                'name': 'Text Classification',
                'template': '{instruction:paraphrase}:\n\nText: "{text:surface}"\nContext: {context:context}\nCategory: {category}',
                'description': 'Text classification with paraphrase instructions, surface text variations, and context variations',
                'sample_data': {
                    'text': ['Book a flight to Paris', 'Cancel my subscription'],
                    'category': ['travel', 'service'],
                    'context': ['Travel booking', 'Customer service']
                }
            },
            # Complex few-shot with multiple fields
            {
                'name': 'Complex Few-shot Classification',
                'template': '{instruction:paraphrase}\n\n{few_shot:[3]}\n\nText: "{text:surface}"\nSentiment: {label}',
                'description': 'Complex example combining few-shot examples with text classification and surface variations',
                'sample_data': {
                    'text': ['I absolutely love this product!', 'This is the worst service ever!', 'It\'s okay, nothing special', 'Amazing quality and fast delivery!'],
                    'label': ['positive', 'negative', 'neutral', 'positive']
                }
            }
        ]
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # If in debug mode and starting from a specific step, load demo data
    if debug_mode and start_step > 1 and 'loaded_demo_data' not in st.session_state:
        st.session_state.loaded_demo_data = True
        load_demo_data_for_step(start_step)


def render_current_page(current_page):
    """Render the appropriate page based on the current state"""
    pages = {
        1: upload_data.render,
        2: template_builder.render,
        3: generate_variations.render,
        4: show_results.render
    }
    
    # Add navigation helper
    if current_page > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Previous Step"):
                st.session_state.page = current_page - 1
                st.rerun()
        with col3:
            # Show next button only if current step is complete
            show_next = False
            if current_page == 1 and st.session_state.get('data_loaded', False):
                show_next = True
            elif current_page == 2 and st.session_state.get('template_ready', False):
                show_next = True
            elif current_page == 3 and st.session_state.get('variations_generated', False):
                show_next = True
            
            if show_next and current_page < 4:
                if st.button("Next Step →"):
                    st.session_state.page = current_page + 1
                    st.rerun()
    
    # Call the render function for the current page
    if current_page in pages:
        pages[current_page]()
    else:
        st.error(f"Page {current_page} not found!")


if __name__ == '__main__':
    main()
