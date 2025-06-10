import streamlit as st
import sys
from pathlib import Path
# Add the src directory to the path to import multipromptify
base_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(base_dir))

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
        'template_suggestions': {
            # Sentiment Analysis Templates
            'sentiment_analysis': {
                'category_name': 'Sentiment Analysis',
                'description': 'Templates for text sentiment classification tasks',
                'templates': [
                    {
                        'name': 'Basic Sentiment',
                        'template': {
                            'instruction': 'Classify the sentiment of the following text: "{text}"\nSentiment: {label}',
                            'template': '{instruction:paraphrase}'
                        },
                        'description': 'Simple sentiment classification with paraphrase instructions',
                        'sample_data': {
                            'text': ['I love this movie!', 'This book is terrible.', 'The weather is nice today.'],
                            'label': ['positive', 'negative', 'neutral']
                        }
                    },
                    {
                        'name': 'Sentiment with Surface Variations',
                        'template': {
                            'instruction': 'Classify the sentiment: "{text}"\nSentiment: {label}',
                            'template': '{instruction:paraphrase}\n{text:surface}'
                        },
                        'description': 'Sentiment classification with instruction paraphrase and text surface variations',
                        'sample_data': {
                            'text': ['I love this movie!', 'This book is terrible.', 'The weather is nice today.'],
                            'label': ['positive', 'negative', 'neutral']
                        }
                    },
                    {
                        'name': 'Sentiment with Few-shot (Different Examples)',
                        'template': {
                            'instruction': 'Classify the sentiment of the following text: "{text}"\nSentiment: {label}',
                            'template': '{instruction:paraphrase}\n{few_shot:[2]}'
                        },
                        'description': 'Sentiment classification with 2 different few-shot examples per row',
                        'sample_data': {
                            'text': ['I absolutely love this product!', 'This is the worst service ever!', 'It\'s okay, nothing special', 'Amazing quality!'],
                            'label': ['positive', 'negative', 'neutral', 'positive']
                        }
                    },
                    {
                        'name': 'Sentiment with Same Few-shot Examples',
                        'template': {
                            'instruction': 'Classify the sentiment of the following text: "{text}"\nSentiment: {label}',
                            'template': '{instruction:paraphrase}\n{few_shot:(2)}'
                        },
                        'description': 'Sentiment classification with 2 same few-shot examples for all rows',
                        'sample_data': {
                            'text': ['I absolutely love this product!', 'This is the worst service ever!', 'It\'s okay, nothing special', 'Amazing quality!'],
                            'label': ['positive', 'negative', 'neutral', 'positive']
                        }
                    }
                ]
            },
            
            # Question Answering Templates
            'question_answering': {
                'category_name': 'Question Answering',
                'description': 'Templates for question-answer tasks',
                'templates': [
                    {
                        'name': 'Basic Q&A',
                        'template': {
                            'instruction': 'Answer the following question: {question}\nAnswer: {answer}',
                            'template': '{instruction:paraphrase}'
                        },
                        'description': 'Simple Q&A with paraphrase instructions',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'How many days in a week?', 'Who wrote Romeo and Juliet?'],
                            'answer': ['Paris', '7', 'Shakespeare']
                        }
                    },
                    {
                        'name': 'Q&A with Surface Question Variations',
                        'template': {
                            'instruction': 'Answer the following question: {question}\nAnswer: {answer}',
                            'template': '{instruction:paraphrase}\n{question:surface}'
                        },
                        'description': 'Q&A with instruction paraphrase and surface question variations',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'How many days in a week?', 'Who wrote Romeo and Juliet?'],
                            'answer': ['Paris', '7', 'Shakespeare']
                        }
                    },
                    {
                        'name': 'Q&A with Context',
                        'template': {
                            'instruction': 'Based on the context, answer the question: Context: {context}\nQuestion: {question}\nAnswer: {answer}',
                            'template': '{instruction:paraphrase}\n{context:context}'
                        },
                        'description': 'Q&A with context variations and paraphrase instructions',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'How many days in a week?', 'Who wrote Romeo and Juliet?'],
                            'answer': ['Paris', '7', 'Shakespeare'],
                            'context': ['Geography', 'Time and calendar', 'Literature']
                        }
                    },
                    {
                        'name': 'Q&A with Few-shot Examples',
                        'template': {
                            'instruction': 'Answer the following question: {question}\nAnswer: {answer}',
                            'template': '{instruction:paraphrase}\n{few_shot:[3]}'
                        },
                        'description': 'Q&A with 3 different few-shot examples per row',
                        'sample_data': {
                            'question': ['What is 12+8?', 'What is 15-7?', 'What is 6*4?', 'What is 20/5?', 'What is 9*3?'],
                            'answer': ['20', '8', '24', '4', '27']
                        }
                    },
                ]
            },
            
            # Multiple Choice Templates
            'multiple_choice': {
                'category_name': 'Multiple Choice',
                'description': 'Templates for multiple choice question tasks',
                'templates': [
                    {
                        'name': 'Basic Multiple Choice',
                        'template': {
                            'instruction': 'Answer the following multiple choice question: Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            'template': '{instruction:paraphrase}'
                        },
                        'description': 'Multiple choice with paraphrase instructions',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?', 'What is the fastest land animal?'],
                            'options': ['A) Earth B) Jupiter C) Mars', 'A) Oxygen B) Gold C) Silver', 'A) Lion B) Cheetah C) Horse'],
                            'answer': ['B', 'A', 'B']
                        }
                    },
                    {
                        'name': 'Multiple Choice with Surface Variations',
                        'template': {
                            'instruction': 'Answer the following multiple choice question: Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            'template': '{instruction:paraphrase}\n{question:surface}\n{options:multiple-choice}'
                        },
                        'description': 'Multiple choice with instruction paraphrase, surface question variations, and choice formatting',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?', 'What is the fastest land animal?'],
                            'options': ['A) Earth B) Jupiter C) Mars', 'A) Oxygen B) Gold C) Silver', 'A) Lion B) Cheetah C) Horse'],
                            'answer': ['B', 'A', 'B']
                        }
                    },
                    {
                        'name': 'Multiple Choice with Subject',
                        'template': {
                            'instruction': 'Answer the multiple choice question in the following subject: Subject: {subject}\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}',
                            'template': '{instruction:paraphrase}\n{question:surface}'
                        },
                        'description': 'Multiple choice with subject context and surface question variations',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?', 'What is the fastest land animal?'],
                            'options': ['A) Earth B) Jupiter C) Mars', 'A) Oxygen B) Gold C) Silver', 'A) Lion B) Cheetah C) Horse'],
                            'answer': ['B', 'A', 'B'],
                            'subject': ['Astronomy', 'Chemistry', 'Biology']
                        }
                    },
                    {
                        'name': 'Multiple Choice with Few-shot',
                        'template': {
                            'instruction': 'Answer the following multiple choice question: Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            'template': '{instruction:paraphrase}\n{few_shot:(2)}'
                        },
                        'description': 'Multiple choice with 2 same few-shot examples for all rows',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?', 'What is the fastest land animal?', 'What is the smallest prime number?'],
                            'options': ['A) Earth B) Jupiter C) Mars', 'A) Oxygen B) Gold C) Silver', 'A) Lion B) Cheetah C) Horse', 'A) 1 B) 2 C) 3'],
                            'answer': ['B', 'A', 'B', 'B']
                        }
                    }
                ]
            },
            
            # Text Classification Templates
            'text_classification': {
                'category_name': 'Text Classification',
                'description': 'Templates for text classification and intent detection tasks',
                'templates': [
                    {
                        'name': 'Basic Text Classification',
                        'template': {
                            'instruction': 'Classify the following text into a category: Text: "{text}"\nCategory: {category}',
                            'template': '{instruction:paraphrase}'
                        },
                        'description': 'Simple text classification with paraphrase instructions',
                        'sample_data': {
                            'text': ['Book a flight to Paris', 'Cancel my subscription', 'What is the weather today?'],
                            'category': ['travel', 'service', 'information']
                        }
                    },
                    {
                        'name': 'Text Classification with Surface Variations',
                        'template': {
                            'instruction': 'Classify the following text: Text: "{text}"\nCategory: {category}',
                            'template': '{instruction:paraphrase}\n{text:surface}'
                        },
                        'description': 'Text classification with instruction paraphrase and surface text variations',
                        'sample_data': {
                            'text': ['Book a flight to Paris', 'Cancel my subscription', 'What is the weather today?'],
                            'category': ['travel', 'service', 'information']
                        }
                    },
                    {
                        'name': 'Text Classification with Intent',
                        'template': {
                            'instruction': 'Classify the text and identify its intent: Text: "{text}"\nCategory: {category}\nIntent: {intent}',
                            'template': '{instruction:paraphrase}\n{text:surface}'
                        },
                        'description': 'Text classification with both category and intent fields',
                        'sample_data': {
                            'text': ['Book a flight to Paris', 'Cancel my subscription', 'What is the weather today?', 'Set a reminder for 3pm'],
                            'category': ['travel', 'service', 'information', 'productivity'],
                            'intent': ['booking', 'cancellation', 'query', 'scheduling']
                        }
                    },
                    {
                        'name': 'Text Classification with Context',
                        'template': {
                            'instruction': 'Classify the text based on the context: Text: "{text}"\nContext: {context}\nCategory: {category}',
                            'template': '{instruction:paraphrase}\n{context:context}'
                        },
                        'description': 'Text classification with context variations and paraphrase instructions',
                        'sample_data': {
                            'text': ['Book a flight to Paris', 'Cancel my subscription', 'What is the weather today?'],
                            'category': ['travel', 'service', 'information'],
                            'context': ['Travel booking', 'Customer service', 'Weather inquiry']
                        }
                    },
                    {
                        'name': 'Text Classification with Few-shot',
                        'template': {
                            'instruction': 'Classify the following text: Text: "{text}"\nCategory: {category}',
                            'template': '{instruction:paraphrase}\n{few_shot:[3]}'
                        },
                        'description': 'Text classification with 3 different few-shot examples per row',
                        'sample_data': {
                            'text': ['Book a flight to Paris', 'Cancel my subscription', 'What is the weather today?', 'Order pizza for dinner', 'Check my account balance'],
                            'category': ['travel', 'service', 'information', 'food', 'banking']
                        }
                    }
                ]
            }
        }
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
