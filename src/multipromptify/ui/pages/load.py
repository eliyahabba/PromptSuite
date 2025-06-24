import sys
from pathlib import Path

import streamlit as st

# Add the src directory to the path to import multipromptify
base_dir = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(base_dir))

from multipromptify.core import SHUFFLE_VARIATION
from multipromptify.ui.pages import (
    upload_data,
    template_builder,
    generate_variations
)
from multipromptify.ui.utils.progress_indicator import show_progress_indicator
from multipromptify.core.template_keys import (
    CONTEXT_KEY,
    PARAPHRASE_WITH_LLM,
    INSTRUCTION, PROMPT_FORMAT, PROMPT_FORMAT_VARIATIONS, INSTRUCTION_VARIATIONS, GOLD_KEY, FEW_SHOT_KEY, OPTIONS_KEY,
    QUESTION_KEY, FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, ENUMERATE_VARIATION
)


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

    # Debug mode removed for cleaner interface

    # Total number of pages in the simplified application
    total_pages = 3

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
                        'name': 'Basic Sentiment Analysis',
                        'template': {
                            INSTRUCTION: 'Classify the sentiment of the following text.',
                            PROMPT_FORMAT: 'Text: "{text}"\nSentiment: {label}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            'text': [TYPOS_AND_NOISE_VARIATION],
                            GOLD_KEY: {
                                'field': 'label',
                                'type': 'value'
                            }
                        },
                        'description': 'Sentiment classification with format structure and noise injection variations',
                        'sample_data': {
                            'text': ['I love this movie!', 'This book is terrible.', 'The weather is nice today.'],
                            'label': ['positive', 'negative', 'neutral']
                        }
                    },
                    {
                        'name': 'Advanced Sentiment with Few-shot',
                        'template': {
                            INSTRUCTION: 'Classify the sentiment of the following text.',
                            PROMPT_FORMAT: 'Text: "{text}"\nSentiment: {label}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            'text': [TYPOS_AND_NOISE_VARIATION, CONTEXT_KEY],
                            GOLD_KEY: {
                                'field': 'label',
                                'type': 'value'
                            },
                            FEW_SHOT_KEY: {
                                'count': 2,
                                'format': 'shared_first_n',
                                'split': 'all'
                            }
                        },
                        'description': 'Sentiment analysis with format structure, noise injection, and context variations',
                        'sample_data': {
                            'text': ['I absolutely love this product!', 'This is the worst service ever!',
                                     'It\'s okay, nothing special', 'Amazing quality!'],
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
                            INSTRUCTION: 'Please answer the following question.',
                            PROMPT_FORMAT: 'question: {question}\nanswer: {answer}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'value'
                            }
                        },
                        'description': 'Q&A with format structure and noise injection variations',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'How many days in a week?',
                                         'Who wrote Romeo and Juliet?'],
                            'answer': ['Paris', '7', 'Shakespeare']
                        }
                    },
                    {
                        'name': 'Q&A with Context and Few-shot',
                        'template': {
                            INSTRUCTION: 'Based on the context, answer the question.',
                            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
                            PROMPT_FORMAT: 'Context: {context}\nQuestion: {question}\nAnswer: {answer}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            CONTEXT_KEY: [CONTEXT_KEY],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'value'
                            },
                            FEW_SHOT_KEY: {
                                'count': 3,
                                'format': 'shared_first_n',
                                'split': 'all'
                            }
                        },
                        'description': 'Q&A with format structure, noise injection, and context variations',
                        'sample_data': {
                            'question': ['What is 12+8?', 'What is 15-7?', 'What is 6*4?', 'What is 20/5?',
                                         'What is 9*3?'],
                            'answer': ['20', '8', '24', '4', '27'],
                            'context': ['Mathematics', 'Arithmetic', 'Basic math', 'Numbers', 'Calculation']
                        }
                    }
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
                            INSTRUCTION: 'The following are multiple choice questions (with answers) about {subject}.',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            OPTIONS_KEY: [SHUFFLE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',  # 'index' or 'value'
                                'options_field': 'options'  # Field containing the list to shuffle
                            }
                        },
                        'description': 'Multiple choice with format structure, noise injection, and option shuffling',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?',
                                         'What is the fastest land animal?'],
                            'options': ['Mars, Earth, Jupiter, Venus', 'Oxygen, Gold, Silver', 'Lion, Cheetah, Horse'],
                            'answer': [2, 0, 1]  # Indices: Jupiter=2, Oxygen=0, Cheetah=1
                        }
                    },
                    {
                        'name': 'Multiple Choice with Paraphrased Instruction',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers) about {subject}.',
                            INSTRUCTION_VARIATIONS: [PARAPHRASE_WITH_LLM],
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            OPTIONS_KEY: [SHUFFLE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            }
                        },
                        'description': 'Multiple choice with format structure, noise injection, option shuffling, and LLM-paraphrased instruction',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?',
                                         'What is the fastest land animal?'],
                            'options': ['Mars, Earth, Jupiter, Venus', 'Oxygen, Gold, Silver', 'Lion, Cheetah, Horse'],
                            'answer': [2, 0, 1]
                        }
                    },
                    {
                        'name': 'Complex Multiple Choice with Few-shot',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers).',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            INSTRUCTION_VARIATIONS: [TYPOS_AND_NOISE_VARIATION],
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION],
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            OPTIONS_KEY: [SHUFFLE_VARIATION, TYPOS_AND_NOISE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            },
                            FEW_SHOT_KEY: {
                                'count': 2,
                                'format': 'shared_first_n',
                                'split': 'all'
                            }
                        },
                        'description': 'Multiple choice with format structure, noise injection, and option shuffling',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?',
                                         'What is the fastest land animal?', 'What is the smallest prime number?'],
                            'options': ['Mars, Earth, Jupiter, Venus', 'Oxygen, Gold, Silver', 'Lion, Cheetah, Horse',
                                        '1, 2, 3'],
                            'answer': [2, 0, 1, 1]  # Indices: Jupiter=2, Oxygen=0, Cheetah=1, 2=1
                        }
                    },
                    {
                        'name': 'Enumerated Multiple Choice',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers).',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            QUESTION_KEY: [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            },
                            'enumerate': {
                                'field': 'options',
                                'type': '1234'
                            }
                        },
                        'description': 'Multiple choice with format structure, noise injection, and automatic enumeration',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?',
                                         'What is the fastest land animal?'],
                            'options': ['Mars, Earth, Jupiter, Venus', 'Oxygen, Gold, Silver, Hydrogen',
                                        'Lion, Cheetah, Horse, Tiger'],
                            'answer': [2, 0, 1]  # Indices: Jupiter=2, Oxygen=0, Cheetah=1
                        }
                    },
                    {
                        'name': 'Lettered Multiple Choice with Enumerate',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers).',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            OPTIONS_KEY: [SHUFFLE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            },
                            'enumerate': {
                                'field': 'options',
                                'type': 'ABCD'
                            }
                        },
                        'description': 'Multiple choice with format structure, noise injection, and letter enumeration',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?',
                                         'What is the fastest land animal?'],
                            'options': ['Mars, Earth, Jupiter, Venus', 'Oxygen, Gold, Silver, Hydrogen',
                                        'Lion, Cheetah, Horse, Tiger'],
                            'answer': [2, 0, 1]  # Indices: Jupiter=2, Oxygen=0, Cheetah=1
                        }
                    },
                    {
                        'name': 'Multiple Choice with Enumerate Only',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            OPTIONS_KEY: [SHUFFLE_VARIATION, ENUMERATE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            }
                        },
                        'description': 'Multiple choice with option shuffling and automatic enumeration (ENUMERATE_VARIATION as field variation)',
                        'sample_data': {
                            'question': ['What is the largest planet?', 'Which element has symbol O?',
                                         'What is the fastest land animal?'],
                            'options': ['Mars, Earth, Jupiter, Venus', 'Oxygen, Gold, Silver', 'Lion, Cheetah, Horse'],
                            'answer': [2, 0, 1]
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
                            INSTRUCTION: 'Classify the following text into a category.',
                            PROMPT_FORMAT: 'Text: "{text}"\nCategory: {category}',
                            'text': [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION],
                            GOLD_KEY: {
                                'field': 'category',
                                'type': 'value'
                            }
                        },
                        'description': 'Text classification with format structure and noise injection variations',
                        'sample_data': {
                            'text': ['Book a flight to Paris', 'Cancel my subscription', 'What is the weather today?'],
                            'category': ['travel', 'service', 'information']
                        }
                    },
                    {
                        'name': 'Multi-field Text Classification',
                        'template': {
                            INSTRUCTION: 'Classify the following text.',
                            PROMPT_FORMAT: 'Text: "{text}"\nCategory: {category}',
                            'text': [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, CONTEXT_KEY],
                            'category': [],  # No variations for output
                            GOLD_KEY: {
                                'field': 'category',
                                'type': 'value'
                            }
                        },
                        'description': 'Text classification with format structure, noise injection, and context variations',
                        'sample_data': {
                            'text': ['Book a flight to Paris', 'Cancel my subscription', 'What is the weather today?'],
                            'category': ['travel', 'service', 'information']
                        }
                    },
                    {
                        'name': 'Text Classification with Few-shot',
                        'template': {
                            INSTRUCTION: 'Classify the following text.',
                            PROMPT_FORMAT: 'Text: "{text}"\nCategory: {category}',
                            'text': [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION],
                            GOLD_KEY: {
                                'field': 'category',
                                'type': 'value'
                            },
                            FEW_SHOT_KEY: {
                                'count': 3,
                                'format': 'shared_first_n',
                                'split': 'all'
                            }
                        },
                        'description': 'Text classification with format structure, noise injection, and rotating few-shot examples',
                        'sample_data': {
                            'text': ['Book a flight to Paris', 'Cancel my subscription', 'What is the weather today?',
                                     'Order pizza for dinner', 'Check my account balance'],
                            'category': ['travel', 'service', 'information', 'food', 'banking']
                        }
                    }
                ]
            },

            # Specialized Augmenters Templates
            'specialized_augmenters': {
                'category_name': 'Specialized Augmenters',
                'description': 'Templates showcasing the new specialized augmenters for format structure and noise injection',
                'templates': [
                    {
                        'name': 'Format Structure Only',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            OPTIONS_KEY: [ENUMERATE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            }
                        },
                        'description': 'Semantic-preserving format structure variations with automatic enumeration',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'What is 2+2?'],
                            'options': ['London, Berlin, Paris, Madrid', '3, 4, 5, 6'],
                            'answer': [2, 1]  # 0-based indices
                        }
                    },
                    {
                        'name': 'Noise Injection Only',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            OPTIONS_KEY: [TYPOS_AND_NOISE_VARIATION, ENUMERATE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            }
                        },
                        'description': 'Robustness testing with noise injection (typos, case changes, etc.)',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'What is 2+2?'],
                            'options': ['London, Berlin, Paris, Madrid', '3, 4, 5, 6'],
                            'answer': [2, 1]  # 0-based indices
                        }
                    },
                    {
                        'name': 'Combined Specialized Augmenters',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            OPTIONS_KEY: [TYPOS_AND_NOISE_VARIATION, ENUMERATE_VARIATION],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            }
                        },
                        'description': 'Both format structure and noise injection augmenters combined',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'What is 2+2?'],
                            'options': ['London, Berlin, Paris, Madrid', '3, 4, 5, 6'],
                            'answer': [2, 1]  # 0-based indices
                        }
                    },
                    {
                        'name': 'Noise Injection with Enumerate',
                        'template': {
                            INSTRUCTION: 'The following are multiple choice questions (with answers) about general knowledge.',
                            PROMPT_FORMAT: 'Question: {question}\nOptions: {options}\nAnswer: {answer}',
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],  # Noise injection for robustness testing
                            OPTIONS_KEY: [TYPOS_AND_NOISE_VARIATION, ENUMERATE_VARIATION],
                            # Noise injection + enumerate
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'index',
                                'options_field': 'options'
                            }
                        },
                        'description': 'Noise injection for robustness testing with automatic enumeration',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'What is 2+2?'],
                            'options': ['London, Berlin, Paris, Madrid', '3, 4, 5, 6'],
                            'answer': [2, 1]  # 0-based indices
                        }
                    }
                ]
            },

            # Advanced Templates
            'advanced_examples': {
                'category_name': 'Advanced Examples',
                'description': 'Complex templates showcasing advanced features',
                'templates': [
                    {
                        'name': 'Multi-Variation Classification',
                        'template': {
                            INSTRUCTION: 'Classify the sentiment of the following text.',
                            PROMPT_FORMAT: 'Text: "{text}"\nLabel: {label}',
                            'text': [FORMAT_STRUCTURE_VARIATION, TYPOS_AND_NOISE_VARIATION, CONTEXT_KEY],
                            'label': [],
                            GOLD_KEY: {
                                'field': 'label',
                                'type': 'value'
                            },
                            FEW_SHOT_KEY: {
                                'count': 2,
                                'format': 'shared_first_n',
                                'split': 'all'
                            }
                        },
                        'description': 'Multiple variations per field with format structure, noise injection, and context',
                        'sample_data': {
                            'text': ['I love this movie!', 'This book is terrible.', 'The weather is nice today.'],
                            'label': ['positive', 'negative', 'neutral']
                        }
                    },
                    {
                        'name': 'Complex Q&A with Fixed Examples',
                        'template': {
                            INSTRUCTION: 'Answer the following question.',
                            PROMPT_FORMAT: 'Question: {question}\nAnswer: {answer}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION, PARAPHRASE_WITH_LLM],
                            'answer': [],
                            GOLD_KEY: {
                                'field': 'answer',
                                'type': 'value'
                            },
                            FEW_SHOT_KEY: {
                                'count': 3,
                                'format': 'shared_first_n',
                                'split': 'train'
                            }
                        },
                        'description': 'Q&A with format structure, noise injection, and paraphrase variations',
                        'sample_data': {
                            'question': ['What is the capital of France?', 'How many days in a week?',
                                         'Who wrote Romeo and Juliet?'],
                            'answer': ['Paris', '7', 'Shakespeare']
                        }
                    }
                ]
            },

            # Add system prompt support to template suggestions and session state
            INSTRUCTION: {
                'category_name': 'System Prompt',
                'description': 'Templates with a system prompt for the first few-shot example',
                'templates': [
                    {
                        'name': 'Math QA with System Prompt',
                        'template': {
                            INSTRUCTION: 'You are a helpful math assistant. Answer clearly.',
                            PROMPT_FORMAT: 'Question: {question}\nAnswer: {answer}',
                            PROMPT_FORMAT_VARIATIONS: [FORMAT_STRUCTURE_VARIATION],
                            QUESTION_KEY: [TYPOS_AND_NOISE_VARIATION],
                            GOLD_KEY: 'answer',
                            FEW_SHOT_KEY: {
                                'count': 2,
                                'format': 'shared_first_n',
                                'split': 'all'
                            }
                        },
                        'description': 'Few-shot QA with format structure, noise injection, and system prompt',
                        'sample_data': {
                            'question': ['What is 2+2?', 'What is 3*3?', 'What is 5+3?'],
                            'answer': ['4', '9', '8']
                        }
                    }
                ]
            }
        }
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Debug mode functionality removed for cleaner user experience


def render_current_page(current_page):
    """Render the appropriate page based on the current state"""
    pages = {
        1: upload_data.render,
        2: template_builder.render,
        3: generate_variations.render
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

            if show_next and current_page < 3:
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
