"""
Step 1: Upload or select data for MultiPromptify 2.0
"""

import streamlit as st
import pandas as pd
import json
import os


def render():
    """Render the data upload/selection interface"""
    st.markdown('<div class="step-header"><h2>📁 Step 1: Upload Your Data</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>What you need:</strong> A dataset in CSV, JSON, or pandas DataFrame format containing your prompt data.
        The data should have columns that you want to vary in your prompts (e.g., questions, context, examples).
    </div>
    """, unsafe_allow_html=True)
    
    # Show current data status if loaded
    if st.session_state.get('data_loaded', False):
        st.success(f"✅ Data loaded: {st.session_state.get('data_source', 'Unknown source')}")
        display_data_preview(st.session_state.uploaded_data)
        
        # Option to load different data
        if st.button("🔄 Load Different Data"):
            clear_data_state()
            st.rerun()
        
        # Continue button
        st.markdown("---")
        if st.button("Continue to Template Builder →", type="primary", key="continue_main"):
            st.session_state.page = 2
            st.rerun()
        return
    
    # Offer three options: upload file, use sample data, or create custom data
    tab1, tab2, tab3 = st.tabs(["📂 Upload File", "🎯 Sample Datasets", "✏️ Create Custom Data"])
    
    with tab1:
        upload_file_interface()
    
    with tab2:
        sample_datasets_interface()
    
    with tab3:
        create_custom_data_interface()


def clear_data_state():
    """Clear data-related session state"""
    keys_to_clear = ['uploaded_data', 'data_source', 'data_loaded', 'selected_sample']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def upload_file_interface():
    """Interface for uploading files"""
    st.subheader("Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'json'],
        help="Upload a CSV or JSON file containing your prompt data"
    )
    
    if uploaded_file is not None:
        try:
            # Determine file type and load accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame(data)
                else:
                    st.error("JSON file should contain either a list of objects or a dictionary with lists as values")
                    return
            
            # Validate and store the data
            if validate_and_store_data(df, f"uploaded file: {uploaded_file.name}"):
                st.success(f"✅ Successfully loaded {len(df)} rows from {uploaded_file.name}")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def sample_datasets_interface():
    """Interface for selecting sample datasets"""
    st.subheader("Choose a Sample Dataset")
    st.write("Perfect for getting started or testing MultiPromptify")
    
    sample_datasets = get_sample_datasets()
    
    # Create columns for sample dataset selection
    cols = st.columns(2)
    
    for i, (name, info) in enumerate(sample_datasets.items()):
        col = cols[i % 2]
        with col:
            with st.container():
                st.markdown(f"**{name}**")
                st.write(info['description'])
                st.write(f"📊 {len(info['data'])} rows, {len(info['data'].columns)} columns")
                
                # Use unique key for each button
                button_key = f"sample_btn_{name.lower().replace(' ', '_')}"
                if st.button(f"Use {name}", key=button_key):
                    try:
                        if validate_and_store_data(info['data'], f"sample dataset: {name}"):
                            st.session_state.selected_sample = name
                            st.success(f"✅ Selected {name} sample dataset")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error loading sample dataset: {str(e)}")


def get_sample_datasets():
    """Get sample datasets dictionary"""
    return {
        "Sentiment Analysis": {
            'data': pd.DataFrame({
                'text': [
                    'I absolutely love this product!', 
                    'This service is terrible and disappointing.', 
                    'The experience was okay, nothing special.',
                    'Outstanding quality and fast delivery!',
                    'Not impressed, could be much better.'
                ],
                'label': ['positive', 'negative', 'neutral', 'positive', 'negative']
            }),
            'description': 'Text sentiment classification dataset with positive, negative, and neutral labels'
        },
        "Question Answering": {
            'data': pd.DataFrame({
                'question': [
                    'What is the capital of France?', 
                    'How many days are in a week?',
                    'Who wrote Romeo and Juliet?',
                    'What is 12 plus 8?',
                    'What color is typically associated with the sky?',
                    'What is 15 minus 7?',
                    'What is 6 times 4?'
                ],
                'answer': ['Paris', '7', 'Shakespeare', '20', 'Blue', '8', '24'],
                'context': [
                    'Geography', 
                    'Time and calendar', 
                    'Literature',
                    'Mathematics',
                    'Nature',
                    'Mathematics',
                    'Mathematics'
                ],
                'split': ['train', 'train', 'train', 'train', 'test', 'test', 'test']
            }),
            'description': 'General knowledge Q&A dataset with context and train/test split'
        },
        "Multiple Choice": {
            'data': pd.DataFrame({
                'question': [
                    'What is the largest planet in our solar system?',
                    'Which chemical element has the symbol O?',
                    'What is the fastest land animal?',
                    'What is the smallest prime number?',
                    'Which continent is known as the "Dark Continent"?'
                ],
                'options': [
                    'Earth, Jupiter, Mars, Venus',
                    'Oxygen, Gold, Silver, Iron', 
                    'Lion, Cheetah, Horse, Leopard',
                    '1, 2, 3, 0',
                    'Asia, Africa, Europe, Australia'
                ],
                'answer': [1, 0, 1, 1, 1],  # Indices: Jupiter=1, Oxygen=0, Cheetah=1, 2=1, Africa=1
                'subject': ['Astronomy', 'Chemistry', 'Biology', 'Mathematics', 'Geography']
            }),
            'description': 'Multiple choice questions with options in comma-separated format, numeric answer indices, and subject categories'
        },
        "Text Classification": {
            'data': pd.DataFrame({
                'text': [
                    'Book a flight to Paris for next week',
                    'Cancel my subscription to the premium service',
                    'What is the weather forecast for tomorrow?',
                    'Set a reminder for my meeting at 3pm',
                    'Order two large pizzas for dinner delivery',
                    'Check my bank account balance',
                    'Find the nearest coffee shop location'
                ],
                'category': ['travel', 'service', 'information', 'productivity', 'food', 'banking', 'location'],
                'intent': ['booking', 'cancellation', 'query', 'scheduling', 'ordering', 'inquiry', 'search'],
                'context': [
                    'Travel booking', 
                    'Customer service', 
                    'Weather inquiry',
                    'Calendar management',
                    'Food ordering',
                    'Banking service',
                    'Local search'
                ]
            }),
            'description': 'Intent and category classification with context for various user requests'
        }
    }


def create_custom_data_interface():
    """Interface for creating custom data"""
    st.subheader("Create Custom Dataset")
    st.write("Build your own dataset by entering data manually")
    
    # Initialize session state for custom data creation
    if 'custom_data_config' not in st.session_state:
        st.session_state.custom_data_config = {
            'num_cols': 3,
            'num_rows': 3,
            'columns': [],
            'data': {}
        }
    
    # Number of columns and rows
    col1, col2 = st.columns(2)
    with col1:
        num_cols = st.number_input(
            "Number of columns", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.custom_data_config['num_cols'],
            key="custom_num_cols"
        )
    with col2:
        num_rows = st.number_input(
            "Number of rows", 
            min_value=1, 
            max_value=20, 
            value=st.session_state.custom_data_config['num_rows'],
            key="custom_num_rows"
        )
    
    # Update config if changed
    if (num_cols != st.session_state.custom_data_config['num_cols'] or 
        num_rows != st.session_state.custom_data_config['num_rows']):
        st.session_state.custom_data_config.update({
            'num_cols': num_cols,
            'num_rows': num_rows
        })
        st.rerun()
    
    # Column names
    st.subheader("Column Names")
    columns = []
    col_widgets = st.columns(min(num_cols, 3))
    
    for i in range(num_cols):
        col_idx = i % 3
        with col_widgets[col_idx]:
            col_name = st.text_input(
                f"Column {i+1}", 
                value=f"column_{i+1}", 
                key=f"col_name_{i}"
            )
            columns.append(col_name)
    
    # Data entry
    st.subheader("Enter Data")
    data = {}
    for col in columns:
        data[col] = []
    
    for row in range(num_rows):
        st.write(f"**Row {row + 1}:**")
        row_cols = st.columns(min(num_cols, 3))
        
        for i, col in enumerate(columns):
            col_idx = i % 3
            with row_cols[col_idx]:
                value = st.text_input(
                    f"{col}", 
                    key=f"data_{row}_{i}", 
                    value=""
                )
                data[col].append(value)
    
    # Create dataset button with better error handling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 Create Dataset", type="primary", use_container_width=True):
            try:
                # Filter out empty values
                filtered_data = {}
                max_len = 0
                
                for col, values in data.items():
                    non_empty = [v.strip() for v in values if v.strip()]
                    if non_empty:
                        filtered_data[col] = non_empty
                        max_len = max(max_len, len(non_empty))
                
                if not filtered_data or max_len == 0:
                    st.error("Please enter at least one row of data with non-empty values")
                    return
                
                # Pad shorter columns with empty strings
                for col in filtered_data:
                    while len(filtered_data[col]) < max_len:
                        filtered_data[col].append("")
                
                df = pd.DataFrame(filtered_data)
                if validate_and_store_data(df, "custom dataset"):
                    st.success(f"✅ Created custom dataset with {len(df)} rows")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error creating dataset: {str(e)}")


def validate_and_store_data(df, source_name):
    """Validate and store the data in session state"""
    try:
        if df is None or df.empty:
            st.error("Dataset is empty")
            return False
        
        if len(df.columns) == 0:
            st.error("Dataset has no columns")
            return False
        
        # Check for valid data
        if df.isnull().all().all():
            st.error("Dataset contains only empty values")
            return False
        
        # Store in session state
        st.session_state.uploaded_data = df
        st.session_state.data_source = source_name
        st.session_state.data_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"Error validating data: {str(e)}")
        return False


def display_data_preview(df):
    """Display a preview of the loaded data"""
    st.subheader("📊 Data Preview")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Data Source", st.session_state.get('data_source', 'Unknown'))
    
    # Column info
    st.write("**Columns:**", ", ".join(df.columns.tolist()))
    
    # Data preview
    st.dataframe(df, use_container_width=True)
    
    # Show data types
    with st.expander("📋 Column Information"):
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Sample Values': [', '.join(df[col].dropna().astype(str).head(3).tolist()) for col in df.columns]
        })
        st.dataframe(info_df, use_container_width=True) 