# File: pages/annotate_prompt.py
import streamlit as st
import pandas as pd

def render():
    st.title("Step 2: Annotate Prompt Parts")

    if 'csv_data' not in st.session_state or st.session_state.csv_data is None:
        st.warning("Please upload a CSV first.")
        return

    # Initialize the number of annotations if not already set
    if 'num_annotations' not in st.session_state:
        st.session_state.num_annotations = 1
    
    # Initialize current example index if not set
    if 'current_example_index' not in st.session_state:
        st.session_state.current_example_index = 0
        
    # Initialize annotated_parts dictionary if not set
    if "annotated_parts" not in st.session_state:
        st.session_state.annotated_parts = {}
        
    # Initialize completed_examples set if not set
    if "completed_examples" not in st.session_state:
        st.session_state.completed_examples = set()

    # Add navigation controls at the top
    st.subheader("Navigation")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_example_index > 0 and st.button("◀️ Previous"):
            st.session_state.current_example_index -= 1
            st.rerun()
            
    with col3:
        if st.session_state.current_example_index < st.session_state.num_annotations - 1 and st.button("Next ▶️"):
            st.session_state.current_example_index += 1
            st.rerun()
    
    with col2:
        # Number input for selecting number of examples to annotate
        max_examples = min(10, len(st.session_state.csv_data))  # Limit to 10 or dataset size
        num_examples = st.number_input(
            "Number of examples to annotate:", 
            min_value=1, 
            max_value=max_examples,
            value=st.session_state.num_annotations,
            step=1,
            key="num_examples_input"
        )
        
        # If number of examples changed
        if num_examples != st.session_state.num_annotations:
            old_num = st.session_state.num_annotations
            st.session_state.num_annotations = num_examples
            
            # If increased, sample more examples
            if num_examples > old_num:
                # Sample new data if needed
                if 'csv_data_sampled' not in st.session_state:
                    st.session_state.csv_data_sampled = st.session_state.csv_data.sample(
                        num_examples, random_state=1)
                # Or if we need more samples
                elif len(st.session_state.csv_data_sampled) < num_examples:
                    # Keep existing samples and add new ones
                    existing_indices = st.session_state.csv_data_sampled.index
                    remaining_data = st.session_state.csv_data[~st.session_state.csv_data.index.isin(existing_indices)]
                    additional_samples = remaining_data.sample(num_examples - old_num, random_state=1)
                    st.session_state.csv_data_sampled = pd.concat([st.session_state.csv_data_sampled, additional_samples])
            
            # If decreased, make sure current index is valid
            if st.session_state.current_example_index >= num_examples:
                st.session_state.current_example_index = num_examples - 1
                
            # Remove completed status for examples that no longer exist
            st.session_state.completed_examples = {
                idx for idx in st.session_state.completed_examples 
                if idx < num_examples
            }
                
            st.rerun()
    
    # Sample data if not already done
    if 'csv_data_sampled' not in st.session_state:
        st.session_state.csv_data_sampled = st.session_state.csv_data.sample(
            st.session_state.num_annotations, random_state=1)
    
    # Get current example index
    idx = st.session_state.current_example_index
    
    # Show progress indicator
    completed_count = len(st.session_state.completed_examples)
    total_count = st.session_state.num_annotations
    st.progress(completed_count / total_count)
    st.write(f"Completed: {completed_count}/{total_count} examples")
    
    # Update prompt based on current index
    prompt = st.session_state.csv_data_sampled['prompt'].iloc[idx]

    # Add a visual indicator if this example is completed
    if idx in st.session_state.completed_examples:
        st.header(f"Prompt {idx + 1}/{st.session_state.num_annotations} ✅")
    else:
        st.header(f"Prompt {idx + 1}/{st.session_state.num_annotations}")
        
    st.text_area("Prompt", value=prompt, height=200, disabled=True)

    predefined_parts = ['Task Description', 'Context', 'Examples']
    parts = {}

    st.subheader("Annotate Prompt Parts")
    for part in predefined_parts:
        part_key = part.lower().replace(" ", "_")
        # Use the current index in the key to ensure unique state per example
        # Check if we already have a value for this part
        default_value = ""
        if idx in st.session_state.annotated_parts and part_key in st.session_state.annotated_parts[idx]:
            default_value = st.session_state.annotated_parts[idx][part_key]
            
        text = st.text_area(f"{part} Text", value=default_value, key=f"{idx}_text_{part_key}")
        parts[part_key] = text

    # Custom sections
    st.subheader("Custom Sections")
    if "custom_parts" not in st.session_state:
        st.session_state.custom_parts = []

    new_part = st.text_input("Add a new custom section", key="new_custom_part")
    if st.button("Add Section") and new_part.strip():
        st.session_state.custom_parts.append(new_part.strip())

    for custom in st.session_state.custom_parts:
        custom_key = custom.lower().replace(" ", "_")
        # Check if we already have a value for this custom part
        default_value = ""
        if idx in st.session_state.annotated_parts and custom_key in st.session_state.annotated_parts[idx]:
            default_value = st.session_state.annotated_parts[idx][custom_key]
            
        text = st.text_area(f"{custom} Text", value=default_value, key=f"{idx}_text_{custom_key}")
        parts[custom_key] = text

    # Save button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Example"):
            # Save prompt parts to session state
            st.session_state.annotated_parts[idx] = parts
            # Mark as completed
            st.session_state.completed_examples.add(idx)
            st.success(f"Example {idx + 1} saved successfully!")
            # Auto-advance to next example if not on the last one
            if idx < st.session_state.num_annotations - 1:
                st.session_state.current_example_index += 1
                st.rerun()
    
    # Bottom navigation - only show "Move to next part" when all examples are completed
    if len(st.session_state.completed_examples) == st.session_state.num_annotations:
        with col2:
            if st.button("Move to next part"):
                st.session_state.page = 3
                st.session_state.current_example_index = 0
                for i in range(st.session_state.num_annotations):
                    parts = st.session_state.annotated_parts[i]
                    full_prompt = st.session_state.csv_data_sampled['prompt'].iloc[i]
                    placeholder_prompt = full_prompt
                    for part in parts:
                        if len(parts[part]) == 0:
                            continue
                        placeholder_prompt = placeholder_prompt.replace(parts[part], "{"+part.upper()+"}")
                    st.session_state.annotated_parts[i] = {"full_prompt": full_prompt,
                                                           "placeholder_prompt": placeholder_prompt,
                                                           "annotations": parts}
                print(st.session_state.annotated_parts)
                st.rerun()
    else:
        # Show warning if not all examples are completed
        st.warning(f"Please complete all {st.session_state.num_annotations} examples before proceeding.")
