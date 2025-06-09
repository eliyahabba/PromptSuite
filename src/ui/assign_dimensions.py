# File: pages/assign_dimensions.py
import streamlit as st
import json
from src.utils.constants import DEFAULT_VARIATIONS_PER_AXIS, MIN_VARIATIONS_PER_AXIS, MAX_VARIATIONS_PER_AXIS


def render():
    st.title("Step 4: Assign Dimensions to Parts")

    with st.expander("ℹ️ What are dimensions?"):
        all_dimensions = st.session_state.base_dimensions + st.session_state.custom_dimensions
        st.json(all_dimensions)

    if "annotated_parts" not in st.session_state or not st.session_state.annotated_parts:
        st.warning("Please annotate prompt parts first.")
        return

    # Initialize state variables if they don't exist
    if "dimension_assignments" not in st.session_state:
        st.session_state.dimension_assignments = {}

    if "dimension_variant_counts" not in st.session_state:
        st.session_state.dimension_variant_counts = {}

    # Collect all unique parts across all prompts
    all_parts = {}
    for example_parts in st.session_state.annotated_parts.values():
        for part_key, text in example_parts["annotations"].items():
            if part_key not in all_parts:
                all_parts[part_key] = text

    # Get all dimension names and descriptions
    all_dimensions = st.session_state.base_dimensions + st.session_state.custom_dimensions
    all_dimension_names = [f"{d['name']} - {d['description']}" for d in all_dimensions]
    dimension_name_to_option = {d['name']: f"{d['name']} - {d['description']}" for d in all_dimensions}
    option_to_dimension_name = {f"{d['name']} - {d['description']}": d['name'] for d in all_dimensions}

    st.subheader("Assign Dimensions to Each Part")

    # Display each part with its dimensions
    for part_key, text in all_parts.items():
        st.markdown(f"### {part_key.replace('_', ' ').title()}")
        st.text_area("Example Text", value=text, disabled=True, key=f"text_preview_{part_key}")

        # Get previously selected dimensions for this part
        previously_selected = st.session_state.dimension_assignments.get(part_key, [])
        
        # Convert dimension names to options for multiselect
        # Select dimensions to vary using multiselect
        multiselect_key = f"dims_{part_key}"
        selected_dims = st.multiselect(
            "Select dimensions to vary:",
            options=all_dimension_names,
            key=multiselect_key
        )
        
        # Convert selected options back to dimension names
        selected_dims = [option_to_dimension_name[option] for option in selected_dims]
        
        # Check if both few-shot options are available
        few_shot_options = ["Which few-shot examples", "How many few-shot examples"]
        selected_few_shot = [dim for dim in selected_dims if dim in few_shot_options]
        
        # If one few-shot option is selected, show a hint about the other
        if len(selected_few_shot) == 1:
            other_option = [opt for opt in few_shot_options if opt not in selected_few_shot][0]
            st.info(f"💡 Tip: You can also select '{other_option}' to vary both aspects of few-shot examples.")
        
        # Update the dimension assignments
        st.session_state.dimension_assignments[part_key] = selected_dims

        # Show variant count inputs if dimensions are selected
        if selected_dims:
            st.markdown("##### Number of Variants per Dimension")

            # Create a more efficient column layout
            cols = st.columns(min(3, len(selected_dims)))

            # Setup counter to cycle through columns
            col_idx = 0

            # Make sure the part exists in the variant counts
            if part_key not in st.session_state.dimension_variant_counts:
                st.session_state.dimension_variant_counts[part_key] = {}

            # Display number inputs for each dimension
            for dim_name in selected_dims:
                # Initialize if needed
                if dim_name not in st.session_state.dimension_variant_counts[part_key]:
                    st.session_state.dimension_variant_counts[part_key][dim_name] = DEFAULT_VARIATIONS_PER_AXIS

                with cols[col_idx]:
                    # Display number input with current value
                    count_key = f"count_{part_key}_{dim_name.replace(' ', '_')}"

                    # Initialize the value in session state if it doesn't exist
                    if count_key not in st.session_state:
                        st.session_state[count_key] = st.session_state.dimension_variant_counts[part_key][dim_name]

                    # Display the number input
                    count_value = st.number_input(
                        f"{dim_name}",
                        min_value=MIN_VARIATIONS_PER_AXIS,
                        max_value=MAX_VARIATIONS_PER_AXIS,
                        value=st.session_state[count_key],
                        key=count_key
                    )

                    # Update the session state for dimension_variant_counts
                    st.session_state.dimension_variant_counts[part_key][dim_name] = count_value

                # Move to next column
                col_idx = (col_idx + 1) % len(cols)

        st.markdown("---")

    # Actions section - all buttons in a vertical sequence
    st.markdown("### Actions")

    # Save button
    if st.button("Save All Assignments to JSON"):
        save_assignments()

    # Create a small space between buttons
    st.markdown("")

    # Continue button
    if st.button("Continue to predict breakdown"):
        if "final_annotations_output" not in st.session_state:
            st.warning("Please save the annotations before proceeding.")
        else:
            st.session_state.page = 5
            st.rerun()


def save_assignments():
    """Extract and save the assignments to the session state"""
    output = []
    for i, parts in st.session_state.annotated_parts.items():
        only_annotations = parts["annotations"]
        entry = {}

        for part, text in only_annotations.items():
            entry[part] = {
                "text": text,
                "dimensions": st.session_state.dimension_assignments.get(part, []),
                "variant_counts": st.session_state.dimension_variant_counts.get(part, {})
            }

        parts["annotations"] = entry
        parts["costume_dimensions"] = st.session_state.custom_dimensions
        output.append(parts)

    st.session_state.final_annotations_output = output

    # Create download button
    json_str = json.dumps(output, indent=2)
    st.download_button(
        "Download JSON",
        data=json_str,
        file_name="final_annotations.json",
        mime="application/json"
    )

    st.success("Assignments saved successfully!")